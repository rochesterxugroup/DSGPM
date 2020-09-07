#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import os
import torch
import torch.optim as optim
import random
import tqdm

from option import arg_parse
from dataset.ham import HAM
from torch_geometric.data import DataLoader, DataListLoader
from model.networks import DSGPM
from model.losses import TripletLoss, PosPairMSE
from utils.util import get_run_name
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data.sampler import SubsetRandomSampler
from utils.stat import AverageMeter, FoldEpochMat
from utils.post_processing import enforce_connectivity, edge_cut_prec_recall_fscore
from sklearn import metrics
from model.graph_cuts import graph_cuts

from warnings import simplefilter
from sklearn.exceptions import UndefinedMetricWarning
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UndefinedMetricWarning)


def train(fold, epoch, train_loader, model, pos_pair_mse_criterion, triplet_criterion, optimizerG, args):
    model.train()
    triplet_loss_meter = AverageMeter()
    pos_pair_loss_meter = AverageMeter()

    train_loader = iter(train_loader)

    tbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True)

    for i, data in tbar:
        data = data.to(torch.device(0))
        model.zero_grad()

        fg_embed = model(data)

        loss = 0
        pos_pair_loss = args.pos_pair_weight * pos_pair_mse_criterion(fg_embed, data.pos_pair_index)
        loss += pos_pair_loss

        pos_pair_loss_meter.update(pos_pair_loss.item())

        if torch.numel(data.triplet_index) > 0:
            triplet_loss = args.triplet_weight * triplet_criterion(fg_embed, data.triplet_index)
            loss += triplet_loss
            triplet_loss_meter.update(triplet_loss.item())

        loss.backward()

        optimizerG.step()

        tbar.set_description('fold:%d [%d/%d] triplet: %.4f, pos_pair: %.4f'
                  % (fold+1, epoch, args.epoch, triplet_loss_meter.avg, pos_pair_loss_meter.avg))

    return triplet_loss_meter.avg, pos_pair_loss_meter.avg


def eval(fold, epoch, test_dataloader, model, args):
    model.eval()
    adjusted_mutual_info_meter = AverageMeter()
    edge_cut_precision_meter = AverageMeter()
    edge_cut_recall_meter = AverageMeter()
    edge_cut_f_score_meter = AverageMeter()

    tbar = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader), dynamic_ncols=True)
    for i, data in tbar:
        data = data[0]
        num_nodes = data.x.shape[0]
        data.batch = torch.zeros(num_nodes).long()
        data = data.to(torch.device(0))

        gt_hard_assigns = data.y.cpu().numpy()
        edge_index_cpu = data.edge_index.cpu().numpy()

        max_num_cg_beads = gt_hard_assigns.max(axis=1) + 1

        fg_embed = model(data)
        dense_adj = torch.sparse.LongTensor(data.edge_index, data.no_bond_edge_attr, (num_nodes, num_nodes)).to_dense()

        for _ in range(args.test_shots):
            best_adjusted_mutual_info, \
            best_precision, best_recall, best_f_score = -1, 0, 0, 0

            for anno_idx, gt_hard_assign in enumerate(gt_hard_assigns):
                hard_assign, _ = graph_cuts(fg_embed, data.edge_index, max_num_cg_beads[anno_idx], args.bandwidth)
                try:
                    hard_assign = enforce_connectivity(hard_assign, edge_index_cpu)
                except:
                    pass
                precision, recall, f_score = edge_cut_prec_recall_fscore(hard_assign, gt_hard_assign,
                                                                         edge_index_cpu)

                adjusted_mutual_info = metrics.adjusted_mutual_info_score(gt_hard_assign, hard_assign)

                best_adjusted_mutual_info = max(adjusted_mutual_info, best_adjusted_mutual_info)
                best_precision = max(precision, best_precision)
                best_recall = max(recall, best_recall)
                best_f_score = max(f_score, best_f_score)

            adjusted_mutual_info_meter.update(best_adjusted_mutual_info)
            edge_cut_precision_meter.update(best_precision)
            edge_cut_recall_meter.update(best_recall)
            edge_cut_f_score_meter.update(best_f_score)
        tbar.set_description('fold:{} [{}/{}]: AMI: {:.4f}, prec: {:.4f}, '
                             'recall: {:.4f}, fscore: {:.4f}'.format(fold+1, epoch, args.epoch, adjusted_mutual_info_meter.avg, edge_cut_precision_meter.avg, edge_cut_recall_meter.avg, edge_cut_f_score_meter.avg))

    return adjusted_mutual_info_meter.avg, edge_cut_precision_meter.avg, edge_cut_recall_meter.avg, edge_cut_f_score_meter.avg


def main():
    args = arg_parse()
    assert args.ckpt is not None, '--ckpt is required'
    args.devices = [int(device_id) for device_id in args.devices.split(',')]

    train_set = HAM(data_root=args.data_root, dataset_type='train', cycle_feat=args.use_cycle_feat, degree_feat=args.use_degree_feat, cross_validation=True, automorphism=True)
    test_set = HAM(data_root=args.data_root, dataset_type='test', cycle_feat=args.use_cycle_feat, degree_feat=args.use_degree_feat, cross_validation=True, automorphism=True)
    assert len(train_set) == len(test_set)

    indices = list(range(len(train_set)))
    random.shuffle(indices)

    test_set_len = int(len(train_set) / args.fold)

    fold_epoch_matrix_manager = FoldEpochMat(args.fold, args.epoch, ['ami', 'cut_fscore'],
                                             'ami', 'cut_prec', 'cut_recall', 'cut_fscore')

    for idx_fold in range(args.fold):
        print('fold [{}/{}]:'.format(idx_fold + 1, args.fold))

        test_indices = indices[idx_fold * test_set_len : (idx_fold + 1) * test_set_len]
        train_indices = list(set(indices) - set(test_indices))

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                                      num_workers=args.num_workers, pin_memory=True)
        test_dataloader = DataListLoader(test_set, batch_size=1, num_workers=0, sampler=test_sampler,
                                     pin_memory=True)

        model = DSGPM(args.input_dim, args.hidden_dim,
                      args.output_dim, args=args).cuda()

        pos_pair_mse_criterion = PosPairMSE().cuda()
        triplet_criterion = TripletLoss(args.margin).cuda()

        # setup optimizer
        optimizerG = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if not args.debug:
            run_name = get_run_name(args.title)

            ckpt_dir = os.path.join(args.ckpt, run_name)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            if args.tb_log:
                tensorboard_dir = os.path.join(args.tb_root, run_name)
                if not os.path.exists(tensorboard_dir):
                    os.mkdir(tensorboard_dir)

                writer = SummaryWriter(tensorboard_dir)
        else:
            writer = None
            ckpt_dir = None

        for e in range(1, args.epoch + 1):
            triplet_loss, pos_pair_loss = train(idx_fold, e, train_dataloader, model, pos_pair_mse_criterion,
                                                triplet_criterion, optimizerG, args)
            if not args.debug and args.tb_log:
                writer.add_scalar('triplet loss', triplet_loss, e)
                writer.add_scalar('pos pair loss', pos_pair_loss, e)

            if e % args.eval_interval == 0 and (args.start_eval_epoch is None or (e >= args.start_eval_epoch)):
                with torch.no_grad():
                    test_adjusted_mutual_info,\
                        test_edge_cut_prec, test_edge_cut_recall, test_edge_cut_f_score = eval(idx_fold, e, test_dataloader, model, args)

                fold_epoch_matrix_manager.update(idx_fold, e - 1, {'ami': test_adjusted_mutual_info,
                                                                   'cut_prec': test_edge_cut_prec,
                                                                   'cut_recall': test_edge_cut_recall,
                                                                   'cut_fscore': test_edge_cut_f_score})

                if not args.debug and args.tb_log:
                    writer.add_scalar('test_adjusted_mutual_info', test_adjusted_mutual_info, e)
                    writer.add_scalar('test_edge_cut_precision', test_edge_cut_prec, e)
                    writer.add_scalar('test_edge_cut_recall', test_edge_cut_recall, e)
                    writer.add_scalar('test_edge_cut_f_score', test_edge_cut_f_score, e)

                if not args.debug:
                    state_dict = model.module.state_dict() if not isinstance(model, DSGPM) else model.state_dict()
                    torch.save(state_dict, os.path.join(ckpt_dir, 'fold_{}_{}.pth'.format(idx_fold+1, e)))

        print('[{}/{}] cross validation result:'.format(idx_fold+1, args.fold))
        cv_mean, cv_std, best_epoch = fold_epoch_matrix_manager.result(idx_fold)
        print('best epoch: {}'.format(best_epoch))
        print('[mean] AMI: {:.4f}, prec: {:.4f}, recall: {:.4f}, fscore: {:.4f}'.format(cv_mean['ami'], cv_mean['cut_prec'], cv_mean['cut_recall'], cv_mean['cut_fscore']))
        print('[std] AMI: {:.4f}, prec: {:.4f}, recall: {:.4f}, fscore: {:.4f}'.format(cv_std['ami'], cv_std['cut_prec'], cv_std['cut_recall'], cv_std['cut_fscore']))


if __name__ == '__main__':
    main()
