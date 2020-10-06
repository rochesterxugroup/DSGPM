#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import torch
import random
import tqdm
import dataset
import glob
import os
import pickle

from option import arg_parse
from torch_geometric.data import DataListLoader
from model.networks import DSGPM
from torch.utils.data.sampler import SubsetRandomSampler
from utils.stat import AverageMeter, FoldElementMat
from utils.post_processing import enforce_connectivity, edge_cut_prec_recall_fscore
from sklearn import metrics
from model.graph_cuts import graph_cuts

from warnings import simplefilter
from sklearn.exceptions import UndefinedMetricWarning
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UndefinedMetricWarning)


def eval(fold, epoch, test_dataloader, model, args, fold_element_matrix_manager):
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

            fold_element_matrix_manager.update(fold,
                                               data.graph_label.item(),
                                               {'ami': best_adjusted_mutual_info,
                                                'cut_prec': best_precision,
                                                'cut_recall': best_recall,
                                                'cut_fscore': best_f_score})

            adjusted_mutual_info_meter.update(best_adjusted_mutual_info)
            edge_cut_precision_meter.update(best_precision)
            edge_cut_recall_meter.update(best_recall)
            edge_cut_f_score_meter.update(best_f_score)
        tbar.set_description('fold:{} [{}/{}]: AMI: {:.4f}, prec: {:.4f}, '
                             'recall: {:.4f}, fscore: {:.4f}'.format(fold+1, epoch, args.epoch, adjusted_mutual_info_meter.avg, edge_cut_precision_meter.avg, edge_cut_recall_meter.avg, edge_cut_f_score_meter.avg))


def main():
    args = arg_parse()
    assert args.ckpt is not None
    assert args.ckpt_suffix is not None, '--ckpt_suffix is required'
    assert args.epoch is not None
    assert args.per_element_save_dir is not None
    args.devices = [int(device_id) for device_id in args.devices.split(',')]

    test_set = dataset.get_dataset_class(args.dataset)(data_root=args.data_root, split='test', cycle_feat=args.use_cycle_feat, degree_feat=args.use_degree_feat, cross_validation=True, automorphism=not args.debug, add_graph_label=True)

    indices = list(range(len(test_set)))
    random.shuffle(indices)

    test_set_len = int(len(test_set) / args.fold)

    fold_element_matrix_manager = FoldElementMat(args.fold, test_set.ATOMS,
                                               'ami', 'cut_prec', 'cut_recall', 'cut_fscore')

    ckpt_folder_lst = glob.glob(os.path.join(args.ckpt, '*{}'.format(args.ckpt_suffix)))
    ckpt_fpath_lst = [None] * args.fold

    for ckpt_folder in ckpt_folder_lst:
        pth_fpaths = glob.glob(os.path.join(ckpt_folder, '*fold_*_{}.pth'.format(args.epoch)))
        assert len(pth_fpaths) == 1
        pth_fpath = pth_fpaths[0]
        splits_of_folder_name = os.path.basename(pth_fpath).split('_')
        fold_idx = int(splits_of_folder_name[splits_of_folder_name.index('fold') + 1])
        assert ckpt_fpath_lst[fold_idx - 1] is None
        ckpt_fpath_lst[fold_idx - 1] = pth_fpath

    for idx_fold in range(args.fold):
        print('fold [{}/{}]:'.format(idx_fold + 1, args.fold))

        test_indices = indices[idx_fold * test_set_len : (idx_fold + 1) * test_set_len]
        test_sampler = SubsetRandomSampler(test_indices)
        test_dataloader = DataListLoader(test_set, batch_size=1, num_workers=0, sampler=test_sampler,
                                         pin_memory=True)

        model = DSGPM(args.num_atoms, args.hidden_dim,
                      args.output_dim, args=args).cuda()

        ckpt = torch.load(ckpt_fpath_lst[idx_fold])
        if 'iris' in args.ckpt_suffix:
            del ckpt['fc_pred_cg_beads_ratio.0.weight']
            del ckpt['fc_pred_cg_beads_ratio.0.bias']
            ckpt['input_fc.weight'] = ckpt['input_fc.weight'].t()
        model.load_state_dict(ckpt)

        e = args.epoch
        with torch.no_grad():
            eval(idx_fold, e, test_dataloader, model, args, fold_element_matrix_manager)

    print('[{}] cross validation result:'.format(args.fold))
    result = fold_element_matrix_manager.result()

    result_fpath = os.path.join(args.per_element_save_dir, '{}.pkl'.format(args.ckpt_suffix))
    with open(result_fpath, 'wb') as f:
        pickle.dump(result, f)
    print('result saved to {}'.format(result_fpath))


if __name__ == '__main__':
    main()
