#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import os
import torch
import torch.optim as optim
import tqdm
import itertools

from option import arg_parse
from dataset.ham import HAM, ATOMS
from torch_geometric.data import DataLoader
from model.networks import DSGPM
from model.losses import TripletLoss, PosPairMSE
from utils.util import get_run_name
from torch.utils.tensorboard import SummaryWriter

from utils.stat import AverageMeter
from utils.transforms import MaskAtomType

from warnings import simplefilter
from sklearn.exceptions import UndefinedMetricWarning
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UndefinedMetricWarning)


class Trainer:
    def __init__(self, args):
        self.args = args
        train_set = HAM(data_root=args.data_root, dataset_type='train', cycle_feat=args.use_cycle_feat,
                        degree_feat=args.use_degree_feat, cross_validation=True, automorphism=not args.debug,
                        transform=MaskAtomType(args.mask_ratio))

        self.train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                      num_workers=args.num_workers, pin_memory=True)

        self.model = DSGPM(args.input_dim, args.hidden_dim,
                      args.output_dim, args=args).cuda()
        final_feat_dim = args.output_dim + len(ATOMS) + 1  # TODO confirm number of atom types
        if self.args.use_cycle_feat:
            final_feat_dim += 1
        if self.args.use_degree_feat:
            final_feat_dim += 1
        self.atom_type_classifier = torch.nn.Linear(final_feat_dim, len(ATOMS)).cuda()  # TODO confirm number of atom types
        self.criterion = torch.nn.CrossEntropyLoss()

        # setup optimizer
        self.optimizer = optim.Adam(itertools.chain(self.model.parameters(),
                                                    self.atom_type_classifier.parameters()),
                                    lr=args.lr, weight_decay=args.weight_decay)

        if not args.debug:
            run_name = get_run_name(args.title)

            self.ckpt_dir = os.path.join(args.ckpt, run_name)
            if not os.path.exists(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)

            if args.tb_log:
                tensorboard_dir = os.path.join(args.tb_root, run_name)
                if not os.path.exists(tensorboard_dir):
                    os.mkdir(tensorboard_dir)

                self.writer = SummaryWriter(tensorboard_dir)

    def train(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()

        train_loader = iter(self.train_loader)

        tbar = tqdm.tqdm(enumerate(train_loader), total=len(self.train_loader), dynamic_ncols=True)

        for i, data in tbar:
            data = data.to(torch.device(0))
            self.optimizer.zero_grad()

            fg_embed = self.model(data)
            pred = self.atom_type_classifier(fg_embed[data.masked_atom_index])
            loss = self.criterion(pred, data.masked_atom_type)
            loss.backward()
            self.optimizer.step()

            accuracy = float(torch.sum(torch.max(pred.detach(), dim=1)[1] == data.masked_atom_type).cpu().item()) / len(pred)
            loss_meter.update(loss.item())
            accuracy_meter.update(accuracy)

            tbar.set_description('[%d/%d] loss: %.4f, accuracy: %.4f'
                                 % (epoch, self.args.epoch, loss_meter.avg, accuracy_meter.avg))

        if not self.args.debug and self.args.tb_log:
            self.writer.add_scalar('loss', loss_meter.avg, epoch)
            self.writer.add_scalar('accuracy', accuracy_meter.avg, epoch)

        if not self.args.debug:
            state_dict = self.model.module.state_dict() if not isinstance(self.model, DSGPM) else self.model.state_dict()
            torch.save(state_dict, os.path.join(self.ckpt_dir, '{}.pth'.format(epoch)))


def main():
    args = arg_parse()
    args.use_mask_embed = True
    assert args.ckpt is not None, '--ckpt is required'
    args.devices = [int(device_id) for device_id in args.devices.split(',')]

    trainer = Trainer(args)
    for e in range(1, args.epoch + 1):
        trainer.train(e)


if __name__ == '__main__':
    main()
