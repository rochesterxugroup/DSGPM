#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import argparse
import random
import numpy as np
import torch

from dataset.ham import ATOMS


def arg_parse():
    parser = argparse.ArgumentParser('Deep Supervised Graph Partitioning Model (DSGPM)')
    parser.add_argument('--title', type=str, default='CG')
    parser.add_argument('--data_root', dest='data_root',
                        help='Directory where benchmark is located', required=True)
    parser.add_argument('--tb_root',
                        help='Tensorboard log directory')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--lr', dest='lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--epoch', dest='epoch', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
                        help='Number of workers to load data.')
    parser.add_argument('--input_dim', dest='input_dim', type=int,
                        help='Input feature dimension')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int,
                        help='Hidden dimension')
    parser.add_argument('--output_dim', dest='output_dim', type=int,
                        help='Output dimension')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--margin', default=1.0, type=float, help='margin in triplet loss')
    parser.add_argument('--entropy_weight', default=1.0, type=float)
    parser.add_argument('--adj_recons_weight', default=1.0, type=float)
    parser.add_argument('--triplet_weight', default=1.0, type=float)
    parser.add_argument('--pos_pair_weight', default=0.1, type=float)
    parser.add_argument('--balanced_cut_weight', default=0.0, type=float)
    parser.add_argument('--test_shots', type=int, default=2)
    parser.add_argument('--bandwidth', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--heads', type=int, default=8, help='number of multi-heads attention in GAT')
    parser.add_argument('--num_nn_iter', type=int, default=6)
    parser.add_argument('--no_cycle_feat', dest='use_cycle_feat', action='store_const',
                        const=False, default=True)
    parser.add_argument('--no_degree_feat', dest='use_degree_feat', action='store_const',
                        const=False, default=True)
    parser.add_argument('--pretrained_ckpt', type=str)
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--tb_log', action='store_true')
    parser.add_argument('--input_fc_bias', action='store_true')
    parser.add_argument('--start_eval_epoch', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_cg_beads', nargs='+', type=int, help='number of CG beads. E.g., --num_cg_beads 2 3 4')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--json_output_dir', type=str)
    parser.add_argument('--inference_method', choices=['dsgpm', 'spec_cluster', 'metis', 'graclus'], default='dsgpm')
    parser.add_argument('--svg', action='store_true')
    parser.add_argument('--vis_root', type=str)

    parser.set_defaults(cuda='0',
                        lr=1e-3,
                        batch_size=1,
                        epoch=500,
                        num_workers=0,
                        hidden_dim=128,
                        output_dim=128,
                        dropout=0.0)

    args = parser.parse_args()
    input_dim = len(ATOMS)
    args.input_dim = input_dim

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    return args