#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from dataset.ham import ATOMS
from dataset.ham import MASK_ATOM_INDEX
NUM_ATOMS = len(ATOMS)


class DSGPM(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, args):
        super(DSGPM, self).__init__()
        self.args = args
        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.input_fc, self.nn_conv, self.gru, self.output_fc = self.build_nnconv_layers(input_dim, hidden_dim,
                                                                                         embedding_dim,
                                                                                         layer=gnn.NNConv)

    def build_nnconv_layers(self, input_dim, hidden_dim, embedding_dim, layer=gnn.NNConv):
        if self.args.use_mask_embed:
            input_fc = nn.Embedding(input_dim + 1, hidden_dim, padding_idx=MASK_ATOM_INDEX)
        else:
            input_fc = nn.Embedding(input_dim, hidden_dim)
        if self.args.use_degree_feat:
            hidden_dim += 1
        if self.args.use_cycle_feat:
            hidden_dim += 1
        edge_nn = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim*hidden_dim)
        )
        nn_conv = layer(hidden_dim, hidden_dim, edge_nn, aggr='add')
        gru = nn.GRU(hidden_dim, hidden_dim)
        output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim)
        )
        return input_fc, nn_conv, gru, output_fc

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr
        if self.args.use_cycle_feat or self.args.use_degree_feat:
            out = F.relu(self.input_fc(x)).squeeze(1)
            out = torch.cat([out, data.degree_or_cycle_feat], dim=1)
        else:
            out = F.relu(self.input_fc(x)).squeeze(1)
        h = out.unsqueeze(0)

        for i in range(self.args.num_nn_iter):
            m = F.relu(self.nn_conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.output_fc(out)

        if self.args.use_mask_embed:
            atom_types_tensor = torch.zeros((x.shape[0], len(ATOMS) + 1), device=x.device)
        else:
            atom_types_tensor = torch.zeros((x.shape[0], len(ATOMS)), device=x.device)
        atom_types_tensor.scatter_(1, x, 1)

        feat_lst = [out, atom_types_tensor]
        if self.args.use_cycle_feat or self.args.use_degree_feat:
            feat_lst.append(data.degree_or_cycle_feat)
        out = torch.cat(feat_lst, dim=1)
        fg_embed = F.normalize(out)

        return fg_embed
