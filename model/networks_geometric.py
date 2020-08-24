#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_scatter import scatter_mean

from dataset.ham import ATOMS
NUM_ATOMS = len(ATOMS)


class CGNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, args):
        super(CGNet, self).__init__()
        self.args = args
        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.input_fc, self.nn_conv, self.gru, self.output_fc = self.build_nnconv_layers(input_dim, hidden_dim,
                                                                                         embedding_dim,
                                                                                         layer=gnn.NNConv)
        if self.args.use_degree_feat:
            input_dim += 1
        if self.args.use_cycle_feat:
            input_dim += 1

        self.fc_pred_cg_beads_ratio = nn.Sequential(
            nn.Linear(embedding_dim + input_dim, 1),
            nn.Sigmoid()
        )

    def build_nnconv_layers(self, input_dim, hidden_dim, embedding_dim, layer=gnn.NNConv):
        input_fc = nn.Linear(input_dim, hidden_dim, bias=self.args.input_fc_bias)
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

    def nn_conv_forward(self, x, edge_index, edge_attr, batch):
        if self.args.use_cycle_feat or self.args.use_degree_feat:
            x, degree_or_cycle_feat = x[:, :NUM_ATOMS], x[:, NUM_ATOMS:]
            out = F.relu(self.input_fc(x))
            out = torch.cat([out, degree_or_cycle_feat], dim=1)
        else:
            out = F.relu(self.input_fc(x))
        h = out.unsqueeze(0)

        for i in range(self.args.num_nn_iter):
            m = F.relu(self.nn_conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.output_fc(out)
        feat_lst = [out, x]
        if self.args.use_cycle_feat or self.args.use_degree_feat:
            feat_lst.append(degree_or_cycle_feat)
        out = torch.cat(feat_lst, dim=1)
        out = F.normalize(out)

        readout = scatter_mean(out, batch, dim=0)
        cg_fg_ratio = self.fc_pred_cg_beads_ratio(readout)
        return out, cg_fg_ratio

    def forward(self, data):
        atom_types, edge_index = data.x, data.edge_index

        edge_attr = data.edge_attr
        fg_embed, cg_fg_ratio = self.nn_conv_forward(atom_types, edge_index, edge_attr, data.batch)

        return fg_embed, cg_fg_ratio
