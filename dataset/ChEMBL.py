#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

# ChEMBL

import os
import json
import torch
import numpy as np
import networkx as nx
import random
import torch.nn.functional as F

from torch.utils.data import Dataset
from networkx.algorithms.cycles import cycle_basis
from torch_geometric.data import Data
from . import BOND_TYPE_DICT


class ChEMBL(Dataset):
    ATOMS = ['B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'Zn', 'As', 'Se', 'Br', 'Te', 'I']
    FREQUENCY = {'C': 39563488, 'N': 6113034, 'O': 6278194, 'S': 711604, 'Cl': 406332, 'F': 700548, 'Br': 84706, 'P': 40590,
                 'Se': 2290, 'B': 3374, 'I': 12108, 'Si': 3269, 'Te': 103, 'As': 205, 'Zn': 4, 'Al': 12}

    def __init__(self, data_root, split_index_folder, sample_ratio=0.05, split='train', for_vis=False, cycle_feat=False, degree_feat=False, transform=None):
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.transform = transform
        split_file = os.path.join(split_index_folder, '{}_sample_ratio_{}.txt'.format(split, sample_ratio))
        assert os.path.exists(split_file), split_file
        with open(split_file) as f:
            json_fname_lst = f.readlines()
        json_fname_lst = [line.strip() for line in json_fname_lst]
        self.json_file_path_lst = [os.path.join(data_root, fname) for fname in json_fname_lst]
        self.for_vis = for_vis
        self.cycle_feat = cycle_feat
        self.degree_feat = degree_feat

    def __getitem__(self, index):
        """
        get index-th data
        :param index: index from lst
        :return: atom_types, fg_adj, cg_adj_gt, mapping_op_gt, atomic_weight
        """
        json_fpath = self.json_file_path_lst[index]
        with open(json_fpath) as f:
            json_data = json.load(f)
        """
            json_data format:
            dict {
                "cgnodes": [],  # [[fg_id...], [fg_id...]],
                "nodes": [
                            {
                                "cg":2,  # cg group_id (starts with 0)
                                "element":"C",  # atom type
                                "id":0  # fg id 
                            },
                            {...}
                         ],
                "edges": [
                            {
                                "source":0,  # from fg_id
                                "target":1   # to fg_id
                                "bondtype": 1.0  # bond type (1.0, 1.5, 2.0, 3.0)
                            }
                         ],
                "smiles": "C[SiH](C)O[Si](C)(CCl)F"
            }
        """
        data = Data()

        if 'smiles' not in json_data:
            smiles = os.path.splitext(os.path.basename(json_fpath))[0]
        else:
            smiles = json_data['smiles']
        smiles = smiles.replace('/', '\\')
        graph = nx.Graph(smiles=smiles)
        for node in json_data['nodes']:
            graph.add_node(node['id'], element=node['element'], cg=node['cg'])

        for edge in json_data['edges']:
            bond_type = edge['bondtype']
            if isinstance(bond_type, str):
                assert bond_type in set(BOND_TYPE_DICT.keys())
                bond_type = {'-': 1.0, '/': 1.0, '\\': 1.0, ':': 1.5, '=': 2.0, '#': 3.0}[bond_type]
            graph.add_edge(edge['source'], edge['target'], bond_type=bond_type)

        # ========== load atom types ==========
        fg_beads: list = json_data['nodes']
        fg_beads.sort(key=lambda x: x['id'])
        atom_types = torch.LongTensor([ChEMBL.ATOMS.index(bead['element']) for bead in fg_beads]).reshape(-1, 1)
        data.x = atom_types

        # ======== degree ===========
        if self.degree_feat:
            degrees = graph.degree
            degrees = np.array(degrees)[:, 1]
            degrees = torch.tensor(degrees).float().unsqueeze(dim=-1) / 4
            data.degree_or_cycle_feat = degrees

        # ========= cycles ==========
        if self.cycle_feat:
            cycle_indicator_per_node = torch.zeros(len(fg_beads)).unsqueeze(-1)
            cycle_lst = cycle_basis(graph)
            if len(cycle_lst) > 0:
                for idx_cycle, cycle in enumerate(cycle_lst):
                    cycle = torch.tensor(cycle)
                    cycle_indicator_per_node[cycle] = 1
            if hasattr(data, 'degree_or_cycle_feat'):
                data.degree_or_cycle_feat = torch.cat([data.degree_or_cycle_feat, cycle_indicator_per_node], dim=1)
            else:
                data.degree_or_cycle_feat = cycle_indicator_per_node

        edges = []
        bond_types = []
        for x in json_data['edges']:
            edges.append([x['source'], x['target']])
            edges.append([x['target'], x['source']])
            bond_types.append(BOND_TYPE_DICT[x['bondtype']])
            bond_types.append(BOND_TYPE_DICT[x['bondtype']])  # add bond types for both directions
        data.edge_index = torch.tensor(edges).long().t()
        data.no_bond_edge_attr = torch.ones(data.edge_index.shape[1])
        data.edge_attr = F.one_hot(torch.tensor(bond_types, dtype=torch.long), num_classes=4).float()  # hard code bond types to 4 types
        assert data.edge_attr.shape == (len(bond_types), 4)

        if self.for_vis or self.split == 'test':
            data.graph = graph

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.json_file_path_lst)

    @staticmethod
    def compute_cls_weight():
        freq_lst = [ChEMBL.FREQUENCY[a] for a in ChEMBL.ATOMS]
        total = sum(freq_lst)
        weight = total / (len(freq_lst) * torch.tensor(freq_lst, dtype=torch.float32))
        return weight
