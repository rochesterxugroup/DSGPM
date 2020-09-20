#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import os
import glob
import json
import torch
import numpy as np
import networkx as nx
import random
import torch.nn.functional as F

from collections import OrderedDict
from torch.utils.data import Dataset
from networkx.algorithms.cycles import cycle_basis
from torch_geometric.data import Data
from tqdm import tqdm
from utils.automorphism_group import node_equal, edge_equal
from . import BOND_TYPE_DICT

# no name for self supervised training
# ATOMS = OrderedDict([('B', 10.81), ('C', 12.011), ('N', 14.007), ('O', 15.999), ('F', 18.998403163), ('Si', 28.085), ('P', 30.973761998), ('S', 32.06), ('Cl', 35.45), ('K', 39.0983), ('Fe', 55.845), ('Se', 78.971), ('Br', 79.904), ('Ru', 101.07), ('Sn', 118.71), ('I', 126.90447), ('Al', 26.9815385), ('Zn', 65.38), ('As', 74.921595), ('Te', 127.6)])


class HAM(Dataset):
    ATOMS = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'K', 'Fe', 'Se', 'Br', 'Ru', 'Sn', 'I']

    def __init__(self, data_root, split='train', for_vis=False, cycle_feat=False, degree_feat=False, cross_validation=False, automorphism=True, transform=None):
        assert split in {'train', 'test'}
        self.split = split
        self.transform = transform
        if not cross_validation:
            jsons_root = os.path.join(data_root, split, '*.json')
        else:
            jsons_root = os.path.join(data_root, '*.json')
        self.json_file_path_lst = glob.glob(jsons_root)
        self.for_vis = for_vis
        self.cycle_feat = cycle_feat
        self.degree_feat = degree_feat

        self.smiles_cluster_idx_dict = {}
        self.smiles_json_fpath_lst = OrderedDict()
        for json_fpath in self.json_file_path_lst:
            with open(json_fpath) as f:
                json_data = json.load(f)
                if 'smiles' not in json_data:
                    smiles = os.path.splitext(os.path.basename(json_fpath))[0]
                else:
                    smiles = json_data['smiles']
                smiles = smiles.replace('/', '\\')
                if smiles not in self.smiles_cluster_idx_dict:
                    self.smiles_cluster_idx_dict[smiles] = []
                cluster_idx = self.compute_cluster_idx(json_data)
                self.smiles_cluster_idx_dict[smiles].append(cluster_idx.unsqueeze(0))

                if smiles not in self.smiles_json_fpath_lst:
                    self.smiles_json_fpath_lst[smiles] = []
                self.smiles_json_fpath_lst[smiles].append(json_fpath)

        self.smiles_lst = list(self.smiles_json_fpath_lst.keys())

        if automorphism:
            for smile in tqdm(self.smiles_lst, desc='automorphism'):
                json_fpath = self.smiles_json_fpath_lst[smile][0]
                with open(json_fpath) as f:
                    json_data = json.load(f)
                graph = nx.Graph()
                for node in json_data['nodes']:
                    graph.add_node(node['id'], element=node['element'], cg=node['cg'])

                for edge in json_data['edges']:
                    bond_type = edge['bondtype']
                    if isinstance(bond_type, str):
                        assert bond_type in set(BOND_TYPE_DICT.keys())
                        bond_type = {'-': 1.0, '/': 1.0, '\\': 1.0, ':': 1.5, '=': 2.0, '#': 3.0}[bond_type]
                    graph.add_edge(edge['source'], edge['target'], bond_type=bond_type)

                gm = nx.isomorphism.GraphMatcher(graph, graph,
                                                 node_match=node_equal,
                                                 edge_match=edge_equal)
                mapping_lst = []
                for node_mapping in gm.isomorphisms_iter():
                    key_value_lst = torch.tensor(list(node_mapping.items())).transpose(1, 0)
                    for original_mapping in self.smiles_cluster_idx_dict[smile]:
                        new_mapping = original_mapping.clone()
                        new_mapping[:, key_value_lst[0]] = new_mapping[:, key_value_lst[1]]
                        mapping_lst.append(new_mapping)
                self.smiles_cluster_idx_dict[smile] = mapping_lst

    def __getitem__(self, index):
        """
        get index-th data
        :param index: index from lst
        :return: atom_types, fg_adj, cg_adj_gt, mapping_op_gt, atomic_weight
        """
        json_fpaths = self.smiles_json_fpath_lst[self.smiles_lst[index]]
        json_fpath = json_fpaths[random.randrange(len(json_fpaths))]

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
        atom_types = torch.LongTensor([HAM.ATOMS.index(bead['element']) for bead in fg_beads]).reshape(-1, 1)
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

        # ========== load ground truth ==========
        assert len(atom_types) == len(json_data['nodes'])
        # if self.dataset_type == 'test' or self.for_vis:
        multi_anno_cluster_idx = self.smiles_cluster_idx_dict[smiles]
        multi_anno_cluster_idx = torch.cat(multi_anno_cluster_idx, dim=0)  # num_annotation x num_nodes
        if self.split == 'train':
            rand_anno_idx = random.randrange(len(multi_anno_cluster_idx))
            multi_anno_cluster_idx = multi_anno_cluster_idx[rand_anno_idx]
        data.y = multi_anno_cluster_idx

        if self.for_vis or self.split == 'test':
            data.graph = graph
            # remove triplet_idx in for vis
        elif self.split == 'train':
            # ========== triplet samples ==========
            # cg_node_indices_with_two_or_more_fg_nodes = [idx for idx, x in enumerate(json_data['cgnodes']) if len(x) > 1]
            # ========== get fg adj ==========
            row_idx, column_idx = zip(*[(x['source'], x['target']) for x in json_data['edges']])
            i = torch.LongTensor([row_idx, column_idx])
            v = torch.ones(len(row_idx))
            fg_adj = torch.sparse.FloatTensor(i, v, torch.Size([len(atom_types), len(atom_types)]))
            fg_adj = fg_adj + fg_adj.transpose(1, 0)
            fg_adj = fg_adj.to_dense()
            np_fg_adj = fg_adj.numpy()

            fg_id_cg_id_dict = {int(x['id']): int(x['cg']) for x in json_data['nodes']}

            anchor_fg_idx_buffer = []
            pos_fg_idx_buffer = []
            neg_fg_idx_buffer = []

            pos_pairs = []

            def find_positive_vertex(fg_id, cur_cg_id):
                neighbors = np.where(np_fg_adj[fg_id] > 0)[0]
                np.random.shuffle(neighbors)

                for n in neighbors:
                    cg_id = fg_id_cg_id_dict[n]
                    if cg_id == cur_cg_id:
                        return n
                return None

            for edge in json_data['edges']:
                u, v = edge['source'], edge['target']
                u_cg, v_cg = fg_id_cg_id_dict[u], fg_id_cg_id_dict[v]

                if u_cg != v_cg:
                    # e_uv is a cut
                    if np.random.random() < 0.5:
                        u, v = v, u
                        u_cg, v_cg = v_cg, u_cg
                    pos = find_positive_vertex(u, u_cg)
                    anchor, neg = u, v
                    if pos is None:
                        pos = find_positive_vertex(v, v_cg)
                        anchor, neg = v, u

                    if pos is not None:
                        anchor_fg_idx_buffer.append(anchor)
                        pos_fg_idx_buffer.append(pos)
                        neg_fg_idx_buffer.append(neg)
                else:
                    pos_pairs.append([u, v])

            triplet_idx = torch.tensor([anchor_fg_idx_buffer, pos_fg_idx_buffer, neg_fg_idx_buffer]).long()  # add long() for empty tensor
            pos_pairs = torch.tensor(pos_pairs).t()

            data.triplet_index = triplet_idx
            data.pos_pair_index = pos_pairs

        if self.split == 'for_simulation' or self.split == 'corona' or self.split == 'peptides' or self.split == 'peptides_martini_prediction' or self.split == 'ref_mappings':
            data.json = json_data
        if self.split == 'peptides_martini_prediction' or self.split == 'peptides' or self.split == 'ref_mappings':
            data.fname = os.path.splitext(os.path.basename(json_fpath))[0]

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.smiles_cluster_idx_dict)

    @staticmethod
    def compute_cluster_idx(json_data):
        node_cluster_index = -1 * torch.ones((len(json_data['nodes']),)).long()
        for node in json_data['nodes']:
            fg_id, cg_id = node['id'], node['cg']
            node_cluster_index[fg_id] = cg_id
        assert torch.all(node_cluster_index >= 0)
        return node_cluster_index
