#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import torch
import tqdm
import numpy as np
import os
import json
import metis
import copy

from dataset.ham_per_file import HAMPerFile
from option import arg_parse
from model.networks import DSGPM
from torch_geometric.data import DataListLoader
from model.graph_cuts import graph_cuts
from utils.post_processing import enforce_connectivity
from model.graph_cuts import graph_cuts_with_adj
from torch_geometric.nn.pool import graclus

from warnings import simplefilter
from sklearn.exceptions import UndefinedMetricWarning
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UndefinedMetricWarning)


def eval(test_dataloader, model, args):
    model.eval()

    tbar = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader), dynamic_ncols=True)
    for i, data in tbar:
        data = data[0]
        json_data = data.json
        json_data['cgnodes'] = []
        num_nodes = data.x.shape[0]
        data.batch = torch.zeros(num_nodes).long()
        data = data.to(torch.device(0))
        edge_index_cpu = data.edge_index.cpu().numpy()
        fg_embed = model(data)
        dense_adj = torch.sparse.LongTensor(data.edge_index, data.no_bond_edge_attr, (num_nodes, num_nodes)).to_dense()

        if args.num_cg_beads is None:
            iter_num_cg_beads = range(2, num_nodes)
        else:
            iter_num_cg_beads = args.num_cg_beads

        for num_cg_bead in iter_num_cg_beads:
            # try:
            if args.inference_method == 'dsgpm':
                hard_assign, _ = graph_cuts(fg_embed, data.edge_index, num_cg_bead, args.bandwidth, device=args.device_for_affinity_matrix)
                hard_assign = enforce_connectivity(hard_assign, edge_index_cpu)
            elif args.inference_method == 'spec_cluster':
                hard_assign = graph_cuts_with_adj(dense_adj, num_cg_bead)
                hard_assign = enforce_connectivity(hard_assign, edge_index_cpu)
            elif args.inference_method == 'metis':
                hard_assign = metis.part_graph(data.graph, nparts=num_cg_bead)[1]
            elif args.inference_method == 'graclus':
                hard_assign = graclus(data.edge_index.cpu()).cpu()
                hard_assign = enforce_connectivity(hard_assign, edge_index_cpu)
            actual_num_cg = max(hard_assign) + 1
            if actual_num_cg != num_cg_bead:
                print('warning: actual vs. expected: {} vs. {}'.format(actual_num_cg, num_cg_bead))
            # except RuntimeError:
            #     print('error under #cg: {}'.format(num_cg_bead))
            #     continue

            result_json = copy.deepcopy(json_data)
            for atom_idx, cg_idx in enumerate(hard_assign):
                result_json['nodes'][atom_idx]['cg'] = int(cg_idx)
            for cg_idx in range(num_cg_bead):
                atom_indices = np.nonzero(hard_assign == cg_idx)[0].tolist()
                atom_indices = [int(x) for x in atom_indices]
                result_json['cgnodes'].append(atom_indices)

            fpath = os.path.join(args.json_output_dir, data.graph.graph['smiles'] + '_cg_{}.json'.format(actual_num_cg))

            if os.path.exists(fpath):
                os.remove(fpath)
            with open(fpath, 'w') as f:
                json.dump(result_json, f, indent=4)

            if args.inference_method == 'graclus':
                break  # because graclus does not need num of cg beads


def main():
    args = arg_parse()
    assert args.pretrained_ckpt is not None, '--pretrained_ckpt is required.'
    assert args.json_output_dir is not None, '--json_output_dir is required.'
    args.devices = [int(device_id) for device_id in args.devices.split(',')]

    args.json_output_dir = os.path.join(args.json_output_dir, args.inference_method)

    if not os.path.exists(args.json_output_dir):
        os.makedirs(args.json_output_dir)

    test_set = HAMPerFile(data_root=args.data_root, cycle_feat=args.use_cycle_feat, degree_feat=args.use_degree_feat, automorphism=args.automorphism)

    test_dataloader = DataListLoader(test_set, batch_size=1, num_workers=args.num_workers,
                                     pin_memory=True)

    model = DSGPM(args.input_dim, args.hidden_dim,
                  args.output_dim, args=args).cuda()
    ckpt = torch.load(args.pretrained_ckpt)
    model.load_state_dict(ckpt)

    with torch.no_grad():
        eval(test_dataloader, model, args)


if __name__ == '__main__':
    main()
