#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import os
import torch
import shutil
import seaborn as sns
import numpy as np
import io

from option import arg_parse
from dataset.ham_per_file import HAMPerFile
from torch_geometric.data import DataListLoader
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from skimage.io import imsave

from warnings import simplefilter
simplefilter(action='ignore', category=Warning)


def draw_graph(graph, hard_assign, args):
    smiles = graph.graph['smiles']
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    rdDepictor.Compute2DCoords(molecule)

    palette = np.array(sns.hls_palette(hard_assign.max() + 1))

    atom_index = list(range(len(graph.nodes)))
    undirected_edges = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in molecule.GetBonds()])
    non_cut_edges_indices = np.nonzero(hard_assign[undirected_edges[:, 0]] == hard_assign[undirected_edges[:, 1]])[0]
    bond_colors = palette[hard_assign[undirected_edges[non_cut_edges_indices][:, 0]]]
    bond_colors = list(map(tuple, bond_colors))
    atom_colors = list(map(tuple, palette[hard_assign]))

    atom_id_to_color_dict = dict(zip(atom_index, atom_colors))
    non_edge_idx_to_color_dict = dict(zip(non_cut_edges_indices.tolist(), bond_colors))

    if args.svg:
        drawer = rdMolDraw2D.MolDraw2DSVG(1200, 600)
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(1200, 600)
    drawer.DrawMolecule(
        molecule,
        highlightAtoms=atom_index,
        highlightBonds=non_cut_edges_indices.tolist(),
        highlightAtomColors=atom_id_to_color_dict,
        highlightBondColors=non_edge_idx_to_color_dict,
        highlightAtomRadii=dict(zip(atom_index, [0.1] * len(atom_index)))
    )
    drawer.FinishDrawing()
    if args.svg:
        img = drawer.GetDrawingText().replace('svg:','')
        #================write to files============================
    else:
        txt = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(txt))
        img = np.asarray(img)

    return img


def gen_vis(dataloader, args):
    vis_path = args.vis_path

    for i, data in enumerate(dataloader):
        # skip saved smiles
        data = data[0]
        num_nodes = data.x.shape[0]
        data.batch = torch.zeros(num_nodes).long()
        graph_nx = data.graph
        gt_hard_assigns = data.y.cpu().numpy()

        if not args.debug:
            gt_img = draw_graph(graph_nx, gt_hard_assigns, args)
            print('success: {}'.format(graph_nx.graph['smiles']))

            if args.svg:
                fpath = os.path.join(vis_path, data.fname + '.svg')
                svg_file = open(fpath, "wt")
                svg_file.write(gt_img)
                svg_file.close()
            else:
                fpath = os.path.join(vis_path, graph_nx.graph['smiles'] + '.png')
                imsave(fpath, gt_img)


def main():
    args = arg_parse()
    assert args.vis_root is not None, '--vis_root is required.'
    args.devices = [int(device_id) for device_id in args.devices]

    # loading data
    test_set = HAMPerFile(data_root=args.data_root, cycle_feat=args.use_cycle_feat, degree_feat=args.use_degree_feat, automorphism=False)
    test_dataloader = DataListLoader(test_set, batch_size=1, num_workers=0, pin_memory=True)

    args.vis_path = os.path.join(args.vis_root, args.title)

    if not args.debug:
        if os.path.exists(args.vis_path):
            shutil.rmtree(args.vis_path)
        os.makedirs(args.vis_path)

    gen_vis(test_dataloader, args)


if __name__ == '__main__':
    main()
