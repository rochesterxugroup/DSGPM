#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import torch
import os

force_root = '/public/gwellawa/3D_forces'

smiles_to_forces = {}

for fname in os.listdir(force_root):
    fpath = os.path.join(force_root, fname)
    data = torch.load(fpath, map_location=torch.device('cpu'))
    smiles = data['SMILES']
    forces = data['Forces']
    if smiles not in smiles_to_forces:
        smiles_to_forces[smiles] = []
    smiles_to_forces[smiles].append(forces)

torch.save(smiles_to_forces, '/scratch/zli82/cg_exp/forces.pth')
