#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import random
import torch

from dataset.ham import MASK_ATOM_INDEX


class MaskAtomType(object):
    def __init__(self, mask_ratio):
        self.mask_ratio = mask_ratio

    def __call__(self, data):
        num_atom = data.x.size(0)
        num_masked_atoms = int(num_atom * self.mask_ratio + 1)

        # do not change the name of 'masked_atom_index'.
        # PyTorch Geometric will shift attributes containing 'index' in a batch.
        # see https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data.masked_atom_index = torch.tensor(random.sample(range(num_atom), num_masked_atoms))
        data.masked_atom_type = data.x[data.masked_atom_index].squeeze(1)

        data.x = data.x + 1  # shift for padding index
        data.x[data.masked_atom_index] = MASK_ATOM_INDEX
        # data.masked_atom_index.unsqueeze_(dim=0)
        return data
