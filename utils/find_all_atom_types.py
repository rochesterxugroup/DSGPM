#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import os
import json
import argparse

from tqdm import tqdm
from multiprocessing import Pool
from collections import OrderedDict
from mendeleev import element


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--workers', type=int, default=40)
    args = parser.parse_args()

    atom_set = set()

    def worker(filename):
        full_file_path = os.path.join(args.data_root, filename)
        with open(full_file_path) as f:
            json_data = json.load(f)
        nodes = json_data['nodes']
        local_atom_set = set()
        for node in nodes:
            atom_type_str = node['element']
            local_atom_set.add(atom_type_str)
        return local_atom_set

    filenames = [f for f in os.listdir(args.data_root) if f.endswith('.json')]
    with Pool(args.workers) as p:
        result = list(tqdm(p.imap(worker, filenames), total=len(filenames), dynamic_ncols=True))

    for r in result:
        atom_set = atom_set.union(r)

    atom_type_str_lst = list(atom_set)
    atom_type_str_lst.sort(key=lambda x: element(x).atomic_number)

    od = OrderedDict()
    for atom_str in atom_type_str_lst:
        weight = element(atom_str).atomic_weight
        od[atom_str] = weight

    print(od)