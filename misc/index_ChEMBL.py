#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import orjson
import os
import random
import pickle

from multiprocessing import Pool
from tqdm import tqdm


if __name__ == '__main__':
    data_root = '/public/gwellawa/mol_graphs_no_metals'
    output_dir = '/scratch/zli82/cg_exp/ChEMBL_split'

    dict_element_to_fname_lst = {'C': [], 'N': [], 'O': [], 'S': [], 'CL': [], 'F': [], 'BR': [], 'P': [], 'SE': [],
                                 'B': [], 'I': [], 'SI': [], 'TE': [], 'AS': [], 'ZN': [], 'AL': []}

    def worker(json_fname):
        json_fpath = os.path.join(data_root, json_fname)
        with open(json_fpath, 'rb') as f:
            json_data = orjson.loads(f.read())
        element_class = json_data['class']
        # dict_element_to_fname_lst[element_class].append(json_fname)
        return element_class, json_fname

    json_fname_lst = [json_fname for json_fname in os.listdir(data_root) if json_fname.endswith('.json')]
    with Pool(36) as p:
        result = list(tqdm(p.imap(worker, json_fname_lst), total=len(json_fname_lst), dynamic_ncols=True))

    print('saving to dict...')
    for element_class, json_fname in tqdm(result, dynamic_ncols=True):
        dict_element_to_fname_lst[element_class].append(json_fname)

    with open(os.path.join(output_dir, 'element_to_json_lst.pkl'), 'wb') as f:
        pickle.dump(dict_element_to_fname_lst, f)
