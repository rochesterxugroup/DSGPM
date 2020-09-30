#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import pickle
import random
import os
import math
import itertools

from tqdm import tqdm


if __name__ == '__main__':
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
    output_dir = '/scratch/zli82/cg_exp/ChEMBL_split'
    pkl_fpath = '/scratch/zli82/cg_exp/ChEMBL_split/element_to_json_lst.pkl'
    with open(pkl_fpath, 'rb') as f:
        dict_element_to_fname_lst: dict = pickle.load(f)

    len_per_element = [len(l) for l in dict_element_to_fname_lst.values()]

    # SAMPLE_RATIO = 0.05
    SAMPLE_RATIO = 1.0

    print('splitting...')
    split_dict = {'train': {}, 'val': {}, 'test': {}}

    for element_name, fname_lst in tqdm(dict_element_to_fname_lst.items(), dynamic_ncols=True):
        random.shuffle(fname_lst)
        len_fname_lst = len(fname_lst)

        # sampling
        if SAMPLE_RATIO != 1 and len_fname_lst >= 1000:
            sample_len = max(math.ceil(SAMPLE_RATIO * len_fname_lst), 3)
            fname_lst = fname_lst[:sample_len]
            len_fname_lst = len(fname_lst)

        end_test_idx = math.ceil(len_fname_lst * TEST_RATIO)
        end_val_idx = end_test_idx + math.ceil(len_fname_lst * VAL_RATIO)

        test_fname_lst = fname_lst[:end_test_idx]
        val_fname_lst = fname_lst[end_test_idx: end_val_idx]
        train_fname_lst = fname_lst[end_val_idx:]

        if len(train_fname_lst) == 0 or len(val_fname_lst) == 0 or len(test_fname_lst) == 0:
            print('error on {} split.'.format(element_name))

        for split_name, split in {'train': train_fname_lst, 'val': val_fname_lst, 'test': test_fname_lst}.items():
            split_dict[split_name][element_name] = split

    with open(os.path.join(output_dir, 'split_dict_sample_ratio_{}.pkl'.format(SAMPLE_RATIO)), 'wb') as f:
        pickle.dump(split_dict, f)

    print('saving to txt files...')
    for split_name in ['train', 'val', 'test']:
        element_to_fname_lst_split = split_dict[split_name]
        split_lst = list(itertools.chain(*element_to_fname_lst_split.values()))
        output_fpath = os.path.join(output_dir, '{}_sample_ratio_{}.txt'.format(split_name, SAMPLE_RATIO))
        if os.path.exists(output_fpath):
            os.remove(output_fpath)
        with open(output_fpath, 'w+') as f:
            for fname in split_lst:
                f.write('{}\n'.format(fname))
