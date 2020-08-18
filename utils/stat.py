#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class StdAverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val_lst = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.val_lst.append(val)

    def std(self):
        return np.std(np.array(self.val_lst))


class FoldEpochMat:
    def __init__(self, num_fold, num_epoch, best_result_keys, *keys):
        self.best_result_keys = best_result_keys
        self.key_to_mat_dict = {}
        for k in keys:
            self.key_to_mat_dict[k] = np.zeros((num_fold, num_epoch))

    def update(self, fold, epoch, key_val_dict: dict):
        for key, val in key_val_dict.items():
            self.key_to_mat_dict[key][fold, epoch] = val

    def result(self, fold):
        fold += 1
        criterion = sum([self.key_to_mat_dict[key][:fold].mean(axis=0) for key in self.best_result_keys])
        best_epoch = criterion.argmax()

        mean_ret = {}
        std_ret = {}
        for key, mat in self.key_to_mat_dict.items():
            colomn = mat[:fold, best_epoch]
            mean_ret[key] = colomn.mean()
            std_ret[key] = colomn.std()

        return mean_ret, std_ret, best_epoch + 1
