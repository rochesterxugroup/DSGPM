#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import socket
import os
import json

from datetime import datetime


def get_run_name(title):
    """ A unique name for each run """
    return datetime.now().strftime(
        '%b%d-%H-%M-%S') + '_' + socket.gethostname() + '_' + title


def save_args(args, ckpt_dir):
    args_dict = vars(args)
    not_supported_keys = [key for key in args_dict.keys() if 'device' in key]
    for key in not_supported_keys:
        del args_dict[key]
    json_fpath = os.path.join(ckpt_dir, 'config.json')
    with open(json_fpath, 'w+') as f:
        json.dump(args_dict, f)
