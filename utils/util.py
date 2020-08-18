#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import socket
from datetime import datetime


def get_run_name(title):
    """ A unique name for each run """
    return datetime.now().strftime(
        '%b%d-%H-%M-%S') + '_' + socket.gethostname() + '_' + title
