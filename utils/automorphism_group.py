#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import networkx as nx

def node_equal(n1, n2, node_key='element'):
    return n1[node_key] == n2[node_key]


def edge_equal(e1, e2, edge_key='bond_type'):
    return e1[edge_key] == e2[edge_key]


