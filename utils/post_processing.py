#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import numpy as np

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix, coo_matrix, triu
from sklearn.metrics import precision_recall_fscore_support


def enforce_connectivity(node_cluster_idx_pred, edge_index_lst):
    # build new adj based on clustering prediction
    num_vertices = len(node_cluster_idx_pred)
    filtered_edge_index_lst = []

    for u, v in edge_index_lst.T:
        if node_cluster_idx_pred[u] == node_cluster_idx_pred[v]:
            filtered_edge_index_lst.append([u, v])
    if len(filtered_edge_index_lst) == 0:
        return node_cluster_idx_pred
    filtered_edge_index = np.array(filtered_edge_index_lst).transpose(1, 0)

    data = np.ones(len(filtered_edge_index_lst))
    row = filtered_edge_index[0]
    column = filtered_edge_index[1]
    filtered_adj = csr_matrix((data, (row, column)), shape=(num_vertices, num_vertices))

    _, labels = connected_components(csgraph=filtered_adj, directed=False, return_labels=True)
    return labels


def edge_cut_prec_recall_fscore(node_cluster_idx_pred, node_cluster_idx_gt, edge_index_lst):
    num_vertices = len(node_cluster_idx_pred)
    data = np.ones(edge_index_lst.shape[1])
    row = edge_index_lst[0]
    column = edge_index_lst[1]
    adj = coo_matrix((data, (row, column)), shape=(num_vertices, num_vertices))

    upper_adj = triu(adj, format='coo')

    num_edges = int(edge_index_lst.shape[1] / 2)

    # 1 if the edge is a cut, otherwise 0
    edge_pred, edge_gt = [-1 * np.ones(num_edges, dtype=np.int32) for _ in range(2)]

    for edge_idx, (u, v) in enumerate(zip(upper_adj.row, upper_adj.col)):
        edge_pred[edge_idx] = int(node_cluster_idx_pred[u] != node_cluster_idx_pred[v])
        edge_gt[edge_idx] = int(node_cluster_idx_gt[u] != node_cluster_idx_gt[v])

    prec, rec, fscore, _ = precision_recall_fscore_support(y_true=edge_gt, y_pred=edge_pred, average='binary')
    return prec, rec, fscore
