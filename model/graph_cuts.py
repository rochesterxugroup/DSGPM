#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import torch
import torch.nn.functional as F

from sklearn.cluster import spectral_clustering


def graph_cuts(fg_embed, adj, num_cg, bandwidth=1.0, kernel='rbf', device=torch.device(0)):
    affinity = compute_affinity(fg_embed, adj, bandwidth, kernel, device)

    pred_cg_idx = spectral_clustering(affinity.cpu().numpy(), n_clusters=num_cg, assign_labels='discretize')
    return pred_cg_idx, affinity


def graph_cuts_with_adj(adj, num_cg):
    pred_cg_idx = spectral_clustering(adj.cpu().numpy(), n_clusters=num_cg, assign_labels='discretize')
    return pred_cg_idx


def compute_affinity(fg_embed, adj, bandwidth=1.0, kernel='rbf', device=torch.device(0)):
    if kernel == 'rbf':
        n, d = fg_embed.shape
        fg_embed = fg_embed.to(device)
        pairwise_dist = torch.norm(fg_embed.reshape(n, 1, d) - fg_embed.reshape(1, n, d), dim=2).to(torch.device(0))

        pairwise_dist = pairwise_dist ** 2
        affinity = torch.exp(-pairwise_dist / (2 * bandwidth ** 2))
        affinity = affinity * adj
    elif kernel == 'linear':
        affinity = F.relu(fg_embed @ fg_embed.t())
    else:
        assert False

    return affinity
