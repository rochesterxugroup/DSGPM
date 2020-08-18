#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu

import torch


def pad_tensor(input_tensor, pad, dim):
    """
    args:
        input_tensor - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(input_tensor.shape)
    pad_size[dim] = pad - input_tensor.size(dim)
    pad_tensor = torch.zeros(*pad_size).type(input_tensor.type())
    return torch.cat([input_tensor, pad_tensor], dim=dim)


def pad_tensor_two_dims(input_tensor, pad: tuple):
    assert len(pad) == 2
    holder = torch.zeros(pad)
    holder[:input_tensor.shape[0], :input_tensor.shape[1]] = input_tensor
    return holder


def pad_tensor_lst_three_dims(input_tensor_lst, pad: tuple):
    assert len(pad) == 3
    holder = torch.zeros(pad)
    for idx, tensor in enumerate(input_tensor_lst):
        holder[idx, :tensor.shape[0], :tensor.shape[1]] = tensor
    return holder


def fg_cg_data_pad_collate_train(batch):
    """
    args:
        batch - list of (atom_types_tensor, fg_adj, mapping_op, cg_adj)

    reutrn:
        batchified atom_types_tensor, fg_adj, mapping_op, cg_adj
    """
    # find largest number of fg atoms
    batch_len = len(batch[0])
    assert batch_len == 5
    # num_fg_atoms = map(lambda x: x[0].shape[0], batch)
    # num_cg_beads = map(lambda x: x[-1].shape[0], batch)
    num_fg_atoms = [data[0].shape[0] for data in batch]
    num_cg_beads = [data[2].shape[1] for data in batch]
    num_triplet_sample = [data[3].shape[0] for data in batch]
    num_pos_pair_sample = [data[4].shape[0] for data in batch]

    max_fg_atom_num = max(num_fg_atoms)
    max_cg_atom_num = max(num_cg_beads)
    max_num_triplet_sample = max(num_triplet_sample)
    max_pos_pair_sample = max(num_pos_pair_sample)

    padded_batch = []
    for data in batch:
        atom_types_tensor, fg_adj, mapping_op, triplet_idx, pos_pair = data
        atom_types_tensor = pad_tensor(atom_types_tensor, max_fg_atom_num, dim=0)
        fg_adj = pad_tensor_two_dims(fg_adj, (max_fg_atom_num, max_fg_atom_num))
        mapping_op = pad_tensor_two_dims(mapping_op, (max_fg_atom_num, max_cg_atom_num))
        triplet_idx = pad_tensor(triplet_idx, max_num_triplet_sample, dim=0)
        pos_pair = pad_tensor(pos_pair, max_pos_pair_sample, dim=0)

        padded_batch.append((atom_types_tensor, fg_adj, mapping_op, triplet_idx, pos_pair))

    ret = tuple([torch.stack([data[i] for data in padded_batch], dim=0) for i in range(batch_len)]) \
             + (torch.LongTensor(num_fg_atoms).reshape(-1, 1),
                torch.LongTensor(num_cg_beads).reshape(-1, 1),
                torch.LongTensor(num_triplet_sample).reshape(-1, 1),
                torch.LongTensor(num_pos_pair_sample).reshape(-1, 1))
    return ret

    # find longest sequence
    # max_len = max(map(lambda x: x[0].shape[self.dim], batch))
    # # pad according to max_len
    # batch = map(lambda (x, y):
    #             (pad_tensor(x, pad=max_len, dim=self.dim), y), batch)
    # # stack all
    # xs = torch.stack(map(lambda x: x[0], batch), dim=0)
    # ys = torch.LongTensor(map(lambda x: x[1], batch))
    # return xs, ys


def fg_cg_data_pad_collate_test(batch):
    """
    args:
        batch - list of (atom_types_tensor, fg_adj, mapping_ops)

    reutrn:
        batchified atom_types_tensor, fg_adj, mapping_ops
    """
    # find largest number of fg atoms
    batch_len = len(batch[0])
    assert batch_len == 3
    num_fg_atoms = [data[0].shape[0] for data in batch]
    # num_cg_beads = [data[2].shape[2] for data in batch]
    num_cg_beads = []
    for data in batch:
        for mapping_op in data[2]:
            num_cg_beads.append(mapping_op.shape[1])
    num_annotations = [len(data[2]) for data in batch]

    max_fg_atom_num = max(num_fg_atoms)
    max_cg_atom_num = max(num_cg_beads)
    max_anno = max(num_annotations)

    padded_batch = []
    for data in batch:
        atom_types_tensor, fg_adj, mapping_ops = data
        atom_types_tensor = pad_tensor(atom_types_tensor, max_fg_atom_num, dim=0)
        fg_adj = pad_tensor_two_dims(fg_adj, (max_fg_atom_num, max_fg_atom_num))
        mapping_ops = pad_tensor_lst_three_dims(mapping_ops, (max_anno, max_fg_atom_num, max_cg_atom_num))

        padded_batch.append((atom_types_tensor, fg_adj, mapping_ops))

    ret = tuple([torch.stack([data[i] for data in padded_batch], dim=0) for i in range(batch_len)]) \
             + (torch.LongTensor(num_fg_atoms).reshape(-1, 1),
                torch.LongTensor(num_cg_beads).reshape(-1, 1),
                torch.LongTensor(num_annotations).reshape(-1, 1))
    return ret


def fg_cg_data_pad_collate_for_vis(batch):
    """
    args:
        batch - list of (atom_types_tensor, fg_adj, mapping_ops, graph)

    reutrn:
        batchified atom_types_tensor, fg_adj, mapping_ops, graph
    """
    # find largest number of fg atoms
    batch_len = len(batch[0])
    assert batch_len == 4
    num_fg_atoms = [data[0].shape[0] for data in batch]
    # num_cg_beads = [data[2].shape[2] for data in batch]
    num_cg_beads = []
    for data in batch:
        for mapping_op in data[2]:
            num_cg_beads.append(mapping_op.shape[1])
    num_annotations = [len(data[2]) for data in batch]

    max_fg_atom_num = max(num_fg_atoms)
    max_cg_atom_num = max(num_cg_beads)
    max_anno = max(num_annotations)

    padded_batch = []
    graph_nxs = []
    for data in batch:
        atom_types_tensor, fg_adj, mapping_ops, graph_nx = data
        atom_types_tensor = pad_tensor(atom_types_tensor, max_fg_atom_num, dim=0)
        fg_adj = pad_tensor_two_dims(fg_adj, (max_fg_atom_num, max_fg_atom_num))
        mapping_ops = pad_tensor_lst_three_dims(mapping_ops, (max_anno, max_fg_atom_num, max_cg_atom_num))

        graph_nxs.append(graph_nx)
        padded_batch.append((atom_types_tensor, fg_adj, mapping_ops))

    ret = tuple([torch.stack([data[i] for data in padded_batch], dim=0) for i in range(batch_len - 1)]) \
             + (graph_nxs,
                torch.LongTensor(num_fg_atoms).reshape(-1, 1),
                torch.LongTensor(num_cg_beads).reshape(-1, 1),
                torch.LongTensor(num_annotations).reshape(-1, 1))
    return ret
