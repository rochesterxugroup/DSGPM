#
# Copyright (c) 2020
# Licensed under The MIT License
# Written by Zhiheng Li
# Email: zhiheng.li@rochester.edu
#

CUDA_VISIBLE_DEVICES=0 python self-sup_pre-train.py \
  --title ham_pretrain_mask \
  --data_root /scratch/zli82/dataset/HAM_dataset/data \
  --batch_size 50 \
  --num_workers 28 \
  --ckpt /scratch/zli82/cg_exp/ckpt/ham_pretrain_mask \
  --tb_root /scratch/zli82/cg_exp/experiment/tensorboard