CUDA_VISIBLE_DEVICES=0 python self-sup_pre-train.py \
  --title large_pretrain_mask \
  --data_root /scratch/zli82/cg_exp/self-sup_data \
  --batch_size 64 \
  --num_workers 36 \
  --ckpt /scratch/zli82/cg_exp/ckpt/large_pretrain_mask \
  --tb_root /scratch/zli82/cg_exp/tensorboard \
  --tb_log
