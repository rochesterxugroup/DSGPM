CUDA_VISIBLE_DEVICES=0 python self-sup_pre-train.py \
  --title ChEMBL_no_sample_uniform \
  --data_root /public/gwellawa/mol_graphs_no_metals \
  --split_index_folder /scratch/zli82/cg_exp/ChEMBL_split \
  --epoch 40 \
  --batch_size 50 \
  --num_workers 36 \
  --ckpt /scratch/zli82/cg_exp/ckpt/ChEMBL \
  --dataset ChEMBL \
  --tb_root /scratch/zli82/cg_exp/tensorboard \
  --tb_log \
  --sample_ratio 1.0
#  --weighted_sample_mask
