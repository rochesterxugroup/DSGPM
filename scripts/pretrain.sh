CUDA_VISIBLE_DEVICES=0 python self-sup_pre-train.py \
  --title ChEMBL_mask \
  --data_root /public/gwellawa/mol_graphs_no_metals \
  --batch_size 64 \
  --num_workers 36 \
  --ckpt /scratch/zli82/cg_exp/ckpt/ChEMBL_mask \
  --dataset ChEMBL \
  --split_index_folder /scratch/zli82/cg_exp/ChEMBL_split
#  --tb_root /scratch/zli82/cg_exp/tensorboard \
#  --tb_log
