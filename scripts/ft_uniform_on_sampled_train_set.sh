SAMPLE_RATIO=1.0
CUDA_VISIBLE_DEVICES=1 python ft_on_sampled_train_set.py \
  --seed 0 \
  --title no_sample_private_ChEMBL_uniform_ft_0_05_sampled_ratio_$SAMPLE_RATIO \
  --data_root /scratch/zli82/dataset/HAM_dataset/data \
  --epoch 300 \
  --batch_size 100 \
  --num_workers 10 \
  --ckpt /scratch/zli82/cg_exp/ckpt/ChEMBL_to_HAM_ft \
  --dataset HAM \
  --tb_root /scratch/zli82/cg_exp/tensorboard \
  --tb_log \
  --pretrained_ckpt /scratch/zli82/cg_exp/ckpt/ChEMBL/Sep29-00-28-48_bhd0038_ChEMBL_no_sample_uniform/best.pth \
  --sample_ratio $SAMPLE_RATIO \
  --no_save_ckpt
