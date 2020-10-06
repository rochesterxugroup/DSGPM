SAMPLE_RATIO=1.0
CUDA_VISIBLE_DEVICES=1 python ft_on_sampled_train_set.py \
  --seed 0 \
  --title epoch_500_no_sample_public_ChEMBL_weighted_ft_0_05_sampled_ratio_$SAMPLE_RATIO \
  --data_root /scratch/zli82/dataset/HAM_dataset/data \
  --epoch 500 \
  --batch_size 100 \
  --num_workers 0 \
  --ckpt /scratch/zli82/cg_exp/ckpt/ChEMBL_to_HAM_ft \
  --dataset HAM \
  --tb_root /scratch/zli82/cg_exp/tensorboard \
  --tb_log \
  --pretrained_ckpt /scratch/zli82/cg_exp/ckpt/ChEMBL/Sep29-00-23-56_bhd0038_ChEMBL_no_sample_weighted/best.pth \
  --sample_ratio $SAMPLE_RATIO
