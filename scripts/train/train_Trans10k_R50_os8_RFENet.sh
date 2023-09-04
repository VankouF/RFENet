#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=$1
mkdir -p ${EXP_DIR}

python3 -m torch.distributed.launch --nproc_per_node=4 train.py \
  --dataset Trans10k \
  --arch network.RFENet.RFENet_resnet50_os8 \
  --max_cu_epoch 60 \
  --lr 0.04 \
  --lr_schedule poly \
  --gblur \
  --poly_exp 0.9 \
  --repoly 1.5  \
  --rescale 1.0 \
  --syncbn \
  --sgd \
  --crop_size 512 \
  --apex \
  --log_root ${EXP_DIR} \
  --exp  r50_os8_60epoches \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  --num_cascade 4 \
  --bs_mult 4 \
  --max_epoch 60 \
  --dice_loss \
  --edge_num_points 256 \
  --region_num_points 256 \
  --edge_weight 0.25 \
  --middle_weight 0.01 \
  --seg_weight  1.0 \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt
