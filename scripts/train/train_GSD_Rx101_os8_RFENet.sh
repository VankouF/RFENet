#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=$1
mkdir -p ${EXP_DIR}

python3 -m torch.distributed.launch --nproc_per_node=4 train.py \
  --dataset GSD \
  --arch network.RFENet.RFENet_resnext101_os8 \
  --max_cu_epoch 80 \
  --lr 0.01 \
  --lr_schedule poly \
  --gblur \
  --poly_exp 0.9 \
  --repoly 1.5  \
  --rescale 1.0 \
  --syncbn \
  --sgd \
  --crop_size 384 \
  --weight_decay 0.0005 \
  --max_epoch 80 \
  --dice_loss \
  --apex \
  --bs_mult 6 \
  --num_cascade 4 \
  --edge_num_points 256 \
  --region_num_points 256 \
  --exp  rx101_80epoches \
  --log_root ${EXP_DIR} \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  --edge_weight 0.25 \
  --middle_weight 0.01 \
  --seg_weight  1.0 \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt
