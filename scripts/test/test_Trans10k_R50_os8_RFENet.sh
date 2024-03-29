#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
python3 eval.py \
	--dataset Trans10k \
    --arch network.RFENet.RFENet_resnet50_os8 \
    --inference_mode  whole \
    --single_scale \
    --scales 1.0 \
    --split test \
    --cv_split 0 \
    --resize_scale 512 \
    --mode semantic \
    --with_mae_ber \
    --no_flip \
    --ckpt_path ${2} \
    --snapshot ${1} \
    --num_cascade 4 \
    --edge_num_points 256 \
    --region_num_points 256 \
    --dump_images \
    
