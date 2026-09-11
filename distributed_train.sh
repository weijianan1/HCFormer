#!/bin/bash

DATA_DIR=../imagenet
MODEL=hcformer_nano
OUTPUT_DIR=output/hcformer_nano
GPU_IDS=0,1,2,3,4,5,6,7
NUM_GPUS=8
PORT=29503
BATCH_SIZE=128

CUDA_VISIBLE_DEVICES=$GPU_IDS python3 -m torch.distributed.launch  \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$PORT \
    train.py \
    --data_dir $DATA_DIR \
    --model $MODEL \
    --batch-size $BATCH_SIZE \
    --lr 1e-3 \
    --drop-path 0.1 \
    --output $OUTPUT_DIR \
    --amp

# To resume: RESUME=output/hcformer_tiny/last.pth.tar
