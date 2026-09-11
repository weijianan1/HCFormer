#!/bin/bash

IMAGE=images/A.JPEG
MODEL=hcformer_tiny
CHECKPOINT=output/hcformer_tiny/model_best.pth.tar

python3 cluster_visualize.py \
    --mode visualize \
    --image $IMAGE \
    --model $MODEL \
    --checkpoint $CHECKPOINT \
    --stage 0 \
    --block 0 \
    --head 0 \
    --branch global-hyperbolic

# FEC:
# python3 cluster_visualize.py --mode fec --image $IMAGE \
#     --model $MODEL --checkpoint $CHECKPOINT --num-clusters 4
