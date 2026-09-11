# HCFormer for COCO Detection and Instance Segmentation

This directory contains the MMDetection entry points and shared configuration templates used to evaluate HCFormer with Mask R-CNN on COCO 2017.

Install the versions listed in the repository-level `requirements.txt`, prepare COCO using the standard MMDetection directory layout, and set `data_root` in `configs/_base_/datasets/coco_instance.py`.

Nano, Tiny, Small, and Medium configurations are available:

```text
configs/mask_rcnn_hcformer_nano_fpn_1x_coco.py
configs/mask_rcnn_hcformer_tiny_fpn_1x_coco.py
configs/mask_rcnn_hcformer_small_fpn_1x_coco.py
configs/mask_rcnn_hcformer_medium_fpn_1x_coco.py
```

They use Mask R-CNN, the 1× schedule, AdamW, an initial learning rate of `1e-4`, and the corresponding ImageNet-pretrained HCFormer backbone. Train and evaluate from the repository root:

```bash
bash detection/dist_train.sh \
    detection/configs/mask_rcnn_hcformer_tiny_fpn_1x_coco.py 4 \
    --work-dir output/coco/hcformer

bash detection/dist_test.sh \
    detection/configs/mask_rcnn_hcformer_tiny_fpn_1x_coco.py \
    /path/to/checkpoint.pth 4 \
    --eval bbox segm
```

Model checkpoints are not included; update each config's `model.backbone.init_cfg.checkpoint` before training.
