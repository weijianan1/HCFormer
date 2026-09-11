# HCFormer for ADE20K Semantic Segmentation

This directory contains the MMSegmentation entry points and Semantic FPN configuration used to evaluate HCFormer on ADE20K.

Install the versions listed in the repository-level `requirements.txt`, prepare `ADEChallengeData2016`, and set `data_root` in `configs/_base_/datasets/ade20k.py`.

Run training and evaluation from the repository root:

```bash
bash segmentation/dist_train.sh \
    segmentation/configs/sem_fpn/fpn_hcformer_tiny_ade20k_80k.py \
    4 \
    --work-dir output/ade20k/hcformer_tiny

bash segmentation/dist_test.sh \
    segmentation/configs/sem_fpn/fpn_hcformer_tiny_ade20k_80k.py \
    /path/to/checkpoint.pth 4 \
    --eval mIoU
```

The backbone is registered as `HCFormerTinyBackbone`. ImageNet-pretrained weights are not included.
