# HCFormer

Official implementation of **Hyperbolic Hierarchical Clustering for Visual Representation Learning**.

HCFormer is a hierarchical vision backbone built around **ClusterMixer**, an interpretable token mixer that performs feature aggregation and propagation through soft clustering. It combines:

- patch-level clustering in Euclidean space for fine-grained local structure;
- window-level clustering in Lorentz hyperbolic space for global hierarchical relations;
- relative positional bias and hierarchical feature maps for classification and dense prediction.

The paper evaluates HCFormer on ImageNet-1K classification, ADE20K semantic segmentation, and COCO object detection and instance segmentation.

## Paper results

### ImageNet-1K classification

| Model | Parameters | GFLOPs | Top-1 | Throughput (images/s) | Registered model name |
| --- | ---: | ---: | ---: | ---: | --- |
| HCFormer-Nano | 5.1M | 0.9 | 73.0 ± 0.29 | 719.0 | `hcformer_nano` |
| HCFormer-Tiny | 7.0M | 1.0 | 75.1 ± 0.14 | 536.0 | `hcformer_tiny` |
| HCFormer-Small | 16.1M | 2.9 | 79.8 ± 0.06 | 324.7 | `hcformer_small` |
| HCFormer-Medium | 33.7M | 6.4 | 82.4 ± 0.03 | 235.7 | `hcformer_medium` |

Throughput was measured in the paper on a single V100 GPU with batch size 128.

### ADE20K semantic segmentation

Semantic FPN is used for all experiments.

| Backbone | Parameters | mIoU |
| --- | ---: | ---: |
| HCFormer-Nano | 8.8M | 36.8 |
| HCFormer-Tiny | 10.3M | 37.3 |
| HCFormer-Small | 18.9M | 40.4 |
| HCFormer-Medium | 35.7M | 43.3 |

### COCO detection and instance segmentation

Mask R-CNN with a 1× schedule is used for all experiments.

| Backbone | Parameters | APbox | APmask |
| --- | ---: | ---: | ---: |
| HCFormer-Nano | 24.0M | 36.0 | 34.0 |
| HCFormer-Tiny | 25.5M | 36.4 | 34.5 |
| HCFormer-Small | 34.1M | 38.7 | 36.0 |
| HCFormer-Medium | 50.8M | 40.9 | 37.6 |

## Repository layout

```text
HCFormer/
├── models/                    # HCFormer and Lorentz-space operations
├── detection/                 # MMDetection training and evaluation
├── segmentation/              # MMSegmentation training and evaluation
├── train.py                   # ImageNet training
├── validate.py                # ImageNet evaluation
├── distributed_train.sh       # Distributed ImageNet training example
├── cluster_visualize.py       # Clustering visualization and analysis
└── requirements.txt
```

Model checkpoints and training logs are intentionally not included in this repository.

## Installation

Create a Conda environment named `HCFormer` and install the dependencies from `requirements.txt`:

```bash
conda create -n HCFormer python=3.9 -y
conda activate HCFormer
pip install -r requirements.txt
```

The reference environment uses the following core versions:

```text
Python 3.9
PyTorch 1.12.1 + CUDA 11.3
torchvision 0.13.1
timm 0.9.16
geoopt 0.5.0
einops 0.8.0
```

Classification requires PyTorch, torchvision, timm, geoopt, einops, Pillow, PyYAML, and tqdm. Dense prediction additionally requires the compatible OpenMMLab stack used by this repository:

```text
mmcv-full 1.6.0
mmdet 2.24.0
mmsegmentation 0.24.0
```

`requirements.txt` contains only the direct runtime dependencies and includes the CUDA 11.3 wheel sources for PyTorch and MMCV.

## Data preparation

### ImageNet-1K

Prepare ImageNet using the standard `ImageFolder` layout:

```text
imagenet/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

The example script expects ImageNet at `../imagenet`, relative to the HCFormer repository root. You can also pass a different location through `--data_dir`.

### ADE20K

Prepare `ADEChallengeData2016` using the MMSegmentation directory layout and update `data_root` in:

```text
segmentation/configs/_base_/datasets/ade20k.py
```

### COCO

Prepare COCO 2017 using the MMDetection directory layout and update `data_root` in:

```text
detection/configs/_base_/datasets/coco_instance.py
```

## ImageNet training

The paper trains models for 310 epochs with AdamW, cosine learning-rate decay, five warm-up epochs, weight decay 0.05, Mixup, CutMix, random erasing, and RandAugment.

Edit the variables at the top of `distributed_train.sh`, then run from the repository root:

```bash
bash distributed_train.sh
```

An equivalent explicit command is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29503 \
    train.py \
    --data_dir /path/to/imagenet \
    --model hcformer_tiny \
    --batch-size 256 \
    --epochs 300 \
    --cooldown-epochs 10 \
    --lr 1e-3 \
    --weight-decay 0.05 \
    --drop-path 0.1 \
    --output output/hcformer_tiny \
    --amp
```

`--batch-size` is the per-process batch size. Adjust it and the learning rate according to the available GPU memory and total batch size.

## ImageNet evaluation

```bash
python validate.py /path/to/imagenet \
    --model hcformer_tiny \
    --checkpoint /path/to/checkpoint.pth.tar \
    --batch-size 128
```

Available paper-scale classification models are:

```text
hcformer_nano
hcformer_tiny
hcformer_small
hcformer_medium
```

## ADE20K semantic segmentation

The paper uses Semantic FPN, a crop size of 512 × 512, AdamW, 80k iterations, and a total batch size of 16.

Run from the repository root:

```bash
bash segmentation/dist_train.sh \
    segmentation/configs/sem_fpn/fpn_hcformer_tiny_ade20k_80k.py \
    4 \
    --work-dir output/ade20k/hcformer_tiny
```

Evaluate a trained model with:

```bash
bash segmentation/dist_test.sh \
    segmentation/configs/sem_fpn/fpn_hcformer_tiny_ade20k_80k.py \
    /path/to/checkpoint.pth \
    4 \
    --eval mIoU
```

The backbone initialization path is configured through `model.backbone.init_cfg.checkpoint`. ImageNet-pretrained weights are not included.

## COCO detection and instance segmentation

The paper uses Mask R-CNN, AdamW, a 1× (12 epoch) schedule, a total batch size of 16, and images resized to a maximum size of 1333 × 800.

Four Mask R-CNN configurations are provided for the paper-scale backbones:

```text
detection/configs/mask_rcnn_hcformer_nano_fpn_1x_coco.py
detection/configs/mask_rcnn_hcformer_tiny_fpn_1x_coco.py
detection/configs/mask_rcnn_hcformer_small_fpn_1x_coco.py
detection/configs/mask_rcnn_hcformer_medium_fpn_1x_coco.py
```

The training and evaluation entry points are:

```bash
bash detection/dist_train.sh \
    detection/configs/mask_rcnn_hcformer_tiny_fpn_1x_coco.py 4 \
    --work-dir output/coco/hcformer

bash detection/dist_test.sh \
    detection/configs/mask_rcnn_hcformer_tiny_fpn_1x_coco.py \
    /path/to/checkpoint.pth 4 \
    --eval bbox segm
```

## Clustering visualization

`cluster_visualize.py` can inspect local Euclidean clustering, window-level hyperbolic clustering, and cross-stage cluster aggregation.

```bash
python cluster_visualize.py \
    --mode visualize \
    --image /path/to/image.jpg \
    --model hcformer_tiny \
    --checkpoint /path/to/checkpoint.pth.tar \
    --stage 0 \
    --block 0 \
    --head 0 \
    --branch global-hyperbolic \
    --output-dir output/visualization
```

Cross-stage aggregation with optional K-means merging:

```bash
python cluster_visualize.py \
    --mode fec \
    --image /path/to/image.jpg \
    --model hcformer_tiny \
    --checkpoint /path/to/checkpoint.pth.tar \
    --num-clusters 4 \
    --output-dir output/visualization
```

## Citation

```bibtex
@inproceedings{wei2026hcformer,
  title     = {Hyperbolic Hierarchical Clustering for Visual Representation Learning},
  author    = {Wei, Jianan and Chen, Guikun and Weng, Zhiyuan and Guo, Chunchao and Wang, Yujia and Wang, Wenguan},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgments

This implementation builds on [Context Cluster](https://github.com/ma-xu/Context-Cluster), [timm](https://github.com/huggingface/pytorch-image-models), [MMDetection](https://github.com/open-mmlab/mmdetection), and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

## License

See [LICENSE](LICENSE) for the repository license. Third-party components remain subject to their respective licenses.
