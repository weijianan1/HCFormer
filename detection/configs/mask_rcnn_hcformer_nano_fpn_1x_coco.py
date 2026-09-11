_base_ = [
    './_base_/models/mask_rcnn_r50_fpn.py',
    './_base_/datasets/coco_instance.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py',
]

model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='HCFormerNanoBackbone',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/hcformer_nano.pth.tar',
        ),
    ),
    neck=dict(in_channels=[32, 64, 196, 224]),
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-4,
    weight_decay=1e-4,
)
