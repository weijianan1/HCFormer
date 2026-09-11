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
        type='HCFormerMediumBackbone',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/hcformer_medium.pth.tar',
        ),
    ),
    neck=dict(in_channels=[64, 128, 320, 512]),
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-4,
    weight_decay=1e-4,
)
