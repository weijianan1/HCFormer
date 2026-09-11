_base_ = [
    '../_base_/models/fpn_r50.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='HCFormerTinyBackbone',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrained/hcformer_tiny.pth.tar',
        ),
    ),
    neck=dict(in_channels=[32, 64, 196, 320]),
    decode_head=dict(num_classes=150))


# optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU')
device = 'cuda'
