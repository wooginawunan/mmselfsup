# model settings
model = dict(
    type='MultiCropBreastSwAV',
    ffdm_backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=1,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True),
    us_backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=1,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN'),
        zero_init_residual=True),
    neck=dict(
        type='SwAVNeck',
        in_channels=512,
        hid_channels=256,
        out_channels=128,
        with_avg_pool=True),
    head=dict(
        type='BreastMultiCropSwAVHead',
        feat_dim=128,  # equal to neck['out_channels']
        epsilon=0.05,
        temperature=0.1,
        num_prototypes=20,
    ))
