_base_ = [
    'base.py',
]

dataset_type = 'BreastUSClassification'
train_pipeline = [
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='RandomVerticalFlip', p=0.5),
    dict(type='RandomAffine', degrees=(-10, 10), translate=(0.1,0.1), scale=(0.9, 1.1)),
    dict(type='Resize', size=(224, 224)),
    dict(type='ToNumpy'), # TODO
    dict(type='ToTensor'),
    dict(type='Standardizer')
]
val_pipeline = [
    dict(type='Resize', size=(224, 224)),
    dict(type='ToNumpy'), # TODO
    dict(type='ToTensor'),
    dict(type='Standardizer')
]

# prefetch
prefetch = False
img_norm_cfg = dict()

# dataset summary
data = dict(
    train=dict(type=dataset_type, pipeline=train_pipeline),
    val=dict(type=dataset_type, pipeline=val_pipeline))

model = dict(type='USClassification')

# optimizer
optimizer = dict(type='Adam', lr=1e-4, weight_decay=1e-6)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# # optimizer
# optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05)

# # learning policy
# lr_config = dict(
#     policy='CosineAnnealing',
#     min_lr=0.,
#     warmup='linear',
#     warmup_iters=5,
#     warmup_ratio=1e-4,  # cannot be 0
#     warmup_by_epoch=True)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)

# swav setting
# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
evaluation = dict(interval=1)
