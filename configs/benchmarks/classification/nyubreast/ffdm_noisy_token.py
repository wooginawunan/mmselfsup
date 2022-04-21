from curses import qiflush


_base_ = [
    'base.py',
]

dataset_type = 'BreastFFDMNoisyToken'
train_pipeline = [
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='RandomVerticalFlip', p=0.5),
    dict(type='RandomAffine', degrees=(-10, 10), translate=(0.1,0.1), scale=(0.9, 1.1)),
    dict(type='Resize', size=(736, 480)),
    dict(type='ToNumpy'), 
    dict(type='ToTensor'),
    dict(type='Standardizer'),
]

val_pipeline = [
    dict(type='Resize', size=(736, 480)),
    dict(type='ToNumpy'), 
    dict(type='ToTensor'),
    dict(type='Standardizer'),
]

# dataset summary
data = dict(
    train=dict(type=dataset_type, pipeline=train_pipeline),
    val=dict(type=dataset_type, pipeline=val_pipeline))

# model
model = dict(
    type='FFDMClassification',
    head=dict(
        type='MultiLabelHead', with_avg_pool=True, in_channels=512,
        num_classes=371))

# optimizer
optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=1e-6)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)


# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)

# swav setting
# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=1, max_keep_ckpts=3)

evaluation = dict(interval=1)
seed=3
