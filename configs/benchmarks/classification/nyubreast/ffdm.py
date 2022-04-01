# dataset settings
data_source = 'NYUBreastScreening'
dataset_type = 'BreastFFDMClassification'

train_ffdm_pipeline = [
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='RandomVerticalFlip', p=0.5),
    dict(type='RandomAffine', degrees=(-10, 10), translate=(0.1,0.1), scale=(0.9, 1.1)),
    dict(type='Resize', size=(736, 480)),
    dict(type='ToNumpy'), # TODO
    dict(type='ToTensor'),
    dict(type='Standardizer'),
    # dict(type='CopyChannel')
]

val_ffdm_pipeline = [
    dict(type='Resize', size=(736, 480)),
    dict(type='ToNumpy'), # TODO
    dict(type='ToTensor'),
    dict(type='Standardizer'),
]

# prefetch
prefetch = False
img_norm_cfg = dict()

# dataset summary
data = dict(
    samples_per_gpu=64,  # total 32*8=256
    workers_per_gpu=20,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='/gpfs/data/geraslab/Nan/data/breast_mml_datalists/20220111/breasts_lists/ffdm_screening_only/full',
            test_mode=False,
        ),
        pipeline=train_ffdm_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='/gpfs/data/geraslab/Nan/data/breast_mml_datalists/20220111/breasts_lists/ffdm_screening_only/full',
            test_mode=True,
        ),
        pipeline=val_ffdm_pipeline)
    )

evaluation = dict(interval=1, topk=(1))
model = dict(
    type='Classification',
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=1,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN'),
        frozen_stages=4),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=512,
        num_classes=2))


# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=1e-6)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)


train_cfg = {}
test_cfg = {}
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

# runtime settings
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
persistent_workers = True

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'


# swav setting
# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
