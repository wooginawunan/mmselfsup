# dataset settings
data_prefix = '/gpfs/data/geraslab/Nan/data/breast_mml_datalists/20220111/breasts_lists/ffdm_screening_only/'
data_source = 'NYUBreastScreening'

# dataset summary
data = dict(
    samples_per_gpu=64,  # total 32*8=256
    workers_per_gpu=20,
    drop_last=False,
    num_subsamples=4000,
    train=dict(
        data_source=dict(
            type=data_source,
            data_prefix=data_prefix+'full',
            test_mode=False,
            color_type='gray',
        )),
    val=dict(
        data_source=dict(
            type=data_source,
            data_prefix=data_prefix+'balanced',
            test_mode=True,
            color_type='gray',
        )))

# prefetch
prefetch = False
img_norm_cfg = dict()


# model
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
        #norm_cfg=dict(type='SyncBN'),
        #zero_init_residual=True,
        frozen_stages=4
        ),
    head=dict(
        type='ClsHead', with_avg_pool=True, in_channels=512,
        num_classes=2))

train_cfg = {}
test_cfg = {}
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb
# yapf:disable
log_config = dict(
    interval=1,
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
