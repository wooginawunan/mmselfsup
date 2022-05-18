# dataset settings
data_prefix = '/gpfs/data/geraslab/Nan/data/breast_mml_datalists/20220111/breasts_lists/nyu_readerstudy_ffdm/'
data_source = 'NYUBreastScreening'

dataset_type = 'NYUMammoReaderStudy'

val_pipeline = [
    dict(type='Resize', size=(736, 480)),
    dict(type='ToNumpy'), 
    dict(type='ToTensor'),
    dict(type='Standardizer'),
]

# dataset summary
data = dict(
    samples_per_gpu=8,  # total 32*8=256
    workers_per_gpu=20,
    drop_last=False,
    val=dict(
        data_source=dict(
            type=data_source,
            data_prefix=data_prefix,
            test_mode=True,
            color_type='gray',
        ),
        type=dataset_type, 
        pipeline=val_pipeline
        ))

# prefetch
prefetch = False
img_norm_cfg = dict()

# model
model = dict(
    type='NYUMammoReaderStudyModel',
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=1,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN'),
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

"""
bsub -q short -Is -n 20 -gpu "num=$GPUS:mode=shared:j_exclusive=yes" python -m torch.distributed.launch \
    --master_addr="127.0.0.1" \
    --master_port=29500 \
    --nproc_per_node=1 \
    tools/test.py configs/benchmarks/classification/nyubreast/ffdm/nyu_readerstudy.py \
    /gpfs/data/geraslab/Nan/mmselfsup/work_dirs/benchmarks/classification/nyubreast/ffdm/nyu_ffdm_screening/imagenet/locality_atten/latest.pth \
    --work-dir /gpfs/data/geraslab/Nan/mmselfsup/work_dirs/benchmarks/classification/nyu_readerstudy_mammo/ffdm/nyu_ffdm_screening/imagenet/locality_atten \
    --launcher="pytorch" \
    --breast \

export PREFIX='/gpfs/data/geraslab/Nan/mmselfsup/work_dirs/benchmarks/classification/'

bsub -q short -Is -n 20 -gpu "num=$GPUS:mode=shared:j_exclusive=yes" python -m torch.distributed.launch \
    --master_addr="127.0.0.1" \
    --master_port=29500 \
    --nproc_per_node=1 \
    tools/test.py configs/benchmarks/classification/nyubreast/ffdm/nyu_readerstudy.py \
    $PREFIX/nyubreast/ffdm/nyu_ffdm_screening/swav_resnet18_milatten1_batch128/pretrained_head/1e-3/latest.pth \
    --work-dir $PREFIX/nyu_readerstudy_mammo/ffdm/nyu_ffdm_screening_supervised/eval_nyu_readerstudy/swav_resnet18_milatten1_batch128/1e-3 \
    --launcher="pytorch" \
    --breast 
"""