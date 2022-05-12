_base_ = [
    '../base.py',
]

dataset_type = 'BreastFFDMGMIC'
train_pipeline = [
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='RandomVerticalFlip', p=0.5),
    dict(type='RandomAffine', degrees=(-10, 10), translate=(0.1,0.1), scale=(0.9, 1.1)),
    dict(type='Resize', size=(1472, 960)),
    dict(type='ToNumpy'), 
    dict(type='ToTensor'),
    dict(type='Standardizer'),
]

val_pipeline = [
    dict(type='Resize', size=(1472, 960)),
    dict(type='ToNumpy'), 
    dict(type='ToTensor'),
    dict(type='Standardizer'),
]

# dataset summary
data = dict(
    samples_per_gpu=32,
    train=dict(type=dataset_type, pipeline=train_pipeline),
    val=dict(type=dataset_type, pipeline=val_pipeline))

# model
model = dict(
    type='FFDMGMIC',
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=1,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN'),
        frozen_stages=4,
        ),
    head=dict(
        type='GMICHead', 
        in_channels=512,
        num_classes=1,
        cam_weight=1,
        percent_k=0.05,
        ))

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

"""
export GPUS=1
export CFG=configs/benchmarks/classification/nyubreast/ffdm/gmic_highreso.py
export PREFIX=/gpfs/data/geraslab/Nan/saves/selfsup/swav_breast/data_20220111_full

python -m torch.distributed.launch \
    --master_addr="127.0.0.2" \
    --master_port=29501 \
    --nproc_per_node=$GPUS \
    tools/train.py $CFG \
    --work-dir "$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/linear/swav_resnet18_avgpool_coslr-200e-ori_resolution_skynet/" \
    --launcher="pytorch" \
    --breast \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
        model.backbone.init_cfg.checkpoint=$PREFIX/swav_resnet18_avgpool_coslr-200e-ori_resolution_skynet/ffdm_latest.pth \

"""