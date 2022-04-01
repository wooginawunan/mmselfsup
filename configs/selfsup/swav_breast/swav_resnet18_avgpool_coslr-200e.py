_base_ = [
    '../_base_/models/swav_breast.py',
    '../_base_/datasets/breastscreening_swav.py',
    '../_base_/schedules/lars_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]


# additional hooks
custom_hooks = [
    dict(
        type='SwAVHook',
        priority='VERY_HIGH',
        batch_size={{_base_.data.samples_per_gpu}},
        epoch_queue_starts=15,
        crops_for_assign=[0, 1],
        feat_dim=128,
        queue_length=1000)
]

# # dataset summary
# data = dict(train=dict(num_views={{_base_.num_crops}}))

# optimizer
optimizer = dict(type='LARS', lr=0.6)
optimizer_config = dict(frozen_layers_cfg=dict(prototypes=5005))

# learning policy
lr_config = dict(_delete_=True, policy='CosineAnnealing', min_lr=6e-4)

# fp16 = dict(loss_scale='dynamic')

# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=1, max_keep_ckpts=10)
