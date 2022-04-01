# dataset settings
data_source = 'NYUBreastScreening'
dataset_type = 'BreastScreeningDataset'

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
train_us_pipeline = [
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='RandomVerticalFlip', p=0.5),
    dict(type='RandomAffine', degrees=(-10, 10), translate=(0.1,0.1), scale=(0.9, 1.1)),
    dict(type='Resize', size=(224, 224)),
    dict(type='ToNumpy'), # TODO
    dict(type='ToTensor'),
    dict(type='Standardizer')
]
val_ffdm_pipeline = [
    dict(type='Resize', size=(736, 480)),
    dict(type='ToNumpy'), # TODO
    dict(type='ToTensor'),
    dict(type='Standardizer'),
]
val_us_pipeline = [
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
    samples_per_gpu=4,  # total 32*8=256
    workers_per_gpu=0,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='/gpfs/data/geraslab/Nan/data/breast_mml_datalists/20220111/breasts_lists/ffdm_screening_only/full',
            test_mode=False,
        ),
        ffdm_pipeline=train_ffdm_pipeline,
        us_pipeline=train_us_pipeline,
        ),
    val=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='/gpfs/data/geraslab/Nan/data/breast_mml_datalists/20220111/breasts_lists/ffdm_screening_only/full',
            test_mode=True,
        ),
        ffdm_pipeline=val_ffdm_pipeline,
        us_pipeline=val_us_pipeline,
        ),
    )

evaluation = dict(interval=1, topk=(1))
