# dataset settings
data_source = 'NYUBreastScreening'
dataset_type = 'BreastScreeningDataset'
ffdm_pipeline = [
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='RandomVerticalFlip', p=0.5),
    dict(type='RandomAffine', degrees=(-10, 10), translate=(0.1,0.1), scale=(0.9, 1.1)),
    dict(type='Resize', size=(736, 480)),
    dict(type='ToNumpy'), # TODO
    dict(type='ToTensor'),
    dict(type='Standardizer'),
]
us_pipeline = [
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='RandomVerticalFlip', p=0.5),
    dict(type='RandomAffine', degrees=(-10, 10), translate=(0.1,0.1), scale=(0.9, 1.1)),
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
    samples_per_gpu=32,  # total 32*8=256
    workers_per_gpu=16,
    drop_last=True,
    num_subsamples=None,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='/gpfs/data/geraslab/Nan/data/breast_mml_datalists/20220111/breasts_lists/ffdm_screening_only/full',
        ),
        ffdm_pipeline=ffdm_pipeline,
        us_pipeline=us_pipeline,
        )
    )