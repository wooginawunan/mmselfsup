# dataset settings
data_source = 'NYUBreastScreening'
dataset_type = 'BreastScreeningMultiViewDataset'
ffdm_pipeline_1 = [
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='RandomVerticalFlip', p=0.5),
    dict(type='RandomAffine', degrees=(-10, 10), translate=(0.1,0.1), scale=(0.9, 1.1)),
    dict(type='Resize', size=(736, 480)),
    dict(type='ToNumpy'), 
    dict(type='ToTensor'),
    dict(type='Standardizer'),
]
ffdm_pipeline_2 = [
    dict(type='RandomResizedCrop', 
        size=(368, 240), 
        scale=(0.08, 1.0), 
        ratio=(0.75, 1.3333333333333333)),
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='RandomVerticalFlip', p=0.5),
    dict(type='RandomAffine', degrees=(-10, 10), translate=(0.1,0.1), scale=(0.9, 1.1)),
    dict(type='ToNumpy'),
    dict(type='ToTensor'),
    dict(type='Standardizer'),
]

us_pipeline = [
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='RandomVerticalFlip', p=0.5),
    dict(type='RandomAffine', degrees=(-10, 10), translate=(0.1,0.1), scale=(0.9, 1.1)),
    dict(type='Resize', size=(224, 224)),
    dict(type='ToNumpy'), 
    dict(type='ToTensor'),
    dict(type='Standardizer')
]

# prefetch
prefetch = False
img_norm_cfg = dict()


# dataset summary
data = dict(
    samples_per_gpu=32,  # total 32*8=256
    workers_per_gpu=20,
    drop_last=True,
    num_subsamples=None,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            color_type='gray',
            data_prefix='/gpfs/data/geraslab/Nan/data/breast_mml_datalists/20220111/breasts_lists/ffdm_screening_only/full',
        ),
        ffdm_num_crops=6,
        ffdm_pipelines=[ffdm_pipeline_1, ffdm_pipeline_2],
        us_pipeline=us_pipeline,
        )
    )