# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
import pandas as pd

from mmcv.utils import build_from_cfg, print_log
from torchvision.transforms import Compose
from sklearn.metrics import roc_auc_score, average_precision_score

from .base import BaseDataset
from .builder import DATASETS, PIPELINES, build_datasource
from .utils import to_numpy


@DATASETS.register_module()
class BreastScreeningDataset(BaseDataset):
    """The dataset outputs data sampel from two modalities.

    The number of views in the output dict depends on `num_views`. The
    image can be processed by one pipeline or multiple piepelines.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        num_views (list): The number of different views.
        pipelines (list[list[dict]]): A list of pipelines, where each pipeline
            contains elements that represents an operation defined in
            `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.

    Examples:
        >>> dataset = CrossModalDataset(data_source, [2], [pipeline])
        >>> output = dataset[idx]
        The output got 2 views processed by one pipeline.

        >>> dataset = CrossModalDataset(
        >>>     data_source, [2, 6], [pipeline1, pipeline2])
        >>> output = dataset[idx]
        The output got 8 views processed by two pipelines, the first two views
        were processed by pipeline1 and the remaining views by pipeline2.
    """

    def __init__(self, data_source, ffdm_pipeline, us_pipeline):
        self.data_source = build_datasource(data_source)
        if self.data_source.color_type=='color':
            ffdm_pipeline.append(dict(type='CopyChannel'))
        self.ffdm_pipeline = Compose([build_from_cfg(p, PIPELINES) for p in ffdm_pipeline])
        self.us_pipeline = Compose([build_from_cfg(p, PIPELINES) for p in us_pipeline])


    def __getitem__(self, idx):
        img_ffdm, imgs_us = self.data_source.get_sample(idx)
        us_counts = len(imgs_us)
        img_ffdm = self.ffdm_pipeline(img_ffdm)
        imgs_us = torch.stack([self.us_pipeline(img) for img in imgs_us])

        return dict(img=[img_ffdm, imgs_us], idx=idx, us_counts=us_counts)
    
    def evaluate(self, results, logger=None):
        return NotImplemented


@DATASETS.register_module()
class BreastScreeningMultiViewDataset(BaseDataset):
    """The dataset outputs data sampel from two modalities.

    The number of views in the output dict depends on `num_views`. The
    image can be processed by one pipeline or multiple piepelines.

    Args:

    Examples:

    """

    def __init__(self, data_source, ffdm_num_crops, ffdm_pipelines, us_pipeline):
        self.data_source = build_datasource(data_source)

        self.ffdm_pipelines = []
        for pipeline in ffdm_pipelines:
            if self.data_source.color_type=='color':
                pipeline.append(dict(type='CopyChannel'))
            self.ffdm_pipelines.append(
                Compose([build_from_cfg(p, PIPELINES) for p in pipeline])
            )

        self.ffdm_num_crops = ffdm_num_crops
            
        self.us_pipeline = Compose([build_from_cfg(p, PIPELINES) \
            for p in us_pipeline])

    def __getitem__(self, idx):
        img_ffdm, imgs_us = self.data_source.get_sample(idx)
        us_counts = len(imgs_us)

        img_ffdm_full = self.ffdm_pipelines[0](img_ffdm)
        imgs_ffdm_crops = torch.stack(
            [self.ffdm_pipelines[1](img_ffdm) for i in range(self.ffdm_num_crops)])

        imgs_us = torch.stack([self.us_pipeline(img) for img in imgs_us])

        return dict(img=[img_ffdm_full, imgs_ffdm_crops, imgs_us], 
                    idx=idx, 
                    us_counts=us_counts)
    
    def evaluate(self, results, logger=None):
        return NotImplemented

@DATASETS.register_module()
class BreastClassificationDataset(BaseDataset):
    def set_epoch_sampler(self, epoch_sample_ref):
        if epoch_sample_ref=='malignant':
            self.batch_sample_labels = self.gt_labels
        elif epoch_sample_ref=='biopsied':
            self.batch_sample_labels = self.biopsed_labels

    def evaluate(self, results, logger=None):
        """The evaluation function to output accuracy.

        Args:
            results (dict): The key-value pair is the output head name and
                corresponding prediction values.
            logger (logging.Logger | str | None, optional): The defined logger
                to be used. Defaults to None.
            topk (tuple(int)): The output includes topk accuracy.
        """
        criterion = torch.nn.CrossEntropyLoss()

        eval_res = {}
        for name, val in results.items():

            """Compute the loss."""
        
            val = torch.from_numpy(val)
            target = torch.LongTensor(self.gt_labels)
            assert val.size(0) == target.size(0), (
                f'Inconsistent length for results and labels, '
                f'{val.size(0)} vs {target.size(0)}')

            num = val.size(0)

            loss = criterion(val, target)
            eval_res[f'{name}_loss'] = loss.item()

            pred = torch.nn.functional.softmax(val, dim=1)[:, 1]
            auc = roc_auc_score(target, pred)
            eval_res[f'{name}_auc'] = auc

            prauc = average_precision_score(target, pred)
            eval_res[f'{name}_prauc'] = prauc
                
            _, pred = val.max(dim=1)
            pred = pred.t()
            correct = pred.eq(target).contiguous().view(-1).float().sum(0).item()
            acc = correct * 100.0 / num
            eval_res[f'{name}_acc'] = acc

            if logger is not None and logger != 'silent':
                print_log(f'{name}_loss: {loss:.03f}', logger=logger)
                print_log(f'{name}_acc: {acc:.03f}', logger=logger)
                print_log(f'{name}_auc: {auc:.03f}', logger=logger)
                print_log(f'{name}_prauc: {prauc:.03f}', logger=logger)
        return eval_res

        
@DATASETS.register_module()
class BreastFFDMClassification(BreastClassificationDataset):

    def __init__(self, data_source, pipeline, epoch_sample_ref='malignant', prefetch=False):
        self.data_source = build_datasource(data_source)
        if self.data_source.color_type=='color':
            pipeline.append(dict(type='CopyChannel'))
        pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline]
        self.pipeline = Compose(pipeline)
        self.prefetch = prefetch
        self.gt_labels = self.data_source.get_gt_labels()
        self.biopsed_labels = self.data_source.get_biopsied_labels()
        self.set_epoch_sampler(epoch_sample_ref)

    def __getitem__(self, idx):
        label = self.gt_labels[idx]
        img = self.data_source.get_img(idx, 'ffdm')
        img = self.pipeline(img)
        if self.prefetch:
            img = torch.from_numpy(to_numpy(img))
        return dict(img=img, label=label, idx=idx)


@DATASETS.register_module()
class NYUMammoReaderStudy(BreastClassificationDataset):

    def __init__(self, data_source, pipeline, epoch_sample_ref='malignant', prefetch=False):
        self.data_source = build_datasource(data_source)
        if self.data_source.color_type=='color':
            pipeline.append(dict(type='CopyChannel'))
        pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline]
        self.pipeline = Compose(pipeline)
        self.prefetch = prefetch
        self.gt_labels = self.data_source.get_gt_labels()
        self.set_epoch_sampler(epoch_sample_ref)

    def __getitem__(self, idx):
        label = self.gt_labels[idx]
        img = self.data_source.get_img(idx, 'ffdm')
        img = self.pipeline(img)
        if self.prefetch:
            img = torch.from_numpy(to_numpy(img))
        accession_number = int(self.data_source.data_infos[idx]['accession_number'])
        lateral = int(self.data_source.data_infos[idx]['lateral']=='left')
        #print(accession_number, lateral)
        return dict(img=img, label=label, idx=idx, acc=accession_number, lateral=lateral)

    def evaluate(self, results, logger=None):
        """The evaluation function to output accuracy.

        Args:
            results (dict): The key-value pair is the output head name and
                corresponding prediction values.
            logger (logging.Logger | str | None, optional): The defined logger
                to be used. Defaults to None.
            topk (tuple(int)): The output includes topk accuracy.
        """

        eval_res = {}
        name = 'head4'
        val = torch.from_numpy(results[name])

        df = {'acc': results['acc'], 
            'lateral': results['lateral'],
            'target': self.gt_labels,
            'pred': torch.nn.functional.softmax(val, dim=1)[:, 1].numpy()}

        df = pd.DataFrame(df)
        agg_scores = df.groupby(['acc', 'lateral']).mean(['target', 'pred'])
        pred = agg_scores['pred'].values
        target = agg_scores['target'].values

        auc = roc_auc_score(target, pred)
        eval_res[f'{name}_auc'] = auc

        prauc = average_precision_score(target, pred)
        eval_res[f'{name}_prauc'] = prauc
                
        if logger is not None and logger != 'silent':
            print_log(f'{name}_auc: {auc:.03f}', logger=logger)
            print_log(f'{name}_prauc: {prauc:.03f}', logger=logger)
        return eval_res

@DATASETS.register_module()
class BreastUSClassification(BreastClassificationDataset):

    def __init__(self, data_source, pipeline, epoch_sample_ref='malignant', prefetch=False):
        self.data_source = build_datasource(data_source)
        pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline]
        self.pipeline = Compose(pipeline)
        self.prefetch = prefetch
        self.CLASSES = self.data_source.CLASSES

        self.gt_labels = self.data_source.get_gt_labels()
        self.biopsed_labels = self.data_source.get_biopsied_labels()
        self.set_epoch_sampler(epoch_sample_ref)

    def __getitem__(self, idx):
        label = self.gt_labels[idx]
        imgs = self.data_source.get_img(idx, 'us')
        us_counts = len(imgs)
        imgs = torch.stack([self.pipeline(img) for img in imgs])

        return dict(img=imgs, label=label, idx=idx, us_counts=us_counts)

@DATASETS.register_module()
class BreastNoisyTokenDataset(BaseDataset):
    def set_epoch_sampler(self, epoch_sample_ref):
        if epoch_sample_ref=='malignant':
            self.batch_sample_labels = self.gt_labels
        elif epoch_sample_ref=='biopsied':
            self.batch_sample_labels = self.biopsed_labels
    
    def evaluate(self, results, logger=None):
        """
        """
        criterion = torch.nn.BCEWithLogitsLoss()

        eval_res = {}
        for name, val in results.items():

            """Compute the loss."""
        
            val = torch.from_numpy(val)
            target = torch.Tensor(self.token_labels).float()
            assert val.size(0) == target.size(0), (
                f'Inconsistent length for results and labels, '
                f'{val.size(0)} vs {target.size(0)}')

            loss = criterion(val, target)
            eval_res[f'{name}_loss'] = loss.item()

            pred = torch.sigmoid(val)
            n_labels = target.size(1)
            auc_micro_list = []
            prauc_micro_list = []
            for i in range(n_labels):
                current_pred = pred.T[i]
                current_label = target.T[i]
                if current_label.sum()!=0:
                    auc_micro = roc_auc_score(current_label.T, current_pred.T)
                    prauc_micro = average_precision_score(current_label.T, current_pred.T)
                auc_micro_list.append(auc_micro)
                prauc_micro_list.append(prauc_micro)

            # auc_micro = roc_auc_score(target.ravel(), pred.ravel())
            auc = np.mean(auc_micro_list)
            prauc = np.mean(prauc_micro_list)
            eval_res[f'{name}_auc_micro'] = auc
            eval_res[f'{name}_prauc_micro'] = prauc

            if logger is not None and logger != 'silent':
                print_log(f'{name}_loss: {loss:.03f}', logger=logger)
                print_log(f'{name}_auc_micro: {auc:.03f}', logger=logger)
                print_log(f'{name}_prauc_micro: {prauc:.03f}', logger=logger)

        return eval_res

@DATASETS.register_module()
class BreastFFDMNoisyToken(BreastNoisyTokenDataset):

    def __init__(self, data_source, pipeline, epoch_sample_ref='malignant', prefetch=False):
        self.data_source = build_datasource(data_source)
        if self.data_source.color_type=='color':
            pipeline.append(dict(type='CopyChannel'))
        pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline]
        self.pipeline = Compose(pipeline)
        self.prefetch = prefetch
        self.gt_labels = self.data_source.get_gt_labels()
        self.biopsed_labels = self.data_source.get_biopsied_labels()
        self.token_labels = self.data_source.get_token_labels()
        self.set_epoch_sampler(epoch_sample_ref)

    def __getitem__(self, idx):
        label = self.token_labels[idx]
        img = self.data_source.get_img(idx, 'ffdm')
        img = self.pipeline(img)
        if self.prefetch:
            img = torch.from_numpy(to_numpy(img))
        return dict(img=img, label=label, idx=idx)


@DATASETS.register_module()
class BreastUSNoisyToken(BreastNoisyTokenDataset):

    def __init__(self, data_source, pipeline, epoch_sample_ref='malignant', prefetch=False):

        self.data_source = build_datasource(data_source)
        pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline]
        self.pipeline = Compose(pipeline)
        self.prefetch = prefetch
        self.CLASSES = self.data_source.CLASSES

        self.gt_labels = self.data_source.get_gt_labels()
        self.biopsed_labels = self.data_source.get_biopsied_labels()
        self.token_labels = self.data_source.get_token_labels()
        self.set_epoch_sampler(epoch_sample_ref)
        
    def __getitem__(self, idx):
        label = self.token_labels[idx]
        imgs = self.data_source.get_img(idx, 'us')
        us_counts = len(imgs)
        imgs = torch.stack([self.pipeline(img) for img in imgs])

        return dict(img=imgs, label=label, idx=idx, us_counts=us_counts)







