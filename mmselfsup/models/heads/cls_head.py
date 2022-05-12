# Copyright (c) OpenMMLab. All rights reserved.
from tkinter import X
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import HEADS
from ..utils import accuracy, auc


@HEADS.register_module()
class ClsHead(BaseModule):
    """Simplest classifier head, with only one fc layer.

    Args:
        with_avg_pool (bool): Whether to apply the average pooling
            after neck. Defaults to False.
        in_channels (int): Number of input channels. Defaults to 2048.
        num_classes (int): Number of classes. Defaults to 1000.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=2048,
                 num_classes=1000,
                 vit_backbone=False,
                 init_cfg=[
                     dict(type='Normal', std=0.01, layer='Linear'),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(ClsHead, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.vit_backbone = vit_backbone

        self.criterion = nn.CrossEntropyLoss()

        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        """Forward head.

        Args:
            x (list[Tensor] | tuple[Tensor]): Feature maps of backbone,
                each tensor has shape (N, C, H, W).

        Returns:
            list[Tensor]: A list of class scores.
        """
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = x[0]
        if self.vit_backbone:
            x = x[-1]
        if self.with_avg_pool:
            assert x.dim() == 4, \
                f'Tensor must has 4 dims, got: {x.dim()}'
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        return [cls_score]

    def loss(self, cls_score, labels):
        """Compute the loss."""
        losses = dict()
        assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
        losses['loss'] = self.criterion(cls_score[0], labels)
        losses['acc'] = accuracy(cls_score[0], labels)
        _, counts = torch.unique(labels, return_counts=True)
        if len(counts)>1:
            losses['pos_size'] = torch.as_tensor(counts[1], dtype = torch.float)
        losses['neg_size'] = torch.as_tensor(counts[0], dtype = torch.float)

        # losses['auc'] = auc(cls_score[0], labels)
        return losses

@HEADS.register_module()
class GMICHead(BaseModule):
    """Simplest classifier head, with only one fc layer.

    Args:
        with_avg_pool (bool): Whether to apply the average pooling
            after neck. Defaults to False.
        in_channels (int): Number of input channels. Defaults to 2048.
        num_classes (int): Number of classes. Defaults to 1000.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels=2048,
                 num_classes=1,
                 cam_weight=1,
                 percent_k=0.05,
                 with_avg_pool=True,
                 init_cfg=None):
        super(GMICHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.percent_k = percent_k
        self.cam_weight = cam_weight

        self.criterion = nn.BCELoss()

        self.cnn_cls = nn.Conv2d(in_channels, 
            num_classes, 
            kernel_size=1, 
            stride=1, 
            bias=False)

    def top_k_percent_pooling(self, saliency_map):
        """
        Function that perform the top k percent pooling
        """
        N, C, W, H = saliency_map.size()
        cam_flatten = saliency_map.view(N, C, -1)
        top_k = int(round(W * H * self.percent_k))
        selected_area = cam_flatten.topk(top_k, dim=2)[0]
        saliency_map_pred = selected_area.mean(dim=2).squeeze(1)
        return saliency_map_pred

    def forward(self, x):
        """Forward head.

        Args:
            x (list[Tensor] | tuple[Tensor]): Feature maps of backbone,
                each tensor has shape (N, C, H, W).

        Returns:
            list[Tensor]: A list of class scores.
        """
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = x[0]
        self.smap = torch.sigmoid(self.cnn_cls(x)) 
        cls_score = self.top_k_percent_pooling(self.smap).float()
        
        return [cls_score]

    def cam_loss(self, smap):
        """
        Computes th class activation map loss to be used as a regularizer on 
        saliency map activations.
        :param prediction: prediction tensor at either image or breast level
        :param labels: label tensor at either image or breast level
        :param other_components: dictionary of model and metadata at the label level
        """
        return smap.mean(dim=-1).mean(dim=-1).mean(axis=0)

    def loss(self, cls_score, labels):
        """Compute the loss."""
        losses = dict()
        assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
        losses['loss_cam'] = self.cam_loss(self.smap)
        losses['loss_cls'] = self.criterion(cls_score[0], labels.float())
        losses['loss'] = self.cam_weight*losses['loss_cam'] + losses['loss_cls']
        _, counts = torch.unique(labels, return_counts=True)
        if len(counts)>1:
            losses['pos_size'] = torch.as_tensor(counts[1], dtype = torch.float)
        losses['neg_size'] = torch.as_tensor(counts[0], dtype = torch.float)

        return losses

@HEADS.register_module()
class MultiLabelHead(BaseModule):
    """Simplest classifier head, with only one fc layer.

    Args:
        with_avg_pool (bool): Whether to apply the average pooling
            after neck. Defaults to False.
        in_channels (int): Number of input channels. Defaults to 2048.
        num_classes (int): Number of classes. Defaults to 1000.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=2048,
                 num_classes=1000,
                 vit_backbone=False,
                 init_cfg=[
                     dict(type='Normal', std=0.01, layer='Linear'),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ]):
        super(MultiLabelHead, self).__init__(init_cfg)
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.vit_backbone = vit_backbone

        self.criterion = torch.nn.BCEWithLogitsLoss()

        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        """Forward head.

        Args:
            x (list[Tensor] | tuple[Tensor]): Feature maps of backbone,
                each tensor has shape (N, C, H, W).

        Returns:
            list[Tensor]: A list of class scores.
        """
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = x[0]
        if self.vit_backbone:
            x = x[-1]
        if self.with_avg_pool:
            assert x.dim() == 4, \
                f'Tensor must has 4 dims, got: {x.dim()}'
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        return [cls_score]

    def loss(self, cls_score, labels):
        """Compute the loss."""
        losses = dict()
        assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
        losses['loss'] = self.criterion(cls_score[0], labels.float())

        return losses
