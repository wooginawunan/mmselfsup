# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead, MultiLabelHead, FocalLossClsHead
from .contrastive_head import ContrastiveHead
from .latent_pred_head import LatentClsHead, LatentPredictHead
from .mae_head import MAEFinetuneHead, MAEPretrainHead
from .mocov3_head import MoCoV3Head
from .multi_cls_head import MultiClsHead
from .simmim_head import SimMIMHead
from .swav_head import SwAVHead, BreastMultiCropSwAVHead

__all__ = [
    'ContrastiveHead', 'ClsHead', 'LatentPredictHead', 'LatentClsHead',
    'MultiClsHead', 'SwAVHead', 'MAEFinetuneHead', 'MAEPretrainHead',
    'MoCoV3Head', 'SimMIMHead', 'MultiLabelHead', 'FocalLossClsHead',
    'BreastMultiCropSwAVHead'
]
