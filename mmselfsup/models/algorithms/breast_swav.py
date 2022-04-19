# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class BreastSwAV(BaseModel):
    """SwAV.

    Implementation of `Unsupervised Learning of Visual Features by Contrasting
    Cluster Assignments <https://arxiv.org/abs/2006.09882>`_.
    The queue is built in `core/hooks/swav_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 ffdm_backbone,
                 us_backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(BreastSwAV, self).__init__(init_cfg)
        self.ffdm_backbone = build_backbone(ffdm_backbone)
        self.us_backbone = build_backbone(us_backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: Backbone outputs.
        """
        ffdm_img, us_imgs = img
        return self.ffdm_backbone(ffdm_img), self.us_backbone(us_imgs)

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        ffdm_img, us_imgs = img

        us_counts = kwargs['us_counts']

        ffdm_output = self.ffdm_backbone(ffdm_img)
        us_output = self.us_backbone(us_imgs)

        start_idx = 0
        _us_output = []
        for i, end_idx in enumerate(torch.cumsum(us_counts, dim=0)):
            _out = torch.mean(us_output[0][start_idx:end_idx], dim=0)
            _us_output.append(_out)

            start_idx = end_idx

        us_output = [torch.stack(_us_output)]

        # ffdm_output: feature maps Batchx512x46x30 
        # us_output: feature maps [256] Batchx512x8x8

        outputs = [ffdm_output, us_output]
        outputs = self.neck(outputs)[0]

        loss = self.head(outputs)
        return loss


@ALGORITHMS.register_module()
class MILAttenBreastSwAV(BaseModel):
    """SwAV.

    Implementation of `Unsupervised Learning of Visual Features by Contrasting
    Cluster Assignments <https://arxiv.org/abs/2006.09882>`_.
    The queue is built in `core/hooks/swav_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 ffdm_backbone,
                 us_backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(MILAttenBreastSwAV, self).__init__(init_cfg)
        self.ffdm_backbone = build_backbone(ffdm_backbone)
        self.us_backbone = build_backbone(us_backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

        self.mil_attn_V = nn.Linear(512, 128, bias=False)
        self.mil_attn_U = nn.Linear(512, 128, bias=False)
        self.mil_attn_w = nn.Linear(128, 1, bias=False)

        self.locality_atten = nn.Conv2d(512, 1, kernel_size=1, stride=1, bias=False)
    
    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: Backbone outputs.
        """
        ffdm_img, us_imgs = img
        return self.ffdm_backbone(ffdm_img), self.us_backbone(us_imgs)

    def ffdm_fusion(self, x):
        atten = 2*torch.sigmoid(self.locality_atten(x))
        return [x*atten] 

    def us_fusion(self, x, us_counts):

        max_intermediate, _ = torch.max(x, dim=2)
        emb, _ = torch.max(max_intermediate, dim=2)

        attn_projection = torch.sigmoid(self.mil_attn_U(emb)) * torch.tanh(self.mil_attn_V(emb))
        attn_score = self.mil_attn_w(attn_projection)

        start_idx = 0
        for i, end_idx in enumerate(torch.cumsum(us_counts, dim=0)):
            # mask out the impact of a percentage of images
            group = range(start_idx, end_idx)
            softmax_input = attn_score[group]
            attn_score[group] = torch.softmax(softmax_input, dim=0)
            start_idx = end_idx

        weighted_x = attn_score.unsqueeze(-1).unsqueeze(-1)*x

        _x = []
        for i, end_idx in enumerate(torch.cumsum(us_counts, dim=0)):
            _out = torch.sum(weighted_x[start_idx:end_idx], dim=0)
            _x.append(_out)
            start_idx = end_idx

        return [torch.stack(_x)]

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        ffdm_img, us_imgs = img

        ffdm_output = self.ffdm_backbone(ffdm_img)
        us_output = self.us_backbone(us_imgs)

        us_output = self.us_fusion(us_output[0], kwargs['us_counts'])
        ffdm_output = self.ffdm_fusion(ffdm_output[0])

        # ffdm_output: feature maps Batchx512x46x30 
        # us_output: feature maps [256] Batchx512x8x8

        outputs = [ffdm_output, us_output]
        outputs = self.neck(outputs)[0]

        loss = self.head(outputs)
        return loss


