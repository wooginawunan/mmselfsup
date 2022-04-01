# Copyright (c) OpenMMLab. All rights reserved.
import torch

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

        BS, C, H, W = us_imgs.shape

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


