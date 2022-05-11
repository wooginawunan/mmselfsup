# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from ..builder import ALGORITHMS, build_backbone, build_head
from ..utils import Sobel
from .base import BaseModel
from mmcv.runner import auto_fp16

@ALGORITHMS.register_module()
class BaseBreastClassification(BaseModel):
    """Simple image classification.
    """
    @auto_fp16(apply_to=('img', ))
    def forward(self, img, mode='train', **kwargs):
        """Forward function of model.

        Calls either forward_train, forward_test or extract_feat function
        according to the mode.
        """
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            x = self.extract_feat(img)
            return self.fusion(x[0], **kwargs)
        else:
            raise Exception(f'No such mode: {mode}')

    def fusion(self, x, **kwargs):
        pass

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        if self.with_sobel:
            img = self.sobel_layer(img)
        x = self.backbone(img)
        return x

    def forward_train(self, img, label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            label (Tensor): Ground-truth labels.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img)
        x = self.fusion(x[0], **kwargs)
        outs = self.head(x)
        loss_inputs = (outs, label)
        losses = self.head.loss(*loss_inputs)
        return losses

    def forward_test(self, img, **kwargs):
        """Forward computation during test.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of output features.
        """
        x = self.extract_feat(img)  # tuple
        x = self.fusion(x[0], **kwargs)
        outs = self.head(x)
        keys = [f'head{i}' for i in self.backbone.out_indices]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

@ALGORITHMS.register_module()
class USClassification(BaseBreastClassification):
    """US classification model with attention agggregation over slices.
    """

    def __init__(self, backbone, with_sobel=False, head=None, init_cfg=None):
        super(USClassification, self).__init__(init_cfg)
        self.with_sobel = with_sobel
        if with_sobel:
            self.sobel_layer = Sobel()
        self.backbone = build_backbone(backbone)
        self.mil_attn_V = nn.Linear(head.in_channels, 128, bias=False)
        self.mil_attn_U = nn.Linear(head.in_channels, 128, bias=False)
        self.mil_attn_w = nn.Linear(128, 1, bias=False)
        assert head is not None
        self.head = build_head(head)

    def fusion(self, x, **kwargs):

        max_intermediate, _ = torch.max(x, dim=2)
        emb, _ = torch.max(max_intermediate, dim=2)

        attn_projection = torch.sigmoid(self.mil_attn_U(emb)) * torch.tanh(self.mil_attn_V(emb))
        attn_score = self.mil_attn_w(attn_projection)

        cumsum_counts = torch.cumsum(kwargs['us_counts'], dim=0)

        start_idx = 0
        for i, end_idx in enumerate(cumsum_counts):
            # mask out the impact of a percentage of images
            group = range(start_idx, end_idx)
            softmax_input = attn_score[group]
            attn_score[group] = torch.softmax(softmax_input, dim=0)
            start_idx = end_idx

        weighted_x = attn_score.unsqueeze(-1).unsqueeze(-1)*x

        _x = []
        for i, end_idx in enumerate(cumsum_counts):
            _out = torch.sum(weighted_x[start_idx:end_idx], dim=0)
            _x.append(_out)
            start_idx = end_idx

        return [torch.stack(_x)]


@ALGORITHMS.register_module()
class USTsne(USClassification):
    """US classification model with attention agggregation over slices.
    """

    def fusion(self, x, **kwargs):

        cumsum_counts = torch.cumsum(kwargs['us_counts'], dim=0)
        start_idx = 0
        _x = []
        for i, end_idx in enumerate(cumsum_counts):
            _out = torch.sum(x[start_idx:end_idx], dim=0)
            _x.append(_out)
            start_idx = end_idx

        return [torch.stack(_x)]

@ALGORITHMS.register_module()
class USMultInstanceTsne(USClassification):
    """US classification model with attention agggregation over slices.
    """

    def fusion(self, x, **kwargs):
        return [x]

@ALGORITHMS.register_module()
class FFDMClassification(BaseBreastClassification):
    """FFDM Classifier with locality attention head
    """
    def __init__(self, backbone, with_sobel=False, head=None, init_cfg=None):
        super(FFDMClassification, self).__init__(init_cfg)
        self.with_sobel = with_sobel
        if with_sobel:
            self.sobel_layer = Sobel()
        self.backbone = build_backbone(backbone)
        self.locality_atten = nn.Conv2d(head.in_channels, 1, kernel_size=1, stride=1, bias=False)
        assert head is not None
        self.head = build_head(head)

    def fusion(self, x, **kwargs):
        atten = 2*torch.sigmoid(self.locality_atten(x))
        return [x*atten] 