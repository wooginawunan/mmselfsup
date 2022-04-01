# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ALGORITHMS, build_backbone, build_head
from ..utils import Sobel
from .base import BaseModel

@ALGORITHMS.register_module()
class FFDMClassification(BaseModel):
    """Simple image classification.

    Args:
        backbone (dict): Config dict for module of backbone.
        with_sobel (bool): Whether to apply a Sobel filter.
            Defaults to False.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self, ffdm_backbone, with_sobel=False, head=None, init_cfg=None):
        super(FFDMClassification, self).__init__(init_cfg)
        self.with_sobel = with_sobel
        if with_sobel:
            self.sobel_layer = Sobel()
        self.ffdm_backbone = build_backbone(ffdm_backbone)
        self.us_backbone = build_backbone(ffdm_backbone)
        assert head is not None
        self.head = build_head(head)

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
        x = self.ffdm_backbone(img)
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
        if self.with_sobel:
            img = self.sobel_layer(img)
        x = self.ffdm_backbone(img)
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
        if self.with_sobel:
            img = self.sobel_layer(img)
        x = self.ffdm_backbone(img)
        outs = self.head(x)
        keys = [f'head{i}' for i in self.ffdm_backbone.out_indices]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))


@ALGORITHMS.register_module()
class USClassification(BaseModel):
    """Simple image classification.

    Args:
        backbone (dict): Config dict for module of backbone.
        with_sobel (bool): Whether to apply a Sobel filter.
            Defaults to False.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self, backbone, with_sobel=False, head=None, init_cfg=None):
        super(USClassification, self).__init__(init_cfg)
        self.with_sobel = with_sobel
        if with_sobel:
            self.sobel_layer = Sobel()
        self.us_backbone = build_backbone(backbone)
        assert head is not None
        self.head = build_head(head)

    def fusion(self, x, us_counts):

        start_idx = 0
        _x = []
        for i, end_idx in enumerate(torch.cumsum(us_counts, dim=0)):
            _x.append(torch.mean(x[start_idx:end_idx], dim=0))
            start_idx = end_idx

        return torch.stack(_x)

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
        x = self.us_backbone(img)
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
        x = [self.fusion(x[0], kwargs['us_counts'])]
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
        x = [self.fusion(x[0], kwargs['us_counts'])]
        outs = self.head(x)
        keys = [f'head{i}' for i in self.us_backbone.out_indices]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))