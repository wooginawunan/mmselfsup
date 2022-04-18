# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule
from sklearn.metrics import roc_auc_score
import torch

def auc(pred, target):
    """Compute auc of predictions.

    Args:
        pred (Tensor): The output of the model.
        target (Tensor): The labels of data.
        topk (int | list[int]): Top-k metric selection. Defaults to 1.
    """

    if len(torch.unique(target))>1:
        pred = torch.nn.functional.softmax(pred, dim=1)[:, 1]
        auc = roc_auc_score(target.cpu(), 
            pred.cpu().detach().numpy())

        return torch.tensor([auc], dtype=torch.float16)
    else:
        return torch.tensor([0], dtype=torch.float16)


class ROCAUC(BaseModule):
    """Implementation of AUC (binary) computation."""

    def __init__(self):
        super().__init__()
        self.topk = topk

    def forward(self, pred, target):
        return auc(pred, target)
