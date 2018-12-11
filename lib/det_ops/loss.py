"""

Loss functions for Detection

"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
__all__ = ['SmoothL1Loss', 'SoftmaxCrossEntropy', 'SigmoidCrossEntropy', 'SigmoidFocalLoss']


SoftmaxCrossEntropy = nn.CrossEntropyLoss
SigmoidCrossEntropy = nn.BCEWithLogitsLoss


class SigmoidFocalLoss(nn.Module):
    """ Focal Loss

    Args:

    Input:
        pred: []
        target: []
    Output:

    """
    def __init__(self, background=0, gamma=2, alpha=0.25):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.background = background

    def forward(self, pred, target):
        # pred.shape = [B, K, C]
        # target.shape = [B, N]
        B, N, C = pred.size()
        pred_sigmoid = pred.sigmoid()
        # # ignore: [B, N]
        # keep_mask = target > -1
        # # [B, N], ignore and background shared 0 index
        # keep_mask = keep_mask.long()
        mask = (target > -1)
        new_target = target * mask.long()
        new_target = new_target.reshape((new_target.size()[0], new_target.size()[1], 1))
        onehot_target = torch.zeros((B, N, C+1)).cuda()
        onehot_target.scatter_(2, new_target, 1.0)
        onehot = onehot_target[:, :, 1:].float()
        mask = mask.unsqueeze(2).float()

        # pred_sigmoid = pred_sigmoid.clamp(min=1e-6, max=1-1e-6)
        # bce = -(onehot * torch.log(pred_sigmoid) + (1-onehot)*torch.log(1-pred_sigmoid))

        weight = self.alpha*((1-pred_sigmoid).pow(self.gamma))*onehot \
            + (1-self.alpha)*(pred_sigmoid.pow(self.gamma))*(1-onehot)
        weight = weight * mask
        avg_factor = torch.sum(target > 0, dim=1).float()
        # embed()
        loss = F.binary_cross_entropy_with_logits(pred, onehot, weight, reduction='none').sum(dim=1).sum(dim=1)
        loss = loss.div_(avg_factor.clamp(min=1.0)).mean()

        return loss


def smooth_l1_loss(pred, target, beta=1.0, reduction='elementwise_mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction == 0:
        return loss
    elif reduction == 1:
        return loss.sum() / pred.numel()
    elif reduction == 2:
        return loss.sum()


class SmoothL1Loss(nn.Module):

    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, offset, target, cls_target):
        # ignore background and ignore label
        # offset B*N*4
        # target B*N*4
        # cls: B*N*C
        # bg_mask = cls_target == 0
        # ig_mask = cls_target == -1

        mask = cls_target > 0  # ig_mask * bg_mask
        # avg_factor = torch.sum(cls_target > 0).float().item() / 4 + 1e-6
        mask = mask.float()
        loss_raw = (smooth_l1_loss(offset, target, beta=0.11, reduction='none').sum(2)) * mask
        loss = loss_raw.sum(dim=1).div_(mask.sum(dim=1).clamp(min=1.0)).mean()
        return loss