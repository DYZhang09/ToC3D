import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from mmdet.models import LOSSES


def kl_div_loss(
    pred,
    target,
    weight = None,
    reduction = 'mean',
    avg_factor = None,
    log_target = True,
):
    loss = F.kl_div(pred, target, reduction='none', log_target=log_target)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class KLDivLoss(nn.Module):
    '''warpper of torch.nn.KLDivLoss'''
    def __init__(
        self, 
        reduction: str = 'mean', 
        log_target: bool = False,
        loss_weight = 1.0,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target
        self.loss_weight = loss_weight
    
    def forward(
        self, 
        pred: Tensor, 
        target: Tensor,
        weight: Tensor = None,
        avg_factor=None,
        reduction_override=None
    ) -> Tensor:
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_kl = self.loss_weight * kl_div_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            log_target=self.log_target
        )
        return loss_kl