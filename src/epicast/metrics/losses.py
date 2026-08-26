import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss, BCEWithLogitsLoss, BCELoss, CrossEntropyLoss, PoissonNLLLoss, HuberLoss
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits, cross_entropy, binary_cross_entropy, l1_loss, huber_loss
from torch import Tensor


import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEWithLogitsLossWrapper(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean'):
        super().__init__()
        self.reduction = reduction

        if pos_weight is not None:
            if isinstance(pos_weight, (int, float)):
                pos_weight = [pos_weight]
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, input, target):
        loss = F.binary_cross_entropy_with_logits(
            input,
            target,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )
        return loss

class MaskedMSELoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mask = torch.isfinite(input) & torch.isfinite(target)

        diff = torch.where(mask, input - target, torch.zeros_like(input))
        loss = diff.square()

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "mean":
            denom = mask.sum().to(loss.dtype).clamp_min(1.0)
            return loss.sum() / denom

        raise ValueError(f"Unsupported reduction: {self.reduction}")



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class WeightedMaskedMSELoss(nn.Module):
    """
    Weighted masked MSE loss.

    thresholds and weights define a piecewise weight:
    - len(weights) == len(thresholds) + 1
    - for ascending thresholds = [t1, t2, ..., tk]
      the weight intervals are:
        target <= t1                  -> weights[0]
        t1 < target <= t2             -> weights[1]
        ...
        target > tk                   -> weights[k]

    example:
        thresholds=[1.0], weights=[1.0, 2.0]
        weight 1 for target <= 1, weight 2 for target > 1
    """

    def __init__(
        self,
        thresholds: list[float],
        weights: list[float],
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}")

        if len(weights) != len(thresholds) + 1:
            raise ValueError(
                "len(weights) must be len(thresholds) + 1, "
                f"but got len(thresholds)={len(thresholds)}, len(weights)={len(weights)}"
            )

        if sorted(thresholds) != list(thresholds):
            raise ValueError("thresholds must be in ascending order")

        self.reduction = reduction

        # a buffer, so that to(device) moves it along
        self.register_buffer("thresholds", torch.tensor(thresholds, dtype=torch.float32))
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def _get_weight_map(self, target: Tensor) -> Tensor:
        """
        Per-element weight derived from target.
        bucketize rule:
            thresholds=[1.0, 3.0]
            idx:
              target <= 1.0   -> 0
              1.0 < target <= 3.0 -> 1
              target > 3.0    -> 2
        """
        thresholds = self.thresholds.to(device=target.device, dtype=target.dtype)
        weights = self.weights.to(device=target.device, dtype=target.dtype)

        # right=False:
        # target <= thresholds[0] -> 0
        # thresholds[i-1] < target <= thresholds[i] -> i
        # target > thresholds[-1] -> len(thresholds)
        idx = torch.bucketize(target, thresholds, right=False)
        return weights[idx]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mask = torch.isfinite(input) & torch.isfinite(target)

        loss = F.mse_loss(input, target, reduction="none")
        weight_map = self._get_weight_map(target)

        weighted_loss = loss * weight_map
        weighted_loss = torch.where(mask, weighted_loss, torch.zeros_like(weighted_loss))

        if self.reduction == "none":
            return weighted_loss

        if self.reduction == "sum":
            return weighted_loss.sum()

        # weighted mean: divide by the weight sum of the valid positions
        valid_weights = torch.where(mask, weight_map, torch.zeros_like(weight_map))
        denom = valid_weights.sum().clamp_min(1.0)
        return weighted_loss.sum() / denom


class MaskedHuberLoss(nn.Module):
    def __init__(self, reduction: str = 'mean', delta: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.delta = delta


    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mask = torch.isfinite(input) & torch.isfinite(target)
        loss = F.huber_loss(input, target, reduction="none", delta=self.delta)  # elementwise
        loss = torch.where(mask, loss, torch.zeros_like(loss))

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()

        denom = mask.sum().to(loss.dtype).clamp_min(1.0)
        return loss.sum() / denom





# adapted from boda2 repository
class L1KLmixed(nn.Module):
    """
    A custom loss module that combines L1 loss with Kullback-Leibler (KL) divergence loss.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
        alpha (float, optional): Scaling factor for the L1 loss term. Default is 1.0.
        beta (float, optional): Scaling factor for the KL divergence loss term. Default is 1.0.

    Attributes:
        reduction (str): The reduction method applied to the losses.
        alpha (float): Scaling factor for the L1 loss term.
        beta (float): Scaling factor for the KL divergence loss term.
        MSE (nn.L1Loss): The L1 loss function.
        KL (nn.KLDivLoss): The Kullback-Leibler divergence loss function.

    Methods:
        forward(preds, targets):
            Calculate the combined loss by combining L1 and KL divergence losses.

    Example:
        loss_fn = L1KLmixed()
        loss = loss_fn(predictions, targets)
    """
    
    def __init__(self, reduction='mean', alpha=1.0, beta=1.0):
        """
        Initialize the L1KLmixed loss module.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the losses. Default is 'mean'.
            alpha (float, optional): Scaling factor for the L1 loss term. Default is 1.0.
            beta (float, optional): Scaling factor for the KL divergence loss term. Default is 1.0.

        Returns:
            None
        """
        super().__init__()
        
        self.reduction = reduction
        self.alpha = alpha
        self.beta  = beta
        
        self.MSE = nn.L1Loss(reduction=reduction.replace('batch',''))
        self.KL  = nn.KLDivLoss(reduction=reduction, log_target=True)
        
    def forward(self, preds, targets):
        """
        Calculate the combined loss by combining L1 and KL divergence losses.

        Args:
            preds (Tensor): The predicted tensor.
            targets (Tensor): The target tensor.

        Returns:
            Tensor: The combined loss tensor.
        """
        preds_log_prob  = preds   - torch.logsumexp(preds, dim=-1, keepdim=True)
        target_log_prob = targets - torch.logsumexp(targets, dim=-1, keepdim=True)
        
        MSE_loss = self.MSE(preds, targets)
        KL_loss  = self.KL(preds_log_prob, target_log_prob)
        
        combined_loss = MSE_loss.mul(self.alpha) + \
                        KL_loss.mul(self.beta)
        
        return combined_loss.div(self.alpha+self.beta)



class CellTypeSpecificHuberLoss(nn.Module):
    def __init__(self, input_shape=(170, 4), res_weight=1.0, reduction='mean', delta=1.0):
        super().__init__()
        self.input_shape = input_shape
        self.res_weight = res_weight
        self.reduction = reduction
        self.delta = delta

    def forward(self, input, target):
        # input shape = (n, c, a), target shape = (n, c, a)
        input = input.view(-1, *self.input_shape)
        target = target.view(-1, *self.input_shape)
        input_mean = input.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)
        input_res = input - input_mean
        target_res = target - target_mean
        loss_mean = F.huber_loss(input_mean, target_mean, reduction=self.reduction, delta=self.delta)
        loss_res = F.huber_loss(input_res, target_res, reduction=self.reduction, delta=self.delta)
        loss = loss_mean + self.res_weight * loss_res
        return loss
