import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss, BCEWithLogitsLoss, BCELoss, CrossEntropyLoss, PoissonNLLLoss, HuberLoss
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits, cross_entropy, binary_cross_entropy, l1_loss, huber_loss
from torch import Tensor


class MaskedMSELoss(nn.Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mask = torch.isfinite(input) & torch.isfinite(target)
        loss = F.mse_loss(input, target, reduction='none')
        loss = torch.where(mask, loss, torch.zeros_like(loss))

        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()

        denom = mask.sum().to(loss.dtype).clamp_min(1.0)
        return loss.sum() / denom





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


if __name__ == '__main__':
    pass
