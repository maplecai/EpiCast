import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss, BCEWithLogitsLoss, BCELoss, CrossEntropyLoss, L1Loss, PoissonNLLLoss
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits, cross_entropy, binary_cross_entropy, l1_loss

class MyMSELoss(nn.Module):
    def __init__(self, reduction='mean', allow_none=True):
        super().__init__()
        self.reduction = reduction
        self.allow_none = allow_none

    def forward(self, input, target, reduction=None):
        if reduction is None:
            reduction = self.reduction
        
        if self.allow_none:
            self.mask = ~torch.isnan(target) & ~torch.isnan(input)
            input = input[self.mask]
            target = target[self.mask]
        
        loss = F.mse_loss(input, target, reduction=reduction)
        return loss



class MyBCELoss(nn.Module):
    def __init__(self, reduction='mean', allow_none=True):
        super().__init__()
        self.reduction = reduction
        self.allow_none = allow_none

    def forward(self, input, target, reduction=None):
        if reduction is None:
            reduction = self.reduction
        
        if self.allow_none:
            # print(target.shape, input.shape)
            self.mask = ~torch.isnan(target) & ~torch.isnan(input)
            input = input[self.mask]
            target = target[self.mask]
        
        loss = F.binary_cross_entropy(input, target, reduction=reduction)
        return loss






# class PearsonCorr(nn.Module):
#     def __init__(self, dim: int = -1, eps: float = 1e-8):
#         super().__init__()
#         self.dim = dim
#         self.eps = eps

#     def forward(self, x: torch.Tensor, y: torch.Tensor):
#         x_mean = x.mean(dim=self.dim, keepdim=True)
#         y_mean = y.mean(dim=self.dim, keepdim=True)

#         xm = x - x_mean
#         ym = y - y_mean

#         cov = (xm * ym).sum(dim=self.dim)
#         x_var = (xm ** 2).sum(dim=self.dim)
#         y_var = (ym ** 2).sum(dim=self.dim)

#         return cov / (torch.sqrt(x_var) * torch.sqrt(y_var) + self.eps)


# class SpearmanCorr(nn.Module):
#     def __init__(self, dim: int = -1, eps: float = 1e-8):
#         super().__init__()
#         self.dim = dim
#         self.eps = eps
#         self.pearson = PearsonCorr(dim=self.dim, eps=self.eps)

#     def rankdata(self, x: torch.Tensor):
#         # ordinal rank differ from scipy
#         tmp = x.argsort(dim=self.dim)
#         ranks = torch.zeros_like(tmp, dtype=torch.float)
#         idx = torch.arange(x.size(self.dim), device=x.device, dtype=torch.float)
#         ranks.scatter_(self.dim, tmp, idx)
#         return ranks

#     def forward(self, x: torch.Tensor, y: torch.Tensor):
#         rx = self.rankdata(x)
#         ry = self.rankdata(y)
#         return self.pearson(rx, ry)



# class WeightedBCELoss(nn.Module):
#     def __init__(self, class_weights=None, reduction='mean'):
#         super().__init__()
#         self.class_weights = class_weights
#         self.reduction = reduction
    
#     def forward(self, input, target):
#         weight = (target == 1) * self.class_weights[1] + (target == 0) * self.class_weights[0]
#         loss = F.binary_cross_entropy(input, target, weight=weight, reduction=self.reduction)
#         return loss


if __name__ == '__main__':
    pass
