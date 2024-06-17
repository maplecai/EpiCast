import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss, BCEWithLogitsLoss, BCELoss, CrossEntropyLoss, L1Loss
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



# class WeightedBCELoss(nn.Module):
#     def __init__(self, class_weights=None, reduction='mean'):
#         super().__init__()
#         self.class_weights = class_weights
#         self.reduction = reduction
    
#     def forward(self, input, target):
#         weight = (target == 1) * self.class_weights[1] + (target == 0) * self.class_weights[0]
#         loss = F.binary_cross_entropy(input, target, weight=weight, reduction=self.reduction)
#         return loss


# class L1L2MixLoss(nn.Module):
#     def __init__(self, l1_weight=0.5, l2_weight=0.5, pearson_weight=0.0):
#         super(L1L2MixLoss, self).__init__()
#         self.l1_weight = l1_weight
#         self.l2_weight = l2_weight
#         # self.pearson_weight = pearson_weight
#         self.l1_loss = nn.L1Loss()
#         self.l2_loss = nn.MSELoss()
#         # self.pearson_loss = Pearson()

#     def forward(self, pred, target):
#         l1_loss = self.l1_loss(pred, target)
#         l2_loss = self.l2_loss(pred, target)
#         # pearson_loss = self.pearson_loss(pred, target)
#         loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss # + self.pearson_weight * pearson_loss
#         return loss
    

# class NLLLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
#         return F.nll_loss(input, target)



class PoissonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # 确保预测值大于0以避免对数运算错误
        input_clamped = input.clamp(min=1e-8)
        loss = torch.mean(input_clamped - target * torch.log(input_clamped))
        return loss



class MultiTaskLoss(nn.Module):
    def __init__(self, loss_list, n_tasks=2, use_uncertainty_weight=False):
        super().__init__()
        self.loss_funcs = {
            'MSELoss': F.mse_loss,
            'BCELoss': F.binary_cross_entropy,
            'CrossEntropyLoss': F.cross_entropy,
            'BCEWithLogitsLoss': F.binary_cross_entropy_with_logits,
            'L1Loss': F.l1_loss,
        }

        self.loss_list = loss_list
        self.n_tasks = n_tasks
        self.use_uncertainty_weight = use_uncertainty_weight

        self.is_regression = torch.tensor([True if loss == 'MSELoss' else False for loss in self.loss_list])
        if use_uncertainty_weight == True:
            # self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
            self.eta = torch.nn.Parameter(torch.zeros(self.n_tasks)) # eta = log(var)

    def forward(self, output, label, task_idx, reduction='mean'):
        # 假设一个batch内的数据都是同一个task
        if len(task_idx.shape) != 0:
            task_idx = task_idx[0]  # 获取当前任务的索引
        loss_func = self.loss_funcs[self.loss_list[task_idx]]  # 根据任务索引获取损失函数
        loss = loss_func(output, label, reduction=reduction)

        if self.use_uncertainty_weight == True:
            weight = torch.exp(-self.eta) / (self.is_regression+1)
            loss = weight[task_idx]*loss + self.eta[task_idx] / 2

        return loss


if __name__ == '__main__':
    pass
