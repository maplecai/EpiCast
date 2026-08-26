import math
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts



# class WarmupCosineAnnealing(_LRScheduler):
#     """
#     Linear warmup, then cosine annealing without restarts.

#     Notes:
#     - call scheduler.step() once per epoch
#     - last_epoch starts at -1, as in the PyTorch schedulers
#     """

#     def __init__(
#         self,
#         optimizer: torch.optim.Optimizer,
#         warmup_epochs: int,
#         max_epochs: int,
#         warmup_start_lr: float = 0.0,
#         eta_min: float = 0.0,
#         last_epoch: int = -1,
#     ):
#         if warmup_epochs < 0:
#             raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
#         if max_epochs <= 0:
#             raise ValueError(f"max_epochs must be > 0, got {max_epochs}")
#         if warmup_epochs >= max_epochs:
#             raise ValueError(
#                 f"warmup_epochs must be < max_epochs, got warmup_epochs={warmup_epochs}, max_epochs={max_epochs}"
#             )

#         self.warmup_epochs = warmup_epochs
#         self.max_epochs = max_epochs
#         self.warmup_start_lr = warmup_start_lr
#         self.eta_min = eta_min

#         super().__init__(optimizer, last_epoch)

#     def get_lr(self):
#         if self.last_epoch < self.warmup_epochs:
#             # linear warmup
#             if self.warmup_epochs == 0:
#                 return list(self.base_lrs)

#             progress = (self.last_epoch + 1) / self.warmup_epochs
#             return [
#                 self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress
#                 for base_lr in self.base_lrs
#             ]

#         # cosine annealing
#         cosine_epochs = self.max_epochs - self.warmup_epochs
#         progress = (self.last_epoch - self.warmup_epochs + 1) / cosine_epochs
#         progress = min(max(progress, 0.0), 1.0)

#         return [
#             self.eta_min
#             + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
#             for base_lr in self.base_lrs
#         ]




class WarmupCosineAnnealingWarmRestarts(LRScheduler):
    """
    Warmup + CosineAnnealingWarmRestarts

    Args:
        optimizer: torch optimizer
        warmup_epochs (float): epochs of linear warmup
        T_0 (float): length of the first cosine period, warmup excluded
        eta_min (float): minimum learning rate
        T_mult (float): period growth factor, as in CosineAnnealingWarmRestarts
        last_epoch (int or float): starting epoch
    """
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        T_0,
        eta_min=1.0e-6,
        T_mult=1.0,
        last_epoch=-1,
    ):
        self.warmup_epochs = float(warmup_epochs)
        self.T_0 = float(T_0)
        self.T_mult = float(T_mult)
        self.eta_min = float(eta_min)

        if self.T_0 <= 0:
            raise ValueError("T_0 must be positive.")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0.")

        super().__init__(optimizer, last_epoch)

    # PyTorch calls this after step() to read the current lr
    def get_lr(self):
        epoch = float(self.last_epoch)
        return self._compute_lr(epoch)

    def _compute_lr(self, epoch: float):
        # 1) Warmup phase
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            warmup_progress = (epoch + 1.0) / max(1.0, self.warmup_epochs)
            warmup_progress = max(0.0, min(warmup_progress, 1.0))
            return [base_lr * warmup_progress for base_lr in self.base_lrs]

        # 2) Cosine annealing + warm restarts
        t = max(epoch - self.warmup_epochs, 0.0)

        T_i = self.T_0

        if self.T_mult == 1.0:
            t_i = t % T_i
        else:
            # the periods grow as T_i, T_i*T_mult, T_i*T_mult^2, ...
            t_i = t
            while t_i >= T_i:
                t_i -= T_i
                T_i *= self.T_mult

        # t_i is now in [0, T_i)
        # the usual cosine annealing, from base_lr down to eta_min
        cos_inner = math.pi * t_i / T_i

        return [
            self.eta_min + (base_lr - self.eta_min) * (1.0 + math.cos(cos_inner)) / 2.0
            for base_lr in self.base_lrs
        ]

