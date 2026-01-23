import math
import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineAnnealingWarmRestarts(LRScheduler):
    """
    Warmup + CosineAnnealingWarmRestarts

    Args:
        optimizer: torch optimizer
        warmup_epochs (float): 线性 warmup 持续的 epoch 数
        T_0 (float): 第一个 cosine 周期长度（不含 warmup 部分）
        eta_min (float): 最小学习率
        T_mult (float): 周期长度倍率，和官方 CosineAnnealingWarmRestarts 一致
        last_epoch (int or float): 起始 epoch
    """
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        T_0,
        eta_min=0.0,
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

    # PyTorch 会在 step() 之后调用这个函数来拿当前 lr
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
            # 周期递增：T_i, T_i*T_mult, T_i*T_mult^2, ...
            t_i = t
            while t_i >= T_i:
                t_i -= T_i
                T_i *= self.T_mult

        # 现在 t_i ∈ [0, T_i)
        # 标准 cosine 退火公式：从 base_lr → eta_min 再回到 eta_min
        cos_inner = math.pi * t_i / T_i

        return [
            self.eta_min + (base_lr - self.eta_min) * (1.0 + math.cos(cos_inner)) / 2.0
            for base_lr in self.base_lrs
        ]

