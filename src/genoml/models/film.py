import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    FiLM: Feature-wise Linear Modulation
    y = gamma(c) * x + beta(c)

    支持输入 x 形状：
      - [B, C]
      - [B, C, L]
      - [B, C, H, W]
      - [B, ..., C] （可选：通过 channel_dim 指定 C 在哪个维度）

    条件 c 形状：
      - [B, cond_dim]
    """
    def __init__(
        self,
        num_features: int,     # C
        cond_dim: int,         # condition 的维度
        hidden_dim: int = 128, # MLP 中间层（可改小/改大/设为 0 表示线性）
        channel_dim: int = 1,  # x 的通道维度位置，默认 [B, C, ...] => 1
        affine: bool = True,   # 是否乘 gamma；若 False 则只加 beta
        init_identity: bool = True,  # 是否初始化为接近 identity（训练更稳）
    ):
        super().__init__()
        self.num_features = num_features
        self.cond_dim = cond_dim
        self.channel_dim = channel_dim
        self.affine = affine

        out_dim = num_features * (2 if affine else 1)

        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Linear(cond_dim, out_dim)

        if init_identity:
            # 让初始输出接近：gamma=1, beta=0（或 beta=0）
            nn.init.zeros_(self.net[-1].weight if isinstance(self.net, nn.Sequential) else self.net.weight)
            if isinstance(self.net, nn.Sequential):
                nn.init.zeros_(self.net[-1].bias)
            else:
                nn.init.zeros_(self.net.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: 输入特征
        c: condition 向量 [B, cond_dim]
        """
        B = x.shape[0]
        params = self.net(c)  # [B, out_dim]

        if self.affine:
            gamma, beta = params.chunk(2, dim=-1)  # [B, C], [B, C]
            # 常见做法：gamma = 1 + delta，避免训练初期破坏特征尺度
            gamma = 1.0 + gamma
        else:
            beta = params
            gamma = None

        # 把 [B, C] reshape 成能广播到 x 的形状
        # 目标：在 channel_dim 位置放 C，其余维度为 1
        shape = [B] + [1] * (x.dim() - 1)
        shape[self.channel_dim] = self.num_features

        beta = beta.view(*shape)
        if gamma is not None:
            gamma = gamma.view(*shape)
            return x * gamma + beta
        else:
            return x + beta
