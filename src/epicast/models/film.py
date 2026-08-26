import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    FiLM: Feature-wise Linear Modulation
    y = gamma(c) * x + beta(c)

    Accepted shapes of x:
      - [B, C]
      - [B, C, L]
      - [B, C, H, W]
      - [B, ..., C] (modulated_dim says which axis carries C)

    Shape of the condition c:
      - [B, cond_dim]
    """
    def __init__(
        self,
        num_features: int,     # C
        cond_dim: int,         # dimension of the mask
        hidden_dim: int = 128, # hidden layer of the MLP; 0 makes it linear
        modulated_dim: int = 1,  # channel axis of x; 1 for [B, C, ...]
        affine: bool = True,   # multiply by gamma; if False only beta is added
        init_identity: bool = True,  # start close to identity, which trains more stably
    ):
        super().__init__()
        self.num_features = num_features
        self.cond_dim = cond_dim
        self.modulated_dim = modulated_dim
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
            # start out at gamma=1, beta=0
            nn.init.zeros_(self.net[-1].weight if isinstance(self.net, nn.Sequential) else self.net.weight)
            if isinstance(self.net, nn.Sequential):
                nn.init.zeros_(self.net[-1].bias)
            else:
                nn.init.zeros_(self.net.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: input features
        c: mask vector [B, cond_dim]
        """
        B = x.shape[0]
        params = self.net(c)  # [B, out_dim]

        if self.affine:
            gamma, beta = params.chunk(2, dim=-1)  # [B, C], [B, C]
            # gamma = 1 + delta keeps the feature scale intact early in training
            gamma = 1.0 + gamma
        else:
            beta = params
            gamma = None

        # reshape [B, C] so that it broadcasts against x
        # C goes to modulated_dim, every other axis is 1
        shape = [B] + [1] * (x.dim() - 1)
        shape[self.modulated_dim] = self.num_features

        beta = beta.view(*shape)
        if gamma is not None:
            gamma = gamma.view(*shape)
            return x * gamma + beta
        else:
            return x + beta
