import torch.nn as nn
import torch
from math import sqrt
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()  # Do all nn.Module setup
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # Weights
        self.weight = nn.Parameter(torch.empty(size=(out_features, in_features), device=self.device, dtype=self.dtype))

        # Initialize Weights
        var = 2 / (in_features + out_features)
        std = sqrt(var)
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=-3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "b s d_in, d_out d_in -> b s d_out")
