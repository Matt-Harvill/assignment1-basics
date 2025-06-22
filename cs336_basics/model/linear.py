import torch.nn as nn
import torch
from math import sqrt
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()  # Do all nn.Module setup
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # Weights
        self.W = nn.Parameter(torch.Tensor(size=(out_features, in_features), device=self.device))

        # Initialize Weights
        var = 2 / (in_features + out_features)
        std = sqrt(var)
        nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3 * std, b=-3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "b s d_in, d_out d_in -> b s d_out")
