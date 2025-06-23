import torch
import torch.nn as nn

from .linear import Linear


class FeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        # Calculate up_proj dimension (Round up_scale * d_model to nearest 64 for hardware efficiency)
        if not d_ff:
            up_scale = 8 / 3
            up_proj_dim = round(self.d_model * up_scale)
            up_proj_dim_mod_64 = up_proj_dim % 64
            up_proj_dim = up_proj_dim if up_proj_dim_mod_64 == 0 else (up_proj_dim + 64 - up_proj_dim_mod_64)
            self.d_ff: int = int(up_proj_dim)
        else:
            self.d_ff = d_ff

        self.W1 = Linear(self.d_model, self.d_ff, self.device, self.dtype)
        self.W3 = Linear(self.d_model, self.d_ff, self.device, self.dtype)
        self.W2 = Linear(self.d_ff, self.d_model, self.device, self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1 = self.W1(x)
        swish = w1 * torch.sigmoid(w1)
        up_proj = self.W3(x) * swish

        return self.W2(up_proj)
