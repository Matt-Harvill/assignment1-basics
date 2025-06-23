import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        # Initialize gains to ones
        self.gains = nn.Parameter(torch.ones(self.d_model, device=self.device, dtype=self.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save in_dtype, but upcast to fp32 so we don't overflow when squaring x
        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean_square = torch.mean(x * x, dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)

        result = self.gains * x / rms

        return result.to(in_dtype)
