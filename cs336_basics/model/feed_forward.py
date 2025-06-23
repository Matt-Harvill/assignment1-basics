import torch
import torch.nn as nn

from .linear import Linear


class FeedForwardSwiGLU(nn.Module):
    """
        Deliverable: Implement the SwiGLU feed-forward network, composed of a SiLU activation
    function and a GLU.
    Note: in this particular case, you should feel free to use torch.sigmoid in your implementation
    for numerical stability.
    You should set dff to approximately 8
    3 × dmodel in your implementation, while ensuring that
    the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your
    hardware. To test your implementation against our provided tests, you will need to implement
    the test adapter at [adapters.run_swiglu]. Then, run uv run pytest -k test_swiglu to
    test your implementation
    """

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

    def forward(self, x):
        w1 = self.W1(x)
        swish = w1 * torch.sigmoid(w1)
        up_proj = self.W3(x) * swish

        return self.W2(up_proj)
