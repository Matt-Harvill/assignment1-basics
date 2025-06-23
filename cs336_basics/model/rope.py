"""
Deliverable: Implement a class RotaryPositionalEmbedding that applies RoPE to the input
tensor.
The following interface is recommended:
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) Construct the
RoPE module and create buffers if needed.
theta: float Θ value for the RoPE
d_k: int dimension of query and key vectors
max_seq_len: int Maximum sequence length that will be inputted
device: torch.device | None = None Device to store the buffer on
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor Process
an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
that you should tolerate x with an arbitrary number of batch dimensions. You should assume
that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
x along the sequence dimension.
You should use the token positions to slice your (possibly precomputed) cos and sin tensors along
the sequence dimension.
To test your implementation, complete [adapters.run_rope] and make sure it passes uv run
pytest -k test_rope.
"""

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # I want to initialize a 2d tensor of the RoPE angles of shape (max_seq_len, d_k // 2)
        # Specifically, angles[i][k] = i / (theta ** (2 * k / d_k))
        i = torch.arange(max_seq_len, device=self.device).unsqueeze(1)
        k = torch.arange(d_k // 2, device=self.device).unsqueeze(0)
        angles = i / (theta ** (2 * k / d_k))

        self.register_buffer("sines", torch.sin(angles), persistent=False)
        self.register_buffer("cosines", torch.cos(angles), persistent=False)

        # Type annotations for linter
        self.sines: torch.Tensor
        self.cosines: torch.Tensor

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_k = x.size()
        assert d_k == self.d_k, f"Input with embed dimension {d_k} is incompatible with RoPE embed dimension {self.d_k}"

        # Get the buffer indexed by the input token_positions
        sines = self.sines[token_positions]  # shape (batch_size, seq_len, d_k // 2)
        cosines = self.cosines[token_positions]  # shape (batch_size, seq_len, d_k // 2)

        # Slice the odds and evens of input x as a (odds), b (evens)
        x_odd = x[..., ::2]  # shape (batch_size, seq_len, d_k // 2)
        x_even = x[..., 1::2]  # shape (batch_size, seq_len, d_k // 2)

        # Perform the (acos$ - bsin$, asin$ + bcos$) per 2D
        odds = x_odd * cosines - x_even * sines
        evens = x_odd * sines + x_even * cosines

        # Fill in the result with the odd and even values
        result = torch.zeros(batch_size, seq_len, self.d_k)
        result[..., ::2] = odds
        result[..., 1::2] = evens

        return result
