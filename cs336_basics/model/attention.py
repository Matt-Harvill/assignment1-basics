import torch
import torch.nn as nn
import einops
import math

from .linear import Linear
from .rope import RotaryPositionalEmbedding

from .softmax import softmax


def scaled_dot_product_attention(
    k: torch.Tensor, q: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None = None
) -> torch.Tensor:
    # Get keys shape
    seq_len, d_k = k.shape[-2], k.shape[-1]

    # Compute dot products
    raw_scores = einops.einsum(q, k, "... s1 d_k, ... s2 d_k -> ... s1 s2")

    # Divide by sqrt of d_k
    scaled_scores = raw_scores / math.sqrt(d_k)

    # Apply masking before computing softmax
    if not attn_mask is not None:
        attn_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)

    # Fill Falses with -inf
    masked_scores = scaled_scores.masked_fill(~attn_mask, float("-inf"))

    # Compute softmax over last dimension
    scores = softmax(masked_scores, dim=-1)

    # Multiply by values
    result = einops.einsum(scores, v, "... s1 s2, ... s2 d_v -> ... s1 d_v")

    return result


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: int | None = None,
        dtype: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = self.d_model // self.num_heads

        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.rope = rope

        self.K = Linear(self.d_model, self.d_model)
        self.Q = Linear(self.d_model, self.d_model)
        self.V = Linear(self.d_model, self.d_model)
        self.O = Linear(self.d_model, self.d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # First do the scaled_dot_product_attention
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)

        # Split into heads
        k = einops.rearrange(k, "b s (nh dh) -> b nh s dh", nh=self.num_heads, dh=self.d_head)
        q = einops.rearrange(q, "b s (nh dh) -> b nh s dh", nh=self.num_heads, dh=self.d_head)
        v = einops.rearrange(v, "b s (nh dh) -> b nh s dh", nh=self.num_heads, dh=self.d_head)

        # Optionally apply RoPE to k and q
        if self.rope is not None and token_positions is not None:
            k = self.rope.forward(k, token_positions)
            q = self.rope.forward(q, token_positions)

        # Compute scaled dot product attention (no mask implies causal)
        attn_out = scaled_dot_product_attention(k, q, v)

        # Reshape attn_out before output projection
        attn_out = einops.rearrange(attn_out, "b nh s dh -> b s (nh dh)", nh=self.num_heads, dh=self.d_head)

        # Apply output projection
        return self.O(attn_out)
