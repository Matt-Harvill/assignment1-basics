import torch
import einops
import math

from .softmax import softmax


def scaled_dot_product_attention(
    k: torch.Tensor, q: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None = None
) -> torch.Tensor:
    # Get keys shape
    _, seq_len, d_k = k.shape

    # Compute dot products
    raw_scores = einops.einsum(q, k, "b s1 d_k, b s2 d_k -> b s1 s2")

    # Divide by sqrt of d_k
    scaled_scores = raw_scores / math.sqrt(d_k)

    # Apply masking before computing softmax
    if not attn_mask is not None:
        attn_mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=1)

    # Fill Falses with -inf
    masked_scores = scaled_scores.masked_fill(~attn_mask, float("-inf"))

    # Compute softmax over last dimension
    scores = softmax(masked_scores, dim=-1)

    # Multiply by values
    result = einops.einsum(scores, v, "b s1 s2, b s2 d_v -> b s1 d_v")

    return result
