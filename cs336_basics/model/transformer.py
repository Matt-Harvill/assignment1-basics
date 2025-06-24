import torch
import torch.nn as nn

from .attention import CausalMultiHeadSelfAttention
from .feed_forward import FeedForwardSwiGLU
from .rms_norm import RMSNorm
from .rope import RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
    """
    Implement the pre-norm Transformer block as described in §3.5 and illustrated in Figure 2. Your
    Transformer block should accept (at least) the following parameters.
    d_model: int Dimensionality of the Transformer block inputs.
    num_heads: int Number of heads to use in multi-head self-attention.
    d_ff: int Dimensionality of the position-wise feed-forward inner layer

    To test your implementation, implement the adapter [adapters.run_transformer_block]. Then
    run uv run pytest -k test_transformer_block to test your implementation.
    Deliverable: Transformer block code that passes the provided tests"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope
        self.device = device
        self.dtype = dtype

        self.pre_attn_norm = RMSNorm(d_model=self.d_model)
        self.mha = CausalMultiHeadSelfAttention(
            d_model=self.d_model, num_heads=self.num_heads, rope=self.rope, device=self.device, dtype=self.dtype
        )

        self.pre_ffn_norm = RMSNorm(d_model=self.d_model)
        self.ffn = FeedForwardSwiGLU(d_model=self.d_model, d_ff=self.d_ff, device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multihead attn
        x = self.mha(self.pre_attn_norm(x)) + x

        # FFN
        x = self.ffn(self.pre_ffn_norm(x)) + x

        return x
