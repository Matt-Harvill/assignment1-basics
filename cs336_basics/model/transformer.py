import torch
import torch.nn as nn

from .embedding import Embedding
from .linear import Linear
from .attention import CausalMultiHeadSelfAttention
from .feed_forward import FeedForwardSwiGLU
from .rms_norm import RMSNorm
from .rope import RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
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

        self.ln1 = RMSNorm(d_model=self.d_model)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=self.d_model, num_heads=self.num_heads, rope=self.rope, device=self.device, dtype=self.dtype
        )

        self.ln2 = RMSNorm(d_model=self.d_model)
        self.ffn = FeedForwardSwiGLU(d_model=self.d_model, d_ff=self.d_ff, device=self.device, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multihead attn
        x = self.attn(self.ln1(x)) + x

        # FFN
        x = self.ffn(self.ln2(x)) + x

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        rope_theta: float,
        num_layers: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.rope_theta = rope_theta
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype

        self.token_embeddings = Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.d_model, device=self.device, dtype=self.dtype
        )

        self.rope = RotaryPositionalEmbedding(
            theta=self.rope_theta,
            d_k=self.d_model // self.num_heads,
            max_seq_len=self.context_length,
            device=self.device,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model=self.d_model, num_heads=self.num_heads, d_ff=self.d_ff, rope=self.rope)
                for _ in range(self.num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model=self.d_model)
        self.lm_head = Linear(
            in_features=self.d_model, out_features=self.vocab_size, device=self.device, dtype=self.dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed the inputs
        x = self.token_embeddings.forward(x)

        # Pass the embeddings through the transformer layers
        for transformer_block in self.layers:
            x = transformer_block.forward(x)

        x = self.lm_head.forward(self.ln_final(x))

        return x
