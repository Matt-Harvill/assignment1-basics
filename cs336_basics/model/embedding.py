import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.embed_matrix = nn.Parameter(
            torch.empty(size=(num_embeddings, embedding_dim), device=device, dtype=self.dtype)
        )
        nn.init.trunc_normal_(self.embed_matrix, mean=0, std=1, a=-3, b=-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has dimensions (batch_size, seq_len) and I'm returning a Tensor with shape (b, s, d)

        return self.embed_matrix[x]
