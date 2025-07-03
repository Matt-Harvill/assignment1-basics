import numpy as np
from numpy.typing import NDArray
import torch
import random


def load_data(
    x: NDArray[np.int_], batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # Sample batch_size random starting indices
    sequence_indices = random.choices(range(len(x) - context_length), k=batch_size)

    # Create index arrays for batch indexing
    # Each row represents the indices for one sequence
    batch_indices = np.array(sequence_indices)[:, None]  # Shape: (batch_size, 1)
    context_indices = np.arange(context_length)  # Shape: (context_length,)

    # Create input indices: batch_indices + context_indices
    input_indices = batch_indices + context_indices  # Shape: (batch_size, context_length)
    # Create target indices: batch_indices + context_indices + 1
    target_indices = batch_indices + context_indices + 1  # Shape: (batch_size, context_length)

    # Use advanced indexing to get the sequences
    input_sequences = x[input_indices]  # Shape: (batch_size, context_length)
    target_sequences = x[target_indices]  # Shape: (batch_size, context_length)

    # Convert to tensors and move to device
    input_tensor = torch.tensor(input_sequences, dtype=torch.long, device=device)
    target_tensor = torch.tensor(target_sequences, dtype=torch.long, device=device)

    return (input_tensor, target_tensor)
