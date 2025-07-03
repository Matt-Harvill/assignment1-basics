import torch
from collections.abc import Iterable


def clip_gradients(params: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """
        Write a function that implements gradient clipping. Your function should take a list of parameters
    and a maximum ℓ2-norm. It should modify each parameter gradient in place. Use ε = 10−6 (the
    PyTorch default). Then, implement the adapter [adapters.run_gradient_clipping] and make sure
    it passes uv run pytest -k test_gradient_clipping
    """
    # Collect all gradients that exist
    gradients = []
    for param in params:
        if param.grad is not None:
            gradients.append(param.grad)

    if not gradients:
        return

    # Calculate the total L2 norm of all gradients
    total_norm_squared = torch.tensor(0.0, device=gradients[0].device, dtype=gradients[0].dtype)
    for grad in gradients:
        total_norm_squared += torch.sum(grad**2)

    total_norm = torch.sqrt(total_norm_squared)

    # If norm exceeds max_l2_norm, scale all gradients
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)  # Add epsilon for numerical stability
        for grad in gradients:
            grad *= clip_coef
