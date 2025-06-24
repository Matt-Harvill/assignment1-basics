import torch


def softmax(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    # Subtract max value for numerical stability
    max_value, _ = torch.max(input=x, dim=dim, keepdim=True)
    x -= max_value

    x = torch.exp(x)

    return x / torch.sum(x, dim=dim, keepdim=True)
