import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Notes
    - logits shape: ..., v
    - targets shape: ..., 1

    1. Subtract max element out for stability
    2. Then compute log(sum)
    3. subtract target index out (rearranged formula to get this)
    """

    # Subtract out largest logit per batch
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
    logits -= max_logits

    # Compute log(sum(exp))
    exp_logits = torch.exp(logits)
    sum_logits = torch.sum(exp_logits, dim=-1)
    log_sum_logits = torch.log(sum_logits)

    # Index to get the logits from target indices
    # Use gather to select the logits corresponding to target indices
    # targets shape: [8] -> we need to add a dimension for gather
    targets_expanded = targets.unsqueeze(-1)  # Shape: [8, 1]
    label_logits = torch.gather(logits, dim=-1, index=targets_expanded)

    # log_sum_logits has shape [8], label_logits has shape [8, 1]
    # We need to squeeze label_logits to match
    label_logits = label_logits.squeeze(-1)

    return torch.mean(log_sum_logits - label_logits)
