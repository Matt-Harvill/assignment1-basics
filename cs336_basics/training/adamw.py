from collections.abc import Callable
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize all the moments to 0
        for group in self.param_groups:
            for p in group["params"]:
                weight_shape = p.data.shape
                self.state[p]["m"] = torch.zeros(weight_shape)
                self.state[p]["v"] = torch.zeros(weight_shape)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.

                # Apply weight decay
                p.data -= lr * self.weight_decay * p.data

                # Update first and second running moments
                state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
                state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad**2)

                # Apply bias corrections
                m_hat = state["m"] / (1 - self.beta1**t)
                v_hat = state["v"] / (1 - self.beta2**t)

                # Update the parameters
                p.data -= lr * m_hat / (torch.sqrt(v_hat) + self.eps)  # Update weight tensor in-place.

                state["t"] = t + 1  # Increment iteration number.

        return loss
