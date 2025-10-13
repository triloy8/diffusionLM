from collections.abc import Callable
from typing import Optional
import math
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros(p.shape, device=p.grad.device))
                v = state.get("v", torch.zeros(p.shape, device=p.grad.device))
                t = state.get("t", 0)
                grad = p.grad.data

                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad ** 2

                lr_t = lr * (math.sqrt(1 - betas[1] ** (t + 1)) / (1 - betas[0] ** (t + 1)))

                p.data -= lr_t * (m / (torch.sqrt(v) + eps))
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss

__all__ = ["AdamW"]
