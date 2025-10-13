import torch
from torch import Tensor


def cross_entropy(inputs: Tensor, targets: Tensor, *, reduction: str = "mean_batch") -> Tensor:
    inputs_max = inputs.max(dim=-1, keepdim=True).values
    inputs_stable = inputs - inputs_max
    exp_inputs_stable = torch.exp(inputs_stable)
    log_sum_exp_inputs_stable = torch.log(exp_inputs_stable.sum(dim=-1, keepdim=True))

    indices = targets.long().unsqueeze(-1)
    gathered_inputs_stable = torch.gather(inputs_stable, dim=-1, index=indices)

    l = (-gathered_inputs_stable + log_sum_exp_inputs_stable).squeeze(-1)

    if reduction == "none":
        return l
    if reduction == "mean_batch":
        return l.mean(dim=0)
    if reduction == "mean":
        return l.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def diffusion_cross_entropy(logits: Tensor, targets: Tensor, mask: Tensor, p_mask: Tensor) -> Tensor:
    per_token = cross_entropy(logits, targets, reduction="none")
    mask_f = mask.to(per_token.dtype)
    weighted = (per_token * mask_f) / p_mask
    denom = targets.shape[0] * targets.shape[1]
    return weighted.sum() / max(denom, 1)


__all__ = ["cross_entropy", "diffusion_cross_entropy"]
