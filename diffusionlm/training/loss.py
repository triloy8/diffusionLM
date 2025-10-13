import torch


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs_max = inputs.max(dim=-1, keepdim=True).values
    inputs_stable = inputs - inputs_max
    exp_inputs_stable = torch.exp(inputs_stable)
    log_sum_exp_inputs_stable = torch.log(exp_inputs_stable.sum(dim=-1, keepdim=True))

    indices = targets.long().unsqueeze(-1)
    gathered_inputs_stable = torch.gather(inputs_stable, dim=-1, index=indices)

    l = -gathered_inputs_stable + log_sum_exp_inputs_stable
    return l.mean(dim=0)

__all__ = ["cross_entropy"]
