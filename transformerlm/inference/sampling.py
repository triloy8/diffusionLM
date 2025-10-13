import torch

def softmax(x: torch.Tensor, dim: int):
    x_max = x.max(dim=dim, keepdim=True).values
    x_stable = x - x_max
    exp_x = torch.exp(x_stable)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp_x

def top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        argmax = probs.argmax(dim=-1)
        out = torch.zeros_like(probs)
        out.scatter_(-1, argmax.unsqueeze(-1), 1.0)
        return out
    if p >= 1:
        return probs / probs.sum(dim=-1, keepdim=True)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    cutoff = (cumulative >= p).float().argmax(dim=-1) + 1
    filtered = torch.zeros_like(probs)
    batch = probs.shape[0]
    for i in range(batch):
        k = cutoff[i].item()
        sel = sorted_indices[i, :k]
        sel_probs = sorted_probs[i, :k]
        filtered[i, sel] = sel_probs
        filtered[i] /= filtered[i].sum()
    return filtered

