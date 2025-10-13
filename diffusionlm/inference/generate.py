import torch
from diffusionlm.inference.sampling import softmax, top_p_filter


@torch.no_grad()
def generate(
    model,
    in_indices: torch.Tensor,
    steps: int,
    *,
    temperature: float = 1.0,
    p: float = 0.0,
    eos_token_id: int | None = None,
    context_length: int | None = None,
):
    """Stateless decode that grows sequences by `steps` using top-p sampling.

    Args:
        model: Autoregressive model returning logits for next-token prediction.
        in_indices: Tensor of shape (B, T0) with token ids.
        steps: Number of new tokens to generate.
        temperature: Softmax temperature.
        p: Top-p (nucleus) sampling threshold.
        eos_token_id: If provided, stop per sequence after sampling eos.
        context_length: Optional context window; defaults to model.context_length if available.
    Returns:
        Tensor of shape (B, T0 + <=steps) with appended tokens.
    """
    device = in_indices.device
    B = in_indices.shape[0]
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    ctx = context_length
    if ctx is None:
        ctx = getattr(model, "context_length", in_indices.shape[1])

    for _ in range(steps):
        context_indices = in_indices if in_indices.shape[1] <= ctx else in_indices[:, -ctx:]
        logits = model(context_indices)
        logits = logits[:, -1, :] / temperature
        probs = softmax(logits, dim=-1)
        filtered = top_p_filter(probs, p)
        index_next = torch.multinomial(filtered, num_samples=1)
        if eos_token_id is not None:
            index_next[finished] = eos_token_id
        in_indices = torch.cat([in_indices, index_next], dim=1)
        if eos_token_id is not None:
            finished = finished | (index_next.squeeze(1) == eos_token_id)
            if finished.all():
                break
    return in_indices
