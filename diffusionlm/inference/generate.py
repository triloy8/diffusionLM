from __future__ import annotations

import torch
from diffusionlm.inference.sampling import add_gumbel_noise, compute_transfer_schedule


@torch.no_grad()
def diffusion_generate(
    model,
    prompt_indices: torch.Tensor,
    *,
    mask_id: int,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate sequences via the diffusion reverse process."""

    if prompt_indices.dim() != 2:
        raise ValueError("prompt_indices must be 2D (batch, seq)")
    if gen_length <= 0:
        raise ValueError("gen_length must be > 0")
    if block_length <= 0:
        raise ValueError("block_length must be > 0")
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")

    blocks = gen_length // block_length
    if steps % blocks != 0:
        raise ValueError("steps must be divisible by the number of blocks")
    steps_per_block = steps // blocks

    device = prompt_indices.device
    batch_size, prompt_len = prompt_indices.shape
    total_len = prompt_len + gen_length

    context_limit = getattr(model, "context_length", None)
    if context_limit is not None and total_len > int(context_limit):
        raise ValueError("prompt length + gen_length exceeds model context_length")

    x = torch.full(
        (batch_size, total_len),
        fill_value=mask_id,
        device=device,
        dtype=prompt_indices.dtype,
    )
    x[:, :prompt_len] = prompt_indices

    for block_idx in range(blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = block_start + block_length
        block_mask = (x[:, block_start:block_end] == mask_id)
        transfer_counts = compute_transfer_schedule(block_mask, steps_per_block)

        for step_idx in range(steps_per_block):
            mask_index = (x == mask_id)
            logits = model(x)
            logits = add_gumbel_noise(logits, temperature, generator=generator)
            predictions = torch.argmax(logits, dim=-1)
            predictions = torch.where(mask_index, predictions, x)

            confidence = torch.rand(
                (batch_size, total_len),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            confidence[:, block_end:] = float("-inf")
            confidence = torch.where(mask_index, confidence, torch.full_like(confidence, float("-inf")))

            transfer_mask = torch.zeros_like(mask_index)
            for b in range(batch_size):
                k = int(transfer_counts[b, step_idx].item())
                if k <= 0:
                    continue
                # Guard in case of numerical issues when fewer masked tokens remain
                available = confidence[b] > float("-inf")
                if available.sum() < k:
                    k = int(available.sum().item())
                if k <= 0:
                    continue
                topk_indices = torch.topk(confidence[b], k=k, dim=-1).indices
                transfer_mask[b, topk_indices] = True

            x = torch.where(transfer_mask, predictions, x)

    return x


# Backwards-compatible alias
generate = diffusion_generate


__all__ = [
    "diffusion_generate",
    "generate",
]
