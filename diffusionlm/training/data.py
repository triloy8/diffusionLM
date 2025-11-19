from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import torch
import random


@dataclass
class DiffusionBatch:
    noisy_inputs: torch.Tensor
    clean_targets: torch.Tensor
    mask: torch.Tensor
    p_mask: torch.Tensor
    metadata: Dict[str, Any]


def _rand_uniform(shape, *, device: torch.device, generator: torch.Generator | None = None):
    if generator is not None:
        return torch.rand(shape, generator=generator, device=device)
    return torch.rand(shape, device=device)


def _rand_int(high: int, *, device: torch.device, generator: torch.Generator | None = None) -> int:
    if generator is not None:
        return int(torch.randint(1, high + 1, (1,), generator=generator, device=device).item())
    return random.randint(1, high)


def get_batch(
    dataset,
    batch_size: int,
    context_length: int,
    device: str,
    *,
    mask_token_id: int,
    noise_epsilon: float = 1e-3,
    random_trunc_prob: float = 0.01,
    generator: torch.Generator | None = None,
) -> DiffusionBatch:
    device_obj = torch.device(device)
    rng_device = generator.device if generator is not None and hasattr(generator, "device") else torch.device("cpu")

    clean_targets = dataset.draw(batch_size=batch_size, context_length=context_length).to(device_obj, dtype=torch.long)

    random_trunc_applied = False
    if random_trunc_prob > 0:
        trunc_flag = _rand_uniform((1,), device=rng_device, generator=generator)
        if bool((trunc_flag < random_trunc_prob).item()):
            max_len = clean_targets.shape[1]
            trunc_len = _rand_int(max_len, device=rng_device, generator=generator)
            clean_targets = clean_targets[:, :trunc_len]
            random_trunc_applied = True

    batch_size, seq_len = clean_targets.shape

    t = _rand_uniform((batch_size,), device=device_obj, generator=generator)
    p_mask = (1.0 - noise_epsilon) * t[:, None] + noise_epsilon
    mask_rand = _rand_uniform((batch_size, seq_len), device=device_obj, generator=generator)
    mask = mask_rand < p_mask

    mask_token_tensor = torch.full_like(clean_targets, fill_value=mask_token_id)
    noisy_inputs = torch.where(mask, mask_token_tensor, clean_targets)

    metadata: Dict[str, Any] = {
        "random_truncation_applied": random_trunc_applied,
        "sequence_length": seq_len,
        "mask_ratio": float(mask.float().mean().detach().cpu().item()),
    }

    return DiffusionBatch(
        noisy_inputs=noisy_inputs,
        clean_targets=clean_targets,
        mask=mask,
        p_mask=p_mask,
        metadata=metadata,
    )


__all__ = ["DiffusionBatch", "get_batch"]
