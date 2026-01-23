from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
import random


@dataclass
class DiffusionBatch:
    noisy_inputs: torch.Tensor
    clean_targets: torch.Tensor
    mask: torch.Tensor
    p_mask: torch.Tensor
    attention_mask: torch.Tensor | None
    loss_mask: torch.Tensor | None
    metadata: Dict[str, Any]


@dataclass
class AutoregressiveBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    attention_mask: torch.Tensor | None
    loss_mask: torch.Tensor | None
    metadata: Dict[str, Any]


def _rand_uniform(shape, *, device: torch.device, generator: torch.Generator | None = None):
    if generator is not None:
        return torch.rand(shape, generator=generator, device=device)
    return torch.rand(shape, device=device)


def _rand_int(high: int, *, device: torch.device, generator: torch.Generator | None = None) -> int:
    if generator is not None:
        return int(torch.randint(1, high + 1, (1,), generator=generator, device=device).item())
    return random.randint(1, high)


def _draw_clean_targets(
    dataset,
    batch_size: int,
    context_length: int,
    device: str,
    *,
    random_trunc_prob: float = 0.01,
    min_length: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    device_obj = torch.device(device)
    rng_device = generator.device if generator is not None and hasattr(generator, "device") else torch.device("cpu")

    attention_mask = None
    drawn = dataset.draw(batch_size=batch_size, context_length=context_length)
    if isinstance(drawn, tuple):
        clean_targets, attention_mask = drawn
    else:
        clean_targets = drawn
    clean_targets = clean_targets.to(device_obj, dtype=torch.long)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device_obj, dtype=torch.bool)

    random_trunc_applied = False
    if random_trunc_prob > 0:
        trunc_flag = _rand_uniform((1,), device=rng_device, generator=generator)
        if bool((trunc_flag < random_trunc_prob).item()):
            max_len = clean_targets.shape[1]
            trunc_len = _rand_int(max_len, device=rng_device, generator=generator)
            if min_length is not None and max_len >= min_length and trunc_len < min_length:
                trunc_len = min_length
            clean_targets = clean_targets[:, :trunc_len]
            random_trunc_applied = True

    if attention_mask is not None and random_trunc_applied:
        attention_mask = attention_mask[:, : clean_targets.shape[1]]

    return clean_targets, attention_mask, random_trunc_applied


def get_batch(
    dataset,
    batch_size: int,
    context_length: int,
    device: str,
    *,
    mask_token_id: int,
    noise_epsilon: float = 1e-3,
    random_trunc_prob: float = 0.01,
    p_mask_override: Optional[float] = None,
    deterministic_mask: bool = False,
    generator: torch.Generator | None = None,
) -> DiffusionBatch:
    clean_targets, attention_mask, random_trunc_applied = _draw_clean_targets(
        dataset,
        batch_size,
        context_length,
        device,
        random_trunc_prob=random_trunc_prob,
        min_length=2,
        generator=generator,
    )
    device_obj = clean_targets.device

    batch_size, seq_len = clean_targets.shape

    if p_mask_override is not None:
        p_mask = torch.full((batch_size, 1), float(p_mask_override), device=device_obj)
    else:
        t = _rand_uniform((batch_size,), device=device_obj, generator=generator)
        p_mask = (1.0 - noise_epsilon) * t[:, None] + noise_epsilon

    if deterministic_mask:
        mask_len = (p_mask.view(-1) * seq_len).floor().to(torch.long)
        positions = torch.arange(seq_len, device=device_obj)
        mask = positions[None, :] < mask_len[:, None]
    else:
        mask_rand = _rand_uniform((batch_size, seq_len), device=device_obj, generator=generator)
        mask = mask_rand < p_mask
    if attention_mask is not None:
        mask = mask & attention_mask

    mask_token_tensor = torch.full_like(clean_targets, fill_value=mask_token_id)
    noisy_inputs = torch.where(mask, mask_token_tensor, clean_targets)

    loss_mask = attention_mask
    token_count = int(loss_mask.sum().item()) if loss_mask is not None else int(clean_targets.numel())
    metadata: Dict[str, Any] = {
        "random_truncation_applied": random_trunc_applied,
        "sequence_length": seq_len,
        "mask_ratio": float(mask.float().mean().detach().cpu().item()),
        "token_count": token_count,
        "p_mask_stats": {
            "mean": float(p_mask.mean().detach().cpu().item()),
            "min": float(p_mask.min().detach().cpu().item()),
            "max": float(p_mask.max().detach().cpu().item()),
            "std": float(p_mask.std(unbiased=False).detach().cpu().item()),
            "inv_mean": float((1.0 / p_mask).mean().detach().cpu().item()),
            "inv_max": float((1.0 / p_mask).max().detach().cpu().item()),
        },
    }

    return DiffusionBatch(
        noisy_inputs=noisy_inputs,
        clean_targets=clean_targets,
        mask=mask,
        p_mask=p_mask,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        metadata=metadata,
    )


def get_autoregressive_batch(
    dataset,
    batch_size: int,
    context_length: int,
    device: str,
    *,
    random_trunc_prob: float = 0.01,
    generator: torch.Generator | None = None,
) -> AutoregressiveBatch:
    clean_targets, attention_mask, random_trunc_applied = _draw_clean_targets(
        dataset,
        batch_size,
        context_length,
        device,
        random_trunc_prob=random_trunc_prob,
        generator=generator,
    )
    if clean_targets.shape[1] < 2:
        raise ValueError("context_length must be >= 2 for autoregressive training")

    inputs = clean_targets[:, :-1]
    targets = clean_targets[:, 1:]
    loss_mask = attention_mask[:, 1:] if attention_mask is not None else None

    seq_len = inputs.shape[1]
    causal_mask = torch.ones((seq_len, seq_len), device=inputs.device, dtype=torch.bool).tril()
    if attention_mask is not None:
        key_mask = attention_mask[:, :-1]
        query_mask = attention_mask[:, :-1]
        causal_mask = causal_mask[None, :, :] & key_mask[:, None, :] & query_mask[:, :, None]
    else:
        causal_mask = causal_mask[None, :, :].expand(inputs.shape[0], -1, -1)

    token_count = int(loss_mask.sum().item()) if loss_mask is not None else int(targets.numel())
    metadata: Dict[str, Any] = {
        "random_truncation_applied": random_trunc_applied,
        "sequence_length": seq_len,
        "token_count": token_count,
    }

    return AutoregressiveBatch(
        inputs=inputs,
        targets=targets,
        attention_mask=causal_mask,
        loss_mask=loss_mask,
        metadata=metadata,
    )

__all__ = ["DiffusionBatch", "AutoregressiveBatch", "get_batch", "get_autoregressive_batch"]
