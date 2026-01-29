from __future__ import annotations

import math

import torch
from trainkit.inference.sampling import add_gumbel_noise, compute_transfer_schedule, softmax, top_p_filter


@torch.no_grad()
def diffusion_generate(
    model,
    prompt_indices: torch.Tensor,
    *,
    mask_id: int,
    eos_token_id: int | None = None,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float = 0.0,
    top_p: float | None = None,
    cfg_scale: float = 0.0,
    remasking: str = "random",
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate sequences via the diffusion reverse process."""

    if prompt_indices.dim() != 2:
        raise ValueError("prompt_indices must be 2D (batch, seq)")
    if block_length <= 0:
        raise ValueError("block_length must be > 0")
    if steps <= 0:
        raise ValueError("steps must be > 0")

    if gen_length <= 0:
        return prompt_indices

    blocks = max(1, math.ceil(gen_length / block_length))
    if steps < blocks:
        raise ValueError("steps must be >= number of blocks")
    base_steps = steps // blocks
    extra_steps = steps % blocks

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
    prompt_index = (x != mask_id)

    for block_idx in range(blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = min(block_start + block_length, total_len)
        block_steps = base_steps + (1 if block_idx < extra_steps else 0)
        if block_steps <= 0:
            block_steps = 1
        block_mask = (x[:, block_start:block_end] == mask_id)
        transfer_counts = compute_transfer_schedule(block_mask, block_steps)

        for step_idx in range(block_steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                logits = model(torch.cat([x, un_x], dim=0))
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
            else:
                logits = model(x)

            if logits_eos_inf and eos_token_id is not None:
                logits[:, :, eos_token_id] = float("-inf")

            if top_p is not None:
                probs = softmax(logits, dim=-1)
                probs = top_p_filter(probs, float(top_p))
                logits = torch.where(
                    probs > 0,
                    logits,
                    torch.full_like(logits, float("-inf")),
                )

            logits_with_noise = add_gumbel_noise(logits, temperature, generator=generator)
            predictions = torch.argmax(logits_with_noise, dim=-1)
            predictions = torch.where(mask_index, predictions, x)

            if remasking == "low_confidence":
                probs = softmax(logits, dim=-1)
                confidence = torch.squeeze(
                    torch.gather(probs, dim=-1, index=torch.unsqueeze(predictions, -1)),
                    -1,
                )
            elif remasking == "random":
                confidence = torch.rand(
                    (batch_size, total_len),
                    device=device,
                    dtype=torch.float32,
                    generator=generator,
                )
            else:
                raise ValueError(f"Unsupported remasking strategy: {remasking}")

            if confidence_eos_eot_inf and eos_token_id is not None:
                confidence = torch.where(
                    predictions == eos_token_id,
                    torch.full_like(confidence, float("-inf")),
                    confidence,
                )

            confidence[:, block_end:] = float("-inf")
            confidence = torch.where(mask_index, confidence, torch.full_like(confidence, float("-inf")))

            transfer_mask = torch.zeros_like(mask_index)
            for b in range(batch_size):
                k = int(transfer_counts[b, step_idx].item())
                if k <= 0:
                    continue
                available = confidence[b] > float("-inf")
                available_count = int(available.sum().item())
                if available_count == 0:
                    continue
                if available_count < k:
                    k = available_count
                topk_indices = torch.topk(confidence[b], k=k, dim=-1).indices
                transfer_mask[b, topk_indices] = True

            x = torch.where(transfer_mask, predictions, x)

    return x


@torch.no_grad()
def autoregressive_generate(
    model,
    prompt_indices: torch.Tensor,
    *,
    gen_length: int,
    temperature: float = 0.0,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    logits_eos_inf: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate sequences token-by-token with a causal attention mask."""

    if prompt_indices.dim() != 2:
        raise ValueError("prompt_indices must be 2D (batch, seq)")
    if gen_length <= 0:
        return prompt_indices

    x = prompt_indices
    for _ in range(gen_length):
        seq_len = x.shape[1]
        causal = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
        attention_mask = causal.unsqueeze(0).expand(x.shape[0], -1, -1)

        logits = model(x, attention_mask=attention_mask)
        next_logits = logits[:, -1, :]

        if logits_eos_inf and eos_token_id is not None:
            next_logits[:, eos_token_id] = float("-inf")

        if temperature <= 0:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            scaled_logits = next_logits / float(temperature)
            probs = softmax(scaled_logits, dim=-1)
            if top_p is not None:
                probs = top_p_filter(probs, float(top_p))
            next_token = torch.multinomial(probs, 1)
        x = torch.cat([x, next_token], dim=1)

    return x


# Backwards-compatible alias
generate = diffusion_generate


__all__ = [
    "diffusion_generate",
    "autoregressive_generate",
    "generate",
]
