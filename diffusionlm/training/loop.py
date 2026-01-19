import os
from pathlib import Path
import torch
from contextlib import nullcontext
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import numpy as np
from typing import Optional, Callable
from logger import Logger
from diffusionlm.training.data import DiffusionBatch
from diffusionlm.training.loss import cross_entropy


def train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    # data
    train_data,
    val_data,
    # batching
    batch_size: int,
    context_length: int,
    device: str,
    # schedule
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    # iterations
    max_train_iteration: int | None,
    max_val_iteration: int | None,
    val_freq_iteration: int,
    grad_accum_steps: int = 1,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    # regularization
    grad_clip_max_l2_norm: float,
    # checkpointing
    ckpting_save_iter: int,
    ckpting_save_folder: Path | str | None,
    # helpers
    get_batch,
    lr_cosine_schedule,
    gradient_clipping,
    compute_loss,
    checkpoint_callback: Optional[
        Callable[[int, torch.nn.Module, torch.optim.Optimizer, Optional[dict], Optional[torch.amp.GradScaler]], None]
    ] = None,
    prepare_batch: Optional[Callable[[object], object]] = None,
    extract_model_inputs: Optional[Callable[[object], torch.Tensor]] = None,
    batch_generator: torch.Generator | None = None,
    # logging
    logger: Optional[Logger] = None,
    train_loss_ema_decay: float = 0.0,
    scaler: Optional[torch.amp.GradScaler] = None,
    # optional logging helpers
    activation_norms: dict | None = None,
    log_activation_norms: bool = False,
    log_weight_norms: bool = False,
    log_grad_norms: bool = False,
    log_p_mask_bucket_loss: bool = False,
    p_mask_bucket_edges: list[float] | None = None,
    val_log_every: int = 0,
    val_log_samples: int = 0,
    val_sample_decode: Optional[Callable[[list[int]], str]] = None,
    # DDP/unified-loop hooks (optional)
    sync_gradients: Optional[Callable[[], None]] = None,
    reduce_metric: Optional[Callable[[float], float]] = None,
    # rank-zero policy
    is_rank_zero: bool = True,
    step_callback: Optional[Callable[[int, torch.nn.Module, torch.optim.Optimizer, float, float], None]] = None,
    start_iteration: int = 0,
    skip_validation: bool = False,
):
    """A minimal training loop extracted into a reusable function.

    `compute_loss` consumes model logits and the prepared batch object.
    """
    if compute_loss is None:
        raise ValueError("compute_loss must be provided.")

    prepare = prepare_batch or (lambda batch: batch)

    def _inputs(batch_obj: object) -> torch.Tensor:
        if extract_model_inputs is not None:
            return extract_model_inputs(batch_obj)
        if hasattr(batch_obj, "noisy_inputs"):
            return getattr(batch_obj, "noisy_inputs")
        if isinstance(batch_obj, dict) and "noisy_inputs" in batch_obj:
            return batch_obj["noisy_inputs"]
        raise ValueError("Prepared batch must expose `noisy_inputs` for model input.")

    def _attention_mask(batch_obj: object) -> Optional[torch.Tensor]:
        if hasattr(batch_obj, "attention_mask"):
            return getattr(batch_obj, "attention_mask")
        if isinstance(batch_obj, dict) and "attention_mask" in batch_obj:
            return batch_obj["attention_mask"]
        return None

    def _resolve_p_mask_edges(edges: list[float] | None) -> list[float]:
        if edges is None:
            return [i / 10.0 for i in range(11)]
        cleaned = sorted({float(e) for e in edges})
        if len(cleaned) < 2:
            return [0.0, 1.0]
        return cleaned

    def _p_mask_bucket_payload(logits: torch.Tensor, batch_obj: object) -> Optional[dict]:
        if not hasattr(batch_obj, "p_mask") or not hasattr(batch_obj, "mask") or not hasattr(batch_obj, "clean_targets"):
            return None
        p_mask = getattr(batch_obj, "p_mask", None)
        mask = getattr(batch_obj, "mask", None)
        targets = getattr(batch_obj, "clean_targets", None)
        if p_mask is None or mask is None or targets is None:
            return None
        loss_mask = getattr(batch_obj, "loss_mask", None)
        edges = _resolve_p_mask_edges(p_mask_bucket_edges)
        if len(edges) < 2:
            return None
        with torch.no_grad():
            per_token = cross_entropy(logits, targets, reduction="none")
            mask_f = mask.to(per_token.dtype)
            if loss_mask is not None:
                loss_mask_f = loss_mask.to(per_token.dtype)
                mask_f = mask_f * loss_mask_f
            else:
                loss_mask_f = None
            weighted = (per_token * mask_f) / p_mask
            if loss_mask_f is not None:
                denom = loss_mask_f.sum(dim=1)
            else:
                denom = torch.full(
                    (targets.shape[0],),
                    targets.shape[1],
                    device=per_token.device,
                    dtype=per_token.dtype,
                )
            per_example_loss = weighted.sum(dim=1) / denom.clamp_min(1)
            p_mask_vals = p_mask.view(-1)
            if len(edges) > 2:
                boundaries = torch.tensor(edges[1:-1], device=p_mask_vals.device, dtype=p_mask_vals.dtype)
                bucket_ids = torch.bucketize(p_mask_vals, boundaries)
            else:
                bucket_ids = torch.zeros_like(p_mask_vals, dtype=torch.long)
            payload = {}
            for i in range(len(edges) - 1):
                in_bucket = bucket_ids == i
                count = int(in_bucket.sum().item())
                if count == 0:
                    continue
                mean_val = float(per_example_loss[in_bucket].mean().item())
                if reduce_metric is not None:
                    mean_val = float(reduce_metric(mean_val))
                label = f"{edges[i]:.2f}-{edges[i + 1]:.2f}"
                payload[f"metrics.p_mask_bucket_loss/{label}"] = mean_val
                payload[f"metrics.p_mask_bucket_count/{label}"] = count
            return payload if payload else None

    train_iteration = start_iteration
    tokens_seen = 0
    accum_steps = max(1, int(grad_accum_steps))
    accum_count = 0
    use_amp = bool(amp_enabled) and device.startswith("cuda") and torch.cuda.is_available()
    amp_dtype = amp_dtype.lower()
    amp_torch_dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    if scaler is None:
        scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp and amp_torch_dtype == torch.float16)
    current_lr = None
    train_loss_ema = None
    val_pass_count = 0
    last_val_metrics: Optional[dict] = None

    def _format_sample_tokens(tokens: list[int]) -> object:
        if val_sample_decode is None:
            return tokens
        return val_sample_decode(tokens)

    def _extract_sample_payload(val_inputs, val_logits, val_batch, max_samples: int) -> Optional[dict]:
        if max_samples <= 0:
            return None
        count = min(int(max_samples), int(val_inputs.shape[0]))
        if count <= 0:
            return None
        targets = getattr(val_batch, "clean_targets", None)
        if targets is None and isinstance(val_batch, dict):
            targets = val_batch.get("clean_targets")
        if targets is None:
            return None
        inputs_list = val_inputs[:count].detach().cpu().tolist()
        preds_list = val_logits[:count].argmax(dim=-1).detach().cpu().tolist()
        targets_list = targets[:count].detach().cpu().tolist()
        return {
            "inputs": inputs_list,
            "predictions": preds_list,
            "targets": targets_list,
        }
    while True:
        model.train()
        raw_train_batch = get_batch(
            dataset=train_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            generator=batch_generator,
        )

        train_batch = prepare(raw_train_batch)
        train_inputs = _inputs(train_batch)
        train_attn_mask = _attention_mask(train_batch)

        # schedule (set LR for this iteration before forward/step)
        logged_lr = None
        group_lrs = {}
        if accum_count == 0:
            for idx, param_group in enumerate(optimizer.param_groups):
                group_max_lr = float(param_group.get("max_lr", max_learning_rate))
                group_min_lr = float(param_group.get("min_lr", min_learning_rate))
                group_warmup = int(param_group.get("warmup_iters", warmup_iters))
                group_cosine = int(param_group.get("cosine_cycle_iters", cosine_cycle_iters))
                new_lr = lr_cosine_schedule(train_iteration, group_max_lr, group_min_lr, group_warmup, group_cosine)
                param_group["lr"] = new_lr
                group_name = str(param_group.get("name", f"group_{idx}"))
                group_lrs[f"metrics.lr/{group_name}"] = float(new_lr)
                if logged_lr is None:
                    logged_lr = float(new_lr)
                    current_lr = float(new_lr)
            if logger is not None and logged_lr is not None:
                logger.log(
                    {
                        "phase": "train",
                        "metrics.lr": logged_lr,
                        **group_lrs,
                    },
                    step=train_iteration,
                )

        # forward
        autocast_ctx = torch.autocast("cuda", dtype=amp_torch_dtype) if use_amp else nullcontext()
        with autocast_ctx:
            if train_attn_mask is not None:
                train_logits = model(train_inputs, attention_mask=train_attn_mask)
            else:
                train_logits = model(train_inputs)
            train_loss = compute_loss(train_logits, train_batch)
        scaled_loss = train_loss / accum_steps
        if logger is not None and device.startswith("cuda") and torch.cuda.is_available():
            logger.log(
                {
                    "phase": "train",
                    "metrics.cuda.mem_allocated_mb/fwd": float(torch.cuda.memory_allocated() / 1e6),
                    "metrics.cuda.mem_reserved_mb/fwd": float(torch.cuda.memory_reserved() / 1e6),
                    "metrics.cuda.max_mem_allocated_mb/fwd": float(torch.cuda.max_memory_allocated() / 1e6),
                },
                step=train_iteration,
            )
        # Optionally reduce metrics across ranks before logging
        tloss_val = float(train_loss.item())
        if reduce_metric is not None:
            tloss_val = float(reduce_metric(tloss_val))
        if train_loss_ema_decay > 0:
            if train_loss_ema is None:
                train_loss_ema = float(tloss_val)
            else:
                decay = float(train_loss_ema_decay)
                train_loss_ema = decay * float(train_loss_ema) + (1.0 - decay) * float(tloss_val)
        if logger is not None:
            payload = {"phase": "train", "metrics.train_loss": tloss_val}
            if train_loss_ema is not None:
                payload["metrics.train_loss_ema"] = float(train_loss_ema)
            logger.log(payload, step=train_iteration)
            batch_metadata = getattr(train_batch, "metadata", None)
            if isinstance(batch_metadata, dict):
                token_count = batch_metadata.get("token_count")
            else:
                token_count = None
            if token_count is None:
                token_count = int(train_inputs.numel())
            tokens_seen += int(token_count)
            logger.log(
                {
                    "phase": "train",
                    "metrics.tokens_batch_rank0": int(token_count),
                    "metrics.tokens_seen_rank0": int(tokens_seen),
                    "metrics.inputs_shape/batch": int(train_inputs.shape[0]),
                    "metrics.inputs_shape/seq_len": int(train_inputs.shape[1]),
                },
                step=train_iteration,
            )
            if device.startswith("cuda") and torch.cuda.is_available():
                logger.log(
                    {
                        "phase": "train",
                        "metrics.cuda.mem_allocated_mb": float(torch.cuda.memory_allocated() / 1e6),
                        "metrics.cuda.mem_reserved_mb": float(torch.cuda.memory_reserved() / 1e6),
                        "metrics.cuda.max_mem_allocated_mb": float(torch.cuda.max_memory_allocated() / 1e6),
                    },
                    step=train_iteration,
                )
            if isinstance(batch_metadata, dict):
                if "mask_ratio" in batch_metadata:
                    logger.log(
                        {
                            "phase": "train",
                            "metrics.mask_ratio": float(batch_metadata["mask_ratio"]),
                        },
                        step=train_iteration,
                    )
                if "random_truncation_applied" in batch_metadata:
                    logger.log(
                        {
                            "phase": "train",
                            "metrics.random_truncation_applied": float(
                                batch_metadata["random_truncation_applied"]
                            ),
                        },
                        step=train_iteration,
                    )
            if log_p_mask_bucket_loss:
                bucket_payload = _p_mask_bucket_payload(train_logits, train_batch)
                if bucket_payload:
                    logger.log({"phase": "train", **bucket_payload}, step=train_iteration)

        # activation norms (if hooks populate activation_norms)
        if logger is not None and log_activation_norms and activation_norms is not None and len(activation_norms) > 0:
            vals = list(activation_norms.values())
            logger.log({
                "phase": "train",
                "metrics.activation_norms/mean": float(np.mean(vals)),
                "metrics.activation_norms/max": float(np.max(vals)),
                "metrics.activation_norms/min": float(np.min(vals)),
                **{f"metrics.activation_norms/{k}": float(v) for k, v in activation_norms.items()},
            }, step=train_iteration)

        # backward
        if accum_count == 0:
            optimizer.zero_grad()
        if scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        if logger is not None and device.startswith("cuda") and torch.cuda.is_available():
            logger.log(
                {
                    "phase": "train",
                    "metrics.cuda.mem_allocated_mb/bwd": float(torch.cuda.memory_allocated() / 1e6),
                    "metrics.cuda.mem_reserved_mb/bwd": float(torch.cuda.memory_reserved() / 1e6),
                    "metrics.cuda.max_mem_allocated_mb/bwd": float(torch.cuda.max_memory_allocated() / 1e6),
                },
                step=train_iteration,
            )
        accum_count += 1
        if accum_count >= accum_steps:
            pre_clip_l2 = None
            if logger is not None and log_grad_norms:
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                l2_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]))
                l2_val = float(l2_norm.item())
                if reduce_metric is not None:
                    l2_val = float(reduce_metric(l2_val))
                logger.log({"phase": "train", "metrics.grad/l2_norm": l2_val}, step=train_iteration)
                pre_clip_l2 = l2_norm
            # Optional DDP gradient synchronization before optimizer step
            if sync_gradients is not None:
                sync_gradients()
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            gradient_clipping(parameters=model.parameters(), max_l2_norm=grad_clip_max_l2_norm)
            if logger is not None and log_grad_norms:
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if grads:
                    post_l2_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]))
                    post_l2_val = float(post_l2_norm.item())
                    if reduce_metric is not None:
                        post_l2_val = float(reduce_metric(post_l2_val))
                    clipped = 0.0
                    clip_ratio = 1.0
                    if pre_clip_l2 is not None:
                        pre_val = float(pre_clip_l2.item())
                        clipped = float(pre_val > grad_clip_max_l2_norm)
                        if pre_val > 0.0:
                            clip_ratio = post_l2_val / pre_val
                    logger.log(
                        {
                            "phase": "train",
                            "metrics.grad/l2_norm_post_clip": post_l2_val,
                            "metrics.grad/clipped": clipped,
                            "metrics.grad/clip_ratio": float(clip_ratio),
                        },
                        step=train_iteration,
                    )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            accum_count = 0

        # weight norms
        if logger is not None and log_weight_norms and accum_count == 0:
            norms = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    norms[name] = float(param.data.norm().item())
            vals = list(norms.values())
            if vals:
                logger.log({
                    "phase": "train",
                    "metrics.weight_norms/mean": float(np.mean(vals)),
                    "metrics.weight_norms/max": float(np.max(vals)),
                    "metrics.weight_norms/min": float(np.min(vals)),
                    **{f"metrics.weight_norms/{k}": v for k, v in norms.items()},
                }, step=train_iteration)
                if device.startswith("cuda") and torch.cuda.is_available():
                    logger.log(
                        {
                            "phase": "train",
                            "metrics.cuda.mem_allocated_mb/weight_norms": float(torch.cuda.memory_allocated() / 1e6),
                            "metrics.cuda.mem_reserved_mb/weight_norms": float(torch.cuda.memory_reserved() / 1e6),
                            "metrics.cuda.max_mem_allocated_mb/weight_norms": float(
                                torch.cuda.max_memory_allocated() / 1e6
                            ),
                        },
                        step=train_iteration,
                    )

        # validation
        if (not skip_validation) and train_iteration % val_freq_iteration == 0:
            model.eval()
            val_pass_count += 1
            log_samples_now = (
                is_rank_zero
                and logger is not None
                and val_log_every > 0
                and val_log_samples > 0
                and (val_pass_count % val_log_every == 0)
            )
            val_iteration = 0
            running_val_loss = 0.0
            val_tokens = 0
            sample_payload = None
            while True:
                with torch.no_grad():
                    raw_val_batch = get_batch(
                        dataset=val_data,
                        batch_size=batch_size,
                        context_length=context_length,
                        device=device,
                        generator=batch_generator,
                    )
                    val_batch = prepare(raw_val_batch)
                    val_inputs = _inputs(val_batch)
                    val_attn_mask = _attention_mask(val_batch)
                    with autocast_ctx:
                        if val_attn_mask is not None:
                            val_logits = model(val_inputs, attention_mask=val_attn_mask)
                        else:
                            val_logits = model(val_inputs)
                        val_loss = compute_loss(val_logits, val_batch)
                    running_val_loss += float(val_loss.item())
                    val_tokens += int(val_inputs.numel())
                    if log_samples_now and sample_payload is None:
                        sample_payload = _extract_sample_payload(
                            val_inputs,
                            val_logits,
                            val_batch,
                            val_log_samples,
                        )
                val_iteration += 1
                if max_val_iteration is not None and val_iteration >= max_val_iteration:
                    break
            avg_val_loss = running_val_loss / (max_val_iteration if max_val_iteration else val_iteration)
            vloss_val = float(avg_val_loss)
            if reduce_metric is not None:
                vloss_val = float(reduce_metric(vloss_val))
            if logger is not None:
                logger.log(
                    {
                        "phase": "val",
                        "metrics.val_loss": vloss_val,
                        "metrics.val_tokens": int(val_tokens),
                    },
                    step=train_iteration,
                )
            last_val_metrics = {"val_loss": vloss_val}
            if log_samples_now and sample_payload is not None and logger is not None:
                rows = []
                for inputs, preds, targets in zip(
                    sample_payload["inputs"],
                    sample_payload["predictions"],
                    sample_payload["targets"],
                ):
                    rows.append(
                        {
                            "noisy_input": _format_sample_tokens(list(inputs)),
                            "prediction": _format_sample_tokens(list(preds)),
                            "target": _format_sample_tokens(list(targets)),
                        }
                    )
                logger.log_table("val/samples", rows, step=train_iteration)

        # checkpoints
        if (
            checkpoint_callback is not None
            and train_iteration > 0
            and train_iteration % ckpting_save_iter == 0
            and ckpting_save_folder is not None
        ):
            amp_scaler = scaler if scaler is not None and scaler.is_enabled() else None
            checkpoint_callback(train_iteration, model, optimizer, last_val_metrics, amp_scaler)

        if step_callback is not None:
            step_callback(
                train_iteration,
                model,
                optimizer,
                tloss_val,
                float(current_lr) if current_lr is not None else float("nan"),
            )

        # termination
        if max_train_iteration is not None and train_iteration >= max_train_iteration:
            break
        train_iteration += 1
