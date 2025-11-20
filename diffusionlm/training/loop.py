import os
from pathlib import Path
import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import numpy as np
from typing import Optional, Callable
from logger import Logger
from diffusionlm.training.data import DiffusionBatch


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
    # regularization
    grad_clip_max_l2_norm: float,
    # checkpointing
    ckpting_save_iter: int,
    ckpting_save_folder: Path | str | None,
    # helpers
    get_batch,
    lr_cosine_schedule,
    gradient_clipping,
    save_checkpoint,
    compute_loss,
    prepare_batch: Optional[Callable[[object], object]] = None,
    extract_model_inputs: Optional[Callable[[object], torch.Tensor]] = None,
    batch_generator: torch.Generator | None = None,
    # logging
    logger: Optional[Logger] = None,
    # optional logging helpers
    activation_norms: dict | None = None,
    log_activation_norms: bool = False,
    log_weight_norms: bool = False,
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

    train_iteration = start_iteration
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

        # forward
        train_logits = model(train_inputs)
        train_loss = compute_loss(train_logits, train_batch)
        # Optionally reduce metrics across ranks before logging
        tloss_val = float(train_loss.item())
        if reduce_metric is not None:
            tloss_val = float(reduce_metric(tloss_val))
        if logger is not None:
            logger.log({"phase": "train", "metrics.train_loss": tloss_val}, step=train_iteration)
            batch_metadata = getattr(train_batch, "metadata", None)
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
        optimizer.zero_grad()
        train_loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        l2_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]))
        l2_val = float(l2_norm.item())
        if reduce_metric is not None:
            l2_val = float(reduce_metric(l2_val))
        if logger is not None:
            logger.log({"phase": "train", "metrics.grad_l2_norm": l2_val}, step=train_iteration)
        # Optional DDP gradient synchronization before optimizer step
        if sync_gradients is not None:
            sync_gradients()
        gradient_clipping(parameters=model.parameters(), max_l2_norm=grad_clip_max_l2_norm)
        optimizer.step()

        # weight norms
        if logger is not None and log_weight_norms:
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

        # schedule
        logged_lr = None
        for param_group in optimizer.param_groups:
            group_max_lr = float(param_group.get("max_lr", max_learning_rate))
            group_min_lr = float(param_group.get("min_lr", min_learning_rate))
            group_warmup = int(param_group.get("warmup_iters", warmup_iters))
            group_cosine = int(param_group.get("cosine_cycle_iters", cosine_cycle_iters))
            new_lr = lr_cosine_schedule(train_iteration, group_max_lr, group_min_lr, group_warmup, group_cosine)
            param_group["lr"] = new_lr
            if logged_lr is None:
                logged_lr = float(new_lr)
        if logger is not None and logged_lr is not None:
            logger.log({"phase": "train", "metrics.lr": logged_lr}, step=train_iteration)

        # validation
        if (not skip_validation) and train_iteration % val_freq_iteration == 0:
            model.eval()
            val_iteration = 0
            running_val_loss = 0.0
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
                    val_logits = model(val_inputs)
                    val_loss = compute_loss(val_logits, val_batch)
                    running_val_loss += float(val_loss.item())
                val_iteration += 1
                if max_val_iteration is not None and val_iteration >= max_val_iteration:
                    break
            avg_val_loss = running_val_loss / (max_val_iteration if max_val_iteration else val_iteration)
            vloss_val = float(avg_val_loss)
            if reduce_metric is not None:
                vloss_val = float(reduce_metric(vloss_val))
            if logger is not None:
                logger.log({"phase": "val", "metrics.val_loss": vloss_val}, step=train_iteration)

        # checkpoints
        if (
            is_rank_zero
            and train_iteration > 0
            and train_iteration % ckpting_save_iter == 0
            and ckpting_save_folder is not None
        ):
            ckpting_save_folder = Path(ckpting_save_folder)
            ckpting_save_folder.mkdir(parents=True, exist_ok=True)
            ckpt_file_iter = ckpting_save_folder / f"{train_iteration}.ckpt"
            save_checkpoint(model, optimizer, train_iteration, ckpt_file_iter)
            if logger is not None:
                try:
                    logger.log_artifact(str(ckpt_file_iter), name=str(ckpt_file_iter), type_="checkpoint")
                except Exception:
                    pass

        if step_callback is not None:
            step_callback(train_iteration, model, optimizer, tloss_val, float(new_lr))

        # termination
        if max_train_iteration is not None and train_iteration >= max_train_iteration:
            break
        train_iteration += 1
