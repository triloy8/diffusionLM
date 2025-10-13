import os
from pathlib import Path
import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import numpy as np
from typing import Optional, Callable, Tuple
from logger import Logger


def train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    # data
    np_arr_train_data,
    np_arr_valid_data,
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
    cross_entropy,
    lr_cosine_schedule,
    gradient_clipping,
    save_checkpoint,
    batch_generator: torch.Generator | None = None,
    # logging
    logger: Optional[Logger] = None,
    # optional logging helpers
    activation_norms: dict | None = None,
    log_activation_norms: bool = False,
    log_weight_norms: bool = False,
    # DDP/unified-loop hooks (optional)
    shard_batch: Optional[Callable[[torch.Tensor, torch.Tensor, int, int], Tuple[torch.Tensor, torch.Tensor]]] = None,
    sync_gradients: Optional[Callable[[], None]] = None,
    reduce_metric: Optional[Callable[[float], float]] = None,
    world_size: int = 1,
    local_rank: int = 0,
    # rank-zero policy
    is_rank_zero: bool = True,
    step_callback: Optional[Callable[[int, torch.nn.Module, torch.optim.Optimizer, float, float], None]] = None,
    start_iteration: int = 0,
):
    """A minimal training loop extracted into a reusable function.

    The caller supplies batching, loss, schedule, clipping, and checkpoint helpers.
    If `log` is provided, it is called as `log(dict, step=iteration)`.
    """
    train_iteration = start_iteration
    while True:
        model.train()
        # If sharding is requested, draw a larger batch so each rank gets a slice
        eff_batch_size = batch_size * world_size if shard_batch is not None else batch_size
        train_batch_sampled_sequence, train_batch_sampled_ids = get_batch(
            dataset=np_arr_train_data,
            batch_size=eff_batch_size,
            context_length=context_length,
            device=device,
            generator=batch_generator,
        )

        # Optional rank sharding for DDP
        if shard_batch is not None:
            train_batch_sampled_sequence, train_batch_sampled_ids = shard_batch(
                train_batch_sampled_sequence, train_batch_sampled_ids, world_size, local_rank
            )

        # forward
        train_logits = model(train_batch_sampled_sequence)
        train_loss = cross_entropy(train_logits, train_batch_sampled_ids).mean()
        # Optionally reduce metrics across ranks before logging
        tloss_val = float(train_loss.item())
        if reduce_metric is not None:
            tloss_val = float(reduce_metric(tloss_val))
        if logger is not None:
            logger.log({"phase": "train", "metrics.train_loss": tloss_val}, step=train_iteration)

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
        new_lr = lr_cosine_schedule(train_iteration, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        if logger is not None:
            logger.log({"phase": "train", "metrics.lr": float(new_lr)}, step=train_iteration)

        # validation
        if train_iteration % val_freq_iteration == 0:
            model.eval()
            val_iteration = 0
            running_val_loss = 0.0
            while True:
                with torch.no_grad():
                    val_batch_sampled_sequence, val_batch_sampled_ids = get_batch(
                        dataset=np_arr_valid_data,
                        batch_size=batch_size,
                        context_length=context_length,
                        device=device,
                        generator=batch_generator,
                    )
                    val_logits = model(val_batch_sampled_sequence)
                    val_loss = cross_entropy(val_logits, val_batch_sampled_ids).mean()
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


def train_loop_ddp(
    ddp_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    np_arr_train_data,
    np_arr_valid_data,
    batch_size: int,
    context_length: int,
    device: str,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    max_train_iteration: int | None,
    max_val_iteration: int | None,
    val_freq_iteration: int,
    grad_clip_max_l2_norm: float,
    ckpting_save_iter: int,
    ckpting_save_folder: Path | str | None,
    get_batch,
    cross_entropy,
    lr_cosine_schedule,
    gradient_clipping,
    save_checkpoint,
    logger: Optional[Logger] = None,
    activation_norms: dict | None = None,
    log_activation_norms: bool = False,
    log_weight_norms: bool = False,
    num_nodes: int,
    node_rank: int,
    local_rank: int,
    world_size: int,
):
    """Thin wrapper calling unified train_loop with DDP hooks."""

    def _shard(seq: torch.Tensor, ids: torch.Tensor, ws: int, rk: int):
        return torch.chunk(seq, ws, dim=0)[rk], torch.chunk(ids, ws, dim=0)[rk]

    def _sync():
        ddp_model.finish_gradient_synchronization()

    # Metrics reduction is identity here; Step 6 will provide allreduce_mean helper.
    _reduce = None

    return train_loop(
        ddp_model,
        optimizer,
        np_arr_train_data=np_arr_train_data,
        np_arr_valid_data=np_arr_valid_data,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
        max_train_iteration=max_train_iteration,
        max_val_iteration=max_val_iteration,
        val_freq_iteration=val_freq_iteration,
        grad_clip_max_l2_norm=grad_clip_max_l2_norm,
        ckpting_save_iter=ckpting_save_iter,
        ckpting_save_folder=ckpting_save_folder,
        get_batch=get_batch,
        cross_entropy=cross_entropy,
        lr_cosine_schedule=lr_cosine_schedule,
        gradient_clipping=gradient_clipping,
        save_checkpoint=save_checkpoint,
        logger=logger,
        activation_norms=activation_norms,
        log_activation_norms=log_activation_norms,
        log_weight_norms=log_weight_norms,
        shard_batch=_shard,
        sync_gradients=_sync,
        reduce_metric=_reduce,
        world_size=world_size,
        local_rank=local_rank,
        is_rank_zero=(local_rank == 0),
    )
