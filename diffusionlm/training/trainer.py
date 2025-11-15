from diffusionlm.models import (
    TransformerLM,
    Linear,
)
from diffusionlm.training.optim import (
    build_optimizer_param_groups,
    resolve_optimizer_cls,
)
from diffusionlm.training.data import get_batch, DiffusionBatch
from diffusionlm.training.loss import diffusion_cross_entropy
from diffusionlm.training.checkpoint import save_checkpoint
from diffusionlm.training.schedule import lr_cosine_schedule
from diffusionlm.training.grad import gradient_clipping
from diffusionlm.training.loop import train_loop

from datetime import datetime
import numpy as np
import os
import random
import torch
import torch.distributed as dist
from functools import partial

from diffusionlm.utils.dtypes import DTYPES
from logger import Logger
from logger import ConsoleLogger, WandbLogger, RankZeroLogger
from ddp import DDP, OptimizerStateSharding
from ddp.utils import broadcast_string, setup_process_group, cleanup_process_group, allreduce_mean


def _seed_everything(seed: int, device: str | torch.device, *, rank: int = 0) -> torch.Generator:
    """Seed python, numpy, torch, and return a generator for reproducible sampling."""

    effective_seed = int(seed) + int(rank)

    if isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = str(device)

    generator_device = "cuda" if device_type == "cuda" and torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(effective_seed)

    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)

    return generator


def _prepare_optimizer_setup(cfg, model):
    optimizer_name = str(getattr(cfg, "optimizer_name", "adamw")).lower()
    setattr(cfg, "optimizer_name", optimizer_name)
    optimizer_cls = resolve_optimizer_cls(optimizer_name)
    param_groups = build_optimizer_param_groups(model, optimizer_name)
    kwargs = {
        "lr": float(cfg.initial_learning_rate),
        "weight_decay": float(cfg.weight_decay),
    }
    betas = getattr(cfg, "betas", (0.9, 0.95))
    beta_tuple = (float(betas[0]), float(betas[1]))
    eps = float(getattr(cfg, "eps", 1e-8))
    if optimizer_name == "adamw":
        kwargs["betas"] = beta_tuple
        kwargs["eps"] = eps
    elif optimizer_name == "muon":
        kwargs["momentum"] = float(getattr(cfg, "muon_momentum", 0.95))
        kwargs["betas"] = beta_tuple
        kwargs["eps"] = eps
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'")
    return optimizer_cls, param_groups, kwargs


def train_transformer(args, *, logger: Logger, run_name: str):
    # checkpoint folder based on run_name provided by logger
    cfg = args
    seed = getattr(cfg, "rng_seed", getattr(cfg, "seed", None))
    torch_generator = None
    if seed is not None:
        torch_generator = _seed_everything(int(seed), cfg.device, rank=0)
    setattr(cfg, "torch_generator", torch_generator)
    ckpting_save_folder = args.runs_path / run_name
    if not os.path.exists(ckpting_save_folder):
        os.makedirs(ckpting_save_folder)

    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
        device=cfg.device,
        dtype=DTYPES[cfg.dtype],
    )

    optimizer_cls, param_groups, optimizer_kwargs = _prepare_optimizer_setup(cfg, model)
    optimizer = optimizer_cls(param_groups, **optimizer_kwargs)

    np_arr_train_data = np.memmap(
        args.np_dat_train_path, dtype=np.int32, mode="r", shape=(args.total_train_tokens,)
    )

    np_arr_valid_data = np.memmap(
        args.np_dat_valid_path, dtype=np.int32, mode="r", shape=(args.total_val_tokens,)
    )

    # activation norm utils
    activation_norms = {}

    def get_activation_norm_hook(name):
        def hook(module, input, output):
            activation_norms[name] = output.norm().item()

        return hook

    for name, module in model.named_modules():
        if isinstance(module, Linear):
            module.register_forward_hook(get_activation_norm_hook(name))

    mask_token_id = getattr(cfg, "mask_token_id", cfg.vocab_size - 1)
    noise_epsilon = getattr(cfg, "noise_epsilon", 1e-3)
    random_trunc_prob = getattr(cfg, "random_trunc_prob", 0.01)

    setattr(cfg, "mask_token_id", mask_token_id)
    setattr(cfg, "noise_epsilon", noise_epsilon)
    setattr(cfg, "random_trunc_prob", random_trunc_prob)

    batch_getter = partial(
        get_batch,
        mask_token_id=mask_token_id,
        noise_epsilon=noise_epsilon,
        random_trunc_prob=random_trunc_prob,
    )

    def _compute_loss(logits: torch.Tensor, batch) -> torch.Tensor:
        return diffusion_cross_entropy(logits, batch.clean_targets, batch.mask, batch.p_mask)

    train_loop(
        model,
        optimizer,
        np_arr_train_data=np_arr_train_data,
        np_arr_valid_data=np_arr_valid_data,
        batch_size=cfg.batch_size,
        context_length=cfg.context_length,
        device=str(cfg.device),
        max_learning_rate=cfg.max_learning_rate,
        min_learning_rate=cfg.min_learning_rate,
        warmup_iters=cfg.warmup_iters,
        cosine_cycle_iters=cfg.cosine_cycle_iters,
        max_train_iteration=cfg.max_train_iteration,
        max_val_iteration=cfg.max_val_iteration,
        val_freq_iteration=cfg.val_freq_iteration,
        grad_clip_max_l2_norm=cfg.grad_clip_max_l2_norm,
        ckpting_save_iter=cfg.ckpting_save_iter,
        ckpting_save_folder=ckpting_save_folder,
        get_batch=batch_getter,
        lr_cosine_schedule=lr_cosine_schedule,
        gradient_clipping=gradient_clipping,
        save_checkpoint=save_checkpoint,
        compute_loss=_compute_loss,
        batch_generator=torch_generator,
        logger=logger,
        activation_norms=activation_norms,
        log_activation_norms=True,
        log_weight_norms=True,
    )


def train_transformer_ddp(rank, args, cfg_dc):
    cfg = args

    setup_process_group(cfg.backend, rank, cfg.world_size)

    seed = getattr(cfg, "rng_seed", getattr(cfg, "seed", None))
    torch_generator = None
    if seed is not None:
        torch_generator = _seed_everything(int(seed), cfg.device, rank=rank)
    setattr(cfg, "torch_generator", torch_generator)

    logger, run_name, ckpting_save_folder = init_logging(rank, cfg, cfg_dc)

    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
        device=cfg.device,
        dtype=DTYPES[cfg.dtype],
    )

    ddp_model = DDP(model, cfg.world_size, cfg.bucket_size_mb)

    optimizer_cls, param_groups, optimizer_kwargs = _prepare_optimizer_setup(cfg, model)
    optimizer = OptimizerStateSharding(
        param_groups,
        optimizer_cls,
        **optimizer_kwargs,
    )

    np_arr_train_data = np.memmap(
        args.np_dat_train_path, dtype=np.int32, mode="r", shape=(args.total_train_tokens,)
    )

    np_arr_valid_data = np.memmap(
        args.np_dat_valid_path, dtype=np.int32, mode="r", shape=(args.total_val_tokens,)
    )

    # activation norm utils
    activation_norms = {}

    def get_activation_norm_hook(name):
        def hook(module, input, output):
            activation_norms[name] = output.norm().item()

        return hook

    for name, module in model.named_modules():
        if isinstance(module, Linear):
            module.register_forward_hook(get_activation_norm_hook(name))

    # broadcast model params/buffers from rank 0 via DDP helper
    ddp_model.broadcast_parameters(src=0)

    mask_token_id = getattr(cfg, "mask_token_id", cfg.vocab_size - 1)
    noise_epsilon = getattr(cfg, "noise_epsilon", 1e-3)
    random_trunc_prob = getattr(cfg, "random_trunc_prob", 0.01)

    setattr(cfg, "mask_token_id", mask_token_id)
    setattr(cfg, "noise_epsilon", noise_epsilon)
    setattr(cfg, "random_trunc_prob", random_trunc_prob)

    batch_getter = partial(
        get_batch,
        mask_token_id=mask_token_id,
        noise_epsilon=noise_epsilon,
        random_trunc_prob=random_trunc_prob,
    )

    def _compute_loss(logits: torch.Tensor, batch) -> torch.Tensor:
        return diffusion_cross_entropy(logits, batch.clean_targets, batch.mask, batch.p_mask)

    def _shard_batch(batch_obj, ws: int, rk: int):
        if isinstance(batch_obj, DiffusionBatch):
            def _chunk(t: torch.Tensor) -> torch.Tensor:
                return torch.chunk(t, ws, dim=0)[rk]

            metadata = dict(batch_obj.metadata) if isinstance(batch_obj.metadata, dict) else {}
            return DiffusionBatch(
                noisy_inputs=_chunk(batch_obj.noisy_inputs),
                clean_targets=_chunk(batch_obj.clean_targets),
                mask=_chunk(batch_obj.mask),
                p_mask=_chunk(batch_obj.p_mask),
                metadata=metadata,
            )
        raise ValueError("DDP expects DiffusionBatch instances for sharding.")

    def _sync():
        ddp_model.finish_gradient_synchronization()

    train_loop(
        ddp_model,
        optimizer,
        np_arr_train_data=np_arr_train_data,
        np_arr_valid_data=np_arr_valid_data,
        batch_size=cfg.batch_size,
        context_length=cfg.context_length,
        device=str(cfg.device),
        max_learning_rate=cfg.max_learning_rate,
        min_learning_rate=cfg.min_learning_rate,
        warmup_iters=cfg.warmup_iters,
        cosine_cycle_iters=cfg.cosine_cycle_iters,
        max_train_iteration=cfg.max_train_iteration,
        max_val_iteration=cfg.max_val_iteration,
        val_freq_iteration=cfg.val_freq_iteration,
        grad_clip_max_l2_norm=cfg.grad_clip_max_l2_norm,
        ckpting_save_iter=cfg.ckpting_save_iter,
        ckpting_save_folder=ckpting_save_folder,
        get_batch=batch_getter,
        lr_cosine_schedule=lr_cosine_schedule,
        gradient_clipping=gradient_clipping,
        save_checkpoint=save_checkpoint,
        compute_loss=_compute_loss,
        batch_generator=torch_generator,
        logger=logger,
        activation_norms=activation_norms,
        log_activation_norms=True,
        log_weight_norms=True,
        shard_batch=_shard_batch,
        sync_gradients=_sync,
        reduce_metric=allreduce_mean,
        world_size=cfg.world_size,
        local_rank=rank,
        is_rank_zero=(rank == 0),
    )

    cleanup_process_group()


def build_run_config(cfg, cfg_dc):
    """Pure builder for run configuration payload.

    No rank checks or I/O.
    """
    run_config = {
        "vocab_size": cfg.vocab_size,
        "context_length": cfg.context_length,
        "d_model": cfg.d_model,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "d_ff": cfg.d_ff,
        "rope_theta": cfg.rope_theta,
        "mask_token_id": cfg.mask_token_id,
        "noise_epsilon": getattr(cfg, "noise_epsilon", None),
        "random_trunc_prob": getattr(cfg, "random_trunc_prob", None),
        "betas": cfg.betas,
        "eps": cfg.eps,
        "weight_decay": cfg.weight_decay,
        "initial_learning_rate": cfg.initial_learning_rate,
        "grad_clip_max_l2_norm": cfg.grad_clip_max_l2_norm,
        "max_learning_rate": cfg.max_learning_rate,
        "min_learning_rate": cfg.min_learning_rate,
        "warmup_iters": cfg.warmup_iters,
        "cosine_cycle_iters": cfg.cosine_cycle_iters,
        "max_train_iteration": cfg.max_train_iteration,
        "max_val_iteration": cfg.max_val_iteration,
        "val_freq_iteration": cfg.val_freq_iteration,
        "batch_size": cfg.batch_size,
        "device": cfg.device,
        "dtype": cfg.dtype,
        "ckpting_save_iter": cfg.ckpting_save_iter,
    }

    seed_value = getattr(cfg, "rng_seed", getattr(cfg, "seed", None))
    if seed_value is not None:
        run_config["rng_seed"] = seed_value

    # Optional metadata from config
    if getattr(cfg_dc, "wandb", None):
        if getattr(cfg_dc.wandb, "architecture", None):
            run_config["architecture"] = cfg_dc.wandb.architecture
        if getattr(cfg_dc.wandb, "dataset", None):
            run_config["dataset"] = cfg_dc.wandb.dataset

    if getattr(cfg_dc, "logging", None):
        if getattr(cfg_dc.logging, "architecture", None):
            run_config["architecture"] = cfg_dc.logging.architecture
        if getattr(cfg_dc.logging, "dataset", None):
            run_config["dataset"] = cfg_dc.logging.dataset

    return run_config


def init_logging(rank: int, cfg, cfg_dc):
    """Initialize logging on rank 0 and broadcast run_name.

    Returns (logger, run_name, ckpt_dir). Only rank 0 will create directories
    and own a real logger; other ranks get a RankZeroLogger(NoOp).
    """
    run_config = build_run_config(cfg, cfg_dc)

    real_logger: Logger | None = None
    run_name: str | None = None

    if rank == 0:
        # Select logger backend (prefer [logging], default to console)
        real_logger = ConsoleLogger()
        if getattr(cfg_dc, "logging", None) and cfg_dc.logging and cfg_dc.logging.backend:
            backend = (cfg_dc.logging.backend or "console").lower()
            if backend == "console":
                real_logger = ConsoleLogger()
            elif backend == "wandb":
                entity = getattr(cfg_dc.wandb, "entity", None) if getattr(cfg_dc, "wandb", None) else None
                project = getattr(cfg_dc.wandb, "project", None) if getattr(cfg_dc, "wandb", None) else None
                default_name = (getattr(cfg_dc.logging, "run_name", None) or datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
                real_logger = WandbLogger(entity=entity, project=project, name=default_name)

        info = real_logger.start_run(run_config)
        run_name = info.get("run_name") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ensure all ranks agree on run_name
    run_name = broadcast_string(run_name, src=0)

    # proxy so only rank 0 logs
    rz_logger: Logger = RankZeroLogger(rank, real_logger)

    # compute ckpt dir, create only on rank 0
    ckpt_dir = cfg.runs_path / run_name
    if rank == 0 and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    return rz_logger, run_name, ckpt_dir
