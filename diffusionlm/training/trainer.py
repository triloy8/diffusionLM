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
from diffusionlm.training.streaming import HFTokenIteratorFactory, StreamingBatcher

from datetime import datetime
import numpy as np
import os
import random
import torch
# import torch.distributed as dist
from functools import partial
# from datasets import load_dataset

from diffusionlm.utils.dtypes import DTYPES
from logger import Logger
from logger import ConsoleLogger, WandbLogger, RankZeroLogger
from ddp import DDP, OptimizerStateSharding
from ddp.utils import broadcast_string, setup_process_group, cleanup_process_group, allreduce_mean
from diffusionlm.tokenizer.tokenizer import Tokenizer


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
    muon_cfg = getattr(cfg, "muon_cfg", None)
    param_groups = build_optimizer_param_groups(model, optimizer_name, muon_cfg)
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
    for group in param_groups:
        group.setdefault("initial_lr", float(cfg.initial_learning_rate))
        group.setdefault("max_lr", float(cfg.max_learning_rate))
        group.setdefault("min_lr", float(cfg.min_learning_rate))
        group.setdefault("warmup_iters", int(cfg.warmup_iters))
        group.setdefault("cosine_cycle_iters", int(cfg.cosine_cycle_iters))
        group.setdefault("lr", float(group.get("initial_lr")))
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

    tokenizer = Tokenizer.from_files(
        str(args.tokenizer_vocab_path),
        str(args.tokenizer_merges_path),
        str(args.tokenizer_special_tokens_path)
    )
    shuffle_seed = getattr(args, "shuffle_seed", None)
    if shuffle_seed is None:
        shuffle_seed = getattr(args, "rng_seed", getattr(cfg, "seed", None))
    eot_token_id = getattr(cfg, "eot_token_id", None)
    if eot_token_id is None:
        raise ValueError("eot_token_id must be set for streaming datasets")
    pad_token_id = getattr(cfg, "pad_token_id", eot_token_id)
    setattr(cfg, "pad_token_id", pad_token_id)
    val_log_every = int(getattr(cfg, "val_log_every", 0))
    val_log_samples = int(getattr(cfg, "val_log_samples", 0))
    train_iterator_factory = HFTokenIteratorFactory(
        dataset_name=str(args.dataset_name),
        dataset_config=(str(args.dataset_config) if args.dataset_config is not None else None),
        split=str(args.train_split),
        text_field=str(args.text_field),
        tokenizer=tokenizer,
        context_length=int(cfg.context_length),
        eot_token_id=int(eot_token_id),
        pad_token_id=int(pad_token_id),
        shuffle_buffer_size=int(getattr(args, "shuffle_buffer_size", 0)),
        shuffle_seed=(int(shuffle_seed) if shuffle_seed is not None else None),
        world_size=1,
        rank=0,
        logger=logger,
        hf_debug_logging=True,
    )
    val_iterator_factory = HFTokenIteratorFactory(
        dataset_name=str(args.dataset_name),
        dataset_config=(str(args.dataset_config) if args.dataset_config is not None else None),
        split=str(args.val_split),
        text_field=str(args.text_field),
        tokenizer=tokenizer,
        context_length=int(cfg.context_length),
        eot_token_id=int(eot_token_id),
        pad_token_id=int(pad_token_id),
        shuffle_buffer_size=0,
        shuffle_seed=None,
        world_size=1,
        rank=0,
        logger=logger,
        hf_debug_logging=True,
    )
    train_batcher = StreamingBatcher(train_iterator_factory, device=str(cfg.device), logger=logger)
    val_batcher = StreamingBatcher(val_iterator_factory, device=str(cfg.device), logger=logger)

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
        train_data=train_batcher,
        val_data=val_batcher,
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
        grad_accum_steps=int(getattr(cfg, "grad_accum_steps", 1)),
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
        log_activation_norms=bool(getattr(cfg, "log_activation_norms", False)),
        log_weight_norms=bool(getattr(cfg, "log_weight_norms", False)),
        val_log_every=val_log_every,
        val_log_samples=val_log_samples,
        val_sample_decode=tokenizer.decode,
        skip_validation=bool(getattr(cfg, "skip_validation", False)),
    )


def train_transformer_ddp(local_rank, args, cfg_dc):
    cfg = args

    num_nodes = int(getattr(cfg, "num_nodes", 1))
    num_gpus_per_node = int(getattr(cfg, "num_gpus_per_node", 1))
    local_rank_int = int(local_rank)
    node_rank = int(getattr(cfg, "node_rank", 0))
    master_addr = getattr(cfg, "master_addr", "localhost")
    master_port = getattr(cfg, "master_port", "29500")

    if getattr(cfg, "nccl_p2p_disable", None) is not None:
        os.environ["NCCL_P2P_DISABLE"] = "1" if cfg.nccl_p2p_disable else "0"

    world_size = max(1, num_nodes * num_gpus_per_node)
    global_rank = node_rank * num_gpus_per_node + local_rank_int
    setattr(cfg, "world_size", world_size)
    setattr(cfg, "global_rank", global_rank)

    setup_process_group(
        backend=cfg.backend,
        local_rank=local_rank_int,
        num_gpus_per_node=num_gpus_per_node,
        num_nodes=num_nodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
    )

    seed = getattr(cfg, "rng_seed", getattr(cfg, "seed", None))
    torch_generator = None
    if seed is not None:
        torch_generator = _seed_everything(int(seed), cfg.device, rank=global_rank)
    setattr(cfg, "torch_generator", torch_generator)

    logger, run_name, ckpting_save_folder = init_logging(global_rank, cfg, cfg_dc)

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

    tokenizer = Tokenizer.from_files(
        str(args.tokenizer_vocab_path),
        str(args.tokenizer_merges_path),
        str(args.tokenizer_special_tokens_path),
    )
    shuffle_seed = getattr(args, "shuffle_seed", None)
    if shuffle_seed is None:
        shuffle_seed = getattr(args, "rng_seed", getattr(cfg, "seed", None))
    eot_token_id = getattr(cfg, "eot_token_id", None)
    if eot_token_id is None:
        raise ValueError("eot_token_id must be set for streaming datasets")
    pad_token_id = getattr(cfg, "pad_token_id", eot_token_id)
    setattr(cfg, "pad_token_id", pad_token_id)
    val_log_every = int(getattr(cfg, "val_log_every", 0))
    val_log_samples = int(getattr(cfg, "val_log_samples", 0))
    per_rank_seed = (int(shuffle_seed) if shuffle_seed is not None else 0) + global_rank
    train_iterator_factory = HFTokenIteratorFactory(
        dataset_name=str(args.dataset_name),
        dataset_config=(str(args.dataset_config) if args.dataset_config is not None else None),
        split=str(args.train_split),
        text_field=str(args.text_field),
        tokenizer=tokenizer,
        context_length=int(cfg.context_length),
        eot_token_id=int(eot_token_id),
        pad_token_id=int(pad_token_id),
        shuffle_buffer_size=int(getattr(args, "shuffle_buffer_size", 0)),
        shuffle_seed=per_rank_seed,
        world_size=cfg.world_size,
        rank=global_rank,
        logger=logger,
        hf_debug_logging=True,
    )
    val_iterator_factory = HFTokenIteratorFactory(
        dataset_name=str(args.dataset_name),
        dataset_config=(str(args.dataset_config) if args.dataset_config is not None else None),
        split=str(args.val_split),
        text_field=str(args.text_field),
        tokenizer=tokenizer,
        context_length=int(cfg.context_length),
        eot_token_id=int(eot_token_id),
        pad_token_id=int(pad_token_id),
        shuffle_buffer_size=0,
        shuffle_seed=None,
        world_size=cfg.world_size,
        rank=global_rank,
        logger=logger,
        hf_debug_logging=True,
    )
    train_batcher = StreamingBatcher(train_iterator_factory, device=str(cfg.device), logger=logger)
    val_batcher = StreamingBatcher(val_iterator_factory, device=str(cfg.device), logger=logger)

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

    def _sync():
        ddp_model.finish_gradient_synchronization()

    train_loop(
        ddp_model,
        optimizer,
        train_data=train_batcher,
        val_data=val_batcher,
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
        grad_accum_steps=int(getattr(cfg, "grad_accum_steps", 1)),
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
        log_activation_norms=bool(getattr(cfg, "log_activation_norms", False)),
        log_weight_norms=bool(getattr(cfg, "log_weight_norms", False)),
        val_log_every=val_log_every,
        val_log_samples=val_log_samples,
        val_sample_decode=tokenizer.decode,
        sync_gradients=_sync,
        reduce_metric=allreduce_mean,
        is_rank_zero=(global_rank == 0),
        skip_validation=bool(getattr(cfg, "skip_validation", False)),
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
        "grad_accum_steps": int(getattr(cfg, "grad_accum_steps", 1)),
        "val_log_every": int(getattr(cfg, "val_log_every", 0)),
        "val_log_samples": int(getattr(cfg, "val_log_samples", 0)),
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
