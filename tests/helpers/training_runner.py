from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import partial
import tempfile

import numpy as np
import random
import torch
import torch.distributed as dist

from diffusionlm.training.loop import train_loop
from diffusionlm.training.data import get_batch, DiffusionBatch
from diffusionlm.training.grad import gradient_clipping
from diffusionlm.training.loss import diffusion_cross_entropy
from diffusionlm.training.schedule import lr_cosine_schedule
from diffusionlm.training.optim import AdamW
from diffusionlm.training.checkpoint import save_checkpoint, load_checkpoint

from ddp import DDP, OptimizerStateSharding
from ddp.utils import setup_process_group, cleanup_process_group, allreduce_mean

from tests.fixtures import TrainingBundle


@dataclass(frozen=True)
class TrainingStepSnapshot:
    step: int
    loss: float
    parameter_tensors: List[Tuple[str, torch.Tensor]]
    gradient_tensors: List[Tuple[str, torch.Tensor]]
    gradient_norms: Dict[str, float]
    optimizer_state: Dict[str, Any]
    learning_rate: float


def _clone_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {"state": {}, "param_groups": []}
    for idx, state in state_dict.get("state", {}).items():
        cloned_state: Dict[str, Any] = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                cloned_state[key] = value.detach().cpu().clone()
            else:
                cloned_state[key] = value
        cloned["state"][idx] = cloned_state

    for group in state_dict.get("param_groups", []):
        cloned_group = dict(group)
        cloned_group["params"] = tuple(group.get("params", ()))
        cloned["param_groups"].append(cloned_group)

    return cloned


def _canonical_name(name: str) -> str:
    return name.split(".", 1)[-1] if name.startswith("model.") else name


def _set_all_seeds(seed: int, device: torch.device) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _shard_diffusion_batch(batch: DiffusionBatch, world_size: int, rank: int) -> DiffusionBatch:
    def _chunk(t: torch.Tensor) -> torch.Tensor:
        return torch.chunk(t, world_size, dim=0)[rank]

    metadata = dict(batch.metadata) if isinstance(batch.metadata, dict) else {}
    return DiffusionBatch(
        noisy_inputs=_chunk(batch.noisy_inputs),
        clean_targets=_chunk(batch.clean_targets),
        mask=_chunk(batch.mask),
        p_mask=_chunk(batch.p_mask),
        metadata=metadata,
    )


def _prepare_data_arrays(bundle: TrainingBundle) -> Tuple[np.ndarray, np.ndarray]:
    train_tokens_np = bundle.dataset.train_tokens.detach().cpu().numpy()
    valid_tokens_np = bundle.dataset.valid_tokens.detach().cpu().numpy()
    return train_tokens_np, valid_tokens_np


def _make_step_callback(
    snapshots: List[TrainingStepSnapshot],
    hook: Optional[Callable[[int, torch.nn.Module, torch.optim.Optimizer], None]] = None,
) -> Callable[[int, torch.nn.Module, torch.optim.Optimizer, float, float], None]:
    def _callback(
        iteration: int,
        model_ref: torch.nn.Module,
        optimizer_ref: torch.optim.Optimizer,
        loss_value: float,
        lr_value: float,
    ) -> None:
        step_index = iteration

        param_tensors = [(_canonical_name(name), param.detach().cpu().clone()) for name, param in model_ref.named_parameters()]

        grad_tensors: List[Tuple[str, torch.Tensor]] = []
        grad_norms: Dict[str, float] = {}
        for name, param in model_ref.named_parameters():
            if param.grad is None:
                continue
            canonical = _canonical_name(name)
            grad_cpu = param.grad.detach().cpu().clone()
            grad_tensors.append((canonical, grad_cpu))
            grad_norms[canonical] = float(torch.linalg.vector_norm(param.grad.detach()).item())

        snapshots.append(TrainingStepSnapshot(
            step=step_index,
            loss=float(loss_value),
            parameter_tensors=param_tensors,
            gradient_tensors=grad_tensors,
            gradient_norms=grad_norms,
            optimizer_state=_clone_state_dict(optimizer_ref.state_dict()),
            learning_rate=float(lr_value),
        ))

        if hook is not None:
            hook(step_index, model_ref, optimizer_ref)

    return _callback


def _run_loop(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    bundle: TrainingBundle,
    generator: torch.Generator,
    train_tokens_np: np.ndarray,
    valid_tokens_np: np.ndarray,
    num_steps: int,
    base_iteration: int,
    snapshots: List[TrainingStepSnapshot],
    hook: Optional[Callable[[int, torch.nn.Module, torch.optim.Optimizer], None]] = None,
    shard_batch: Optional[Callable[[object, int, int], object]] = None,
    sync_gradients: Optional[Callable[[], None]] = None,
    reduce_metric: Optional[Callable[[float], float]] = None,
    world_size: int = 1,
    local_rank: int = 0,
    is_rank_zero: bool = True,
) -> None:
    if num_steps <= 0:
        return

    model_cfg = bundle.train_config.model
    optimizer_cfg = bundle.train_config.optimizer
    training_cfg = bundle.train_config.training

    step_callback = _make_step_callback(snapshots, hook)

    mask_token_id = getattr(model_cfg, "mask_token_id", model_cfg.vocab_size - 1)
    noise_epsilon = getattr(training_cfg, "noise_epsilon", 1e-3)
    random_trunc_prob = getattr(training_cfg, "random_trunc_prob", 0.01)

    batch_getter = partial(
        get_batch,
        mask_token_id=mask_token_id,
        noise_epsilon=noise_epsilon,
        random_trunc_prob=random_trunc_prob,
    )

    def _compute_loss(logits: torch.Tensor, batch) -> torch.Tensor:
        if isinstance(batch, DiffusionBatch):
            return diffusion_cross_entropy(logits, batch.clean_targets, batch.mask, batch.p_mask)
        raise ValueError("Expected DiffusionBatch in training runner.")

    train_loop(
        module,
        optimizer,
        np_arr_train_data=train_tokens_np,
        np_arr_valid_data=valid_tokens_np,
        batch_size=training_cfg.batch_size,
        context_length=model_cfg.context_length,
        device=str(model_cfg.device),
        max_learning_rate=optimizer_cfg.max_learning_rate,
        min_learning_rate=optimizer_cfg.min_learning_rate,
        warmup_iters=optimizer_cfg.warmup_iters,
        cosine_cycle_iters=optimizer_cfg.cosine_cycle_iters,
        max_train_iteration=base_iteration + num_steps - 1,
        max_val_iteration=training_cfg.max_val_iteration,
        val_freq_iteration=training_cfg.val_freq_iteration,
        grad_clip_max_l2_norm=optimizer_cfg.grad_clip_max_l2_norm,
        ckpting_save_iter=training_cfg.ckpting_save_iter,
        ckpting_save_folder=None,
        get_batch=batch_getter,
        lr_cosine_schedule=lr_cosine_schedule,
        gradient_clipping=gradient_clipping,
        save_checkpoint=lambda *_, **__: None,
        compute_loss=_compute_loss,
        batch_generator=generator,
        logger=None,
        activation_norms=None,
        log_activation_norms=False,
        log_weight_norms=False,
        shard_batch=shard_batch,
        sync_gradients=sync_gradients,
        reduce_metric=reduce_metric,
        world_size=world_size,
        local_rank=local_rank,
        is_rank_zero=is_rank_zero,
        step_callback=step_callback,
        start_iteration=base_iteration,
    )


def run_training_steps(bundle: TrainingBundle, *, num_steps: int, seed: int | None = None) -> List[TrainingStepSnapshot]:
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0")

    model_cfg = bundle.train_config.model
    training_cfg = bundle.train_config.training

    device = torch.device(model_cfg.device)
    chosen_seed = seed if seed is not None else (training_cfg.seed or 0)
    generator = _set_all_seeds(chosen_seed, device)

    model = bundle.model_factory()
    optimizer = bundle.optimizer_factory(model.parameters())

    train_tokens_np, valid_tokens_np = _prepare_data_arrays(bundle)
    snapshots: List[TrainingStepSnapshot] = []

    _run_loop(
        module=model,
        optimizer=optimizer,
        bundle=bundle,
        generator=generator,
        train_tokens_np=train_tokens_np,
        valid_tokens_np=valid_tokens_np,
        num_steps=num_steps,
        base_iteration=0,
        snapshots=snapshots,
    )

    return snapshots


def run_training_steps_ddp(bundle: TrainingBundle, *, num_steps: int, seed: int | None = None) -> List[TrainingStepSnapshot]:
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    model_cfg = bundle.train_config.model
    training_cfg = bundle.train_config.training
    optimizer_cfg = bundle.train_config.optimizer

    device = torch.device(model_cfg.device)
    chosen_seed = seed if seed is not None else (training_cfg.seed or 0)
    generator = _set_all_seeds(chosen_seed, device)

    need_cleanup = False
    if not dist.is_initialized():
        setup_process_group("gloo", rank=0, world_size=1)
        need_cleanup = True
    elif dist.get_world_size() != 1:
        raise RuntimeError("Existing process group world_size is not 1")

    try:
        model = bundle.model_factory()
        ddp_model = DDP(model, world_size=1, bucket_size_mb=0)
        ddp_model.broadcast_parameters(src=0)

        optimizer = OptimizerStateSharding(
            model.parameters(),
            AdamW,
            lr=optimizer_cfg.initial_learning_rate,
            betas=optimizer_cfg.betas,
            eps=float(optimizer_cfg.eps),
            weight_decay=optimizer_cfg.weight_decay,
        )

        train_tokens_np, valid_tokens_np = _prepare_data_arrays(bundle)
        snapshots: List[TrainingStepSnapshot] = []

        def _shard(batch_obj: object, ws: int, rk: int):
            if isinstance(batch_obj, DiffusionBatch):
                return _shard_diffusion_batch(batch_obj, ws, rk)
            raise ValueError("Expected DiffusionBatch for sharding in tests.")

        def _sync():
            ddp_model.finish_gradient_synchronization()

        _run_loop(
            module=ddp_model,
            optimizer=optimizer,
            bundle=bundle,
            generator=generator,
            train_tokens_np=train_tokens_np,
            valid_tokens_np=valid_tokens_np,
            num_steps=num_steps,
            base_iteration=0,
            snapshots=snapshots,
            shard_batch=_shard,
            sync_gradients=_sync,
            reduce_metric=allreduce_mean,
            world_size=1,
            local_rank=0,
            is_rank_zero=True,
        )

        return snapshots
    finally:
        if need_cleanup and dist.is_initialized():
            cleanup_process_group()


def run_training_with_checkpoint(
    bundle: TrainingBundle,
    *,
    total_steps: int,
    checkpoint_step: int,
    seed: int | None = None,
) -> Tuple[List[TrainingStepSnapshot], List[TrainingStepSnapshot]]:
    if not (0 < checkpoint_step < total_steps):
        raise ValueError("checkpoint_step must be between 0 and total_steps")

    baseline = run_training_steps(bundle, num_steps=total_steps, seed=seed)

    model_cfg = bundle.train_config.model
    training_cfg = bundle.train_config.training

    device = torch.device(model_cfg.device)
    chosen_seed = seed if seed is not None else (training_cfg.seed or 0)

    generator = _set_all_seeds(chosen_seed, device)
    model = bundle.model_factory()
    optimizer = bundle.optimizer_factory(model.parameters())
    train_tokens_np, valid_tokens_np = _prepare_data_arrays(bundle)

    snapshots_resumed: List[TrainingStepSnapshot] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "resume.ckpt"
        state_holder: Dict[str, Any] = {}

        def _hook(step_idx: int, model_ref: torch.nn.Module, optimizer_ref: torch.optim.Optimizer) -> None:
            if step_idx == checkpoint_step - 1 and "generator_state" not in state_holder:
                save_checkpoint(model_ref, optimizer_ref, step_idx, ckpt_path)
                state_holder["generator_state"] = generator.get_state()
                state_holder["python_state"] = random.getstate()
                state_holder["numpy_state"] = np.random.get_state()
                state_holder["torch_state"] = torch.random.get_rng_state()
                state_holder["iteration"] = step_idx

        _run_loop(
            module=model,
            optimizer=optimizer,
            bundle=bundle,
            generator=generator,
            train_tokens_np=train_tokens_np,
            valid_tokens_np=valid_tokens_np,
            num_steps=checkpoint_step,
            base_iteration=0,
            snapshots=snapshots_resumed,
            hook=_hook,
        )

        if "generator_state" not in state_holder:
            raise RuntimeError("Checkpoint hook did not trigger")

        remaining = total_steps - checkpoint_step
        if remaining > 0:
            resume_base_iteration = int(state_holder["iteration"]) + 1
            generator_resume = torch.Generator(device="cpu")
            generator_resume.set_state(state_holder["generator_state"])

            random.setstate(state_holder["python_state"])
            np.random.set_state(state_holder["numpy_state"])
            torch.random.set_rng_state(state_holder["torch_state"])

            model_resume = bundle.model_factory()
            optimizer_resume = bundle.optimizer_factory(model_resume.parameters())
            load_checkpoint(ckpt_path, model_resume, optimizer_resume)

            random.setstate(state_holder["python_state"])
            np.random.set_state(state_holder["numpy_state"])
            torch.random.set_rng_state(state_holder["torch_state"])

            _run_loop(
                module=model_resume,
                optimizer=optimizer_resume,
                bundle=bundle,
                generator=generator_resume,
                train_tokens_np=train_tokens_np,
                valid_tokens_np=valid_tokens_np,
                num_steps=remaining,
                base_iteration=resume_base_iteration,
                snapshots=snapshots_resumed,
            )

    return baseline, snapshots_resumed


def run_training_with_checkpoint_ddp(
    bundle: TrainingBundle,
    *,
    total_steps: int,
    checkpoint_step: int,
    seed: int | None = None,
) -> Tuple[List[TrainingStepSnapshot], List[TrainingStepSnapshot]]:
    if not (0 < checkpoint_step < total_steps):
        raise ValueError("checkpoint_step must be between 0 and total_steps")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    baseline = run_training_steps_ddp(bundle, num_steps=total_steps, seed=seed)

    model_cfg = bundle.train_config.model
    training_cfg = bundle.train_config.training
    optimizer_cfg = bundle.train_config.optimizer

    device = torch.device(model_cfg.device)
    chosen_seed = seed if seed is not None else (training_cfg.seed or 0)

    need_cleanup = False
    if not dist.is_initialized():
        setup_process_group("gloo", rank=0, world_size=1)
        need_cleanup = True
    elif dist.get_world_size() != 1:
        raise RuntimeError("Existing process group world_size is not 1")

    try:
        generator = _set_all_seeds(chosen_seed, device)
        model = bundle.model_factory()
        ddp_model = DDP(model, world_size=1, bucket_size_mb=0)
        ddp_model.broadcast_parameters(src=0)

        optimizer = OptimizerStateSharding(
            model.parameters(),
            AdamW,
            lr=optimizer_cfg.initial_learning_rate,
            betas=optimizer_cfg.betas,
            eps=float(optimizer_cfg.eps),
            weight_decay=optimizer_cfg.weight_decay,
        )

        train_tokens_np, valid_tokens_np = _prepare_data_arrays(bundle)
        snapshots_resumed: List[TrainingStepSnapshot] = []

        def _shard(batch_obj: object, ws: int, rk: int):
            if isinstance(batch_obj, DiffusionBatch):
                return _shard_diffusion_batch(batch_obj, ws, rk)
            raise ValueError("Expected DiffusionBatch for sharding in tests.")

        def _sync():
            ddp_model.finish_gradient_synchronization()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "resume.ckpt"
            state_holder: Dict[str, Any] = {}

            def _hook(step_idx: int, model_ref: torch.nn.Module, optimizer_ref: torch.optim.Optimizer) -> None:
                if step_idx == checkpoint_step - 1 and "generator_state" not in state_holder:
                    save_checkpoint(model_ref, optimizer_ref, step_idx, ckpt_path)
                    state_holder["generator_state"] = generator.get_state()
                    state_holder["python_state"] = random.getstate()
                    state_holder["numpy_state"] = np.random.get_state()
                    state_holder["torch_state"] = torch.random.get_rng_state()
                    state_holder["iteration"] = step_idx

            _run_loop(
                module=ddp_model,
                optimizer=optimizer,
                bundle=bundle,
                generator=generator,
                train_tokens_np=train_tokens_np,
                valid_tokens_np=valid_tokens_np,
                num_steps=checkpoint_step,
                base_iteration=0,
                snapshots=snapshots_resumed,
                hook=_hook,
                shard_batch=_shard,
                sync_gradients=_sync,
                reduce_metric=allreduce_mean,
                world_size=1,
                local_rank=0,
                is_rank_zero=True,
            )

            if "generator_state" not in state_holder:
                raise RuntimeError("Checkpoint hook did not trigger")

            remaining = total_steps - checkpoint_step
            if remaining > 0:
                resume_base_iteration = int(state_holder["iteration"]) + 1
                generator_resume = torch.Generator(device="cpu")
                generator_resume.set_state(state_holder["generator_state"])

                random.setstate(state_holder["python_state"])
                np.random.set_state(state_holder["numpy_state"])
                torch.random.set_rng_state(state_holder["torch_state"])

                model_resume = bundle.model_factory()
                ddp_model_resume = DDP(model_resume, world_size=1, bucket_size_mb=0)
                ddp_model_resume.broadcast_parameters(src=0)

                optimizer_resume = OptimizerStateSharding(
                    model_resume.parameters(),
                    AdamW,
                    lr=optimizer_cfg.initial_learning_rate,
                    betas=optimizer_cfg.betas,
                    eps=float(optimizer_cfg.eps),
                    weight_decay=optimizer_cfg.weight_decay,
                )
                load_checkpoint(ckpt_path, ddp_model_resume, optimizer_resume)

                random.setstate(state_holder["python_state"])
                np.random.set_state(state_holder["numpy_state"])
                torch.random.set_rng_state(state_holder["torch_state"])

                def _shard_resume(batch_obj: object, ws: int, rk: int):
                    if isinstance(batch_obj, DiffusionBatch):
                        return _shard_diffusion_batch(batch_obj, ws, rk)
                    raise ValueError("Expected DiffusionBatch for sharding in resume tests.")

                def _sync_resume():
                    ddp_model_resume.finish_gradient_synchronization()

                _run_loop(
                    module=ddp_model_resume,
                    optimizer=optimizer_resume,
                    bundle=bundle,
                    generator=generator_resume,
                    train_tokens_np=train_tokens_np,
                    valid_tokens_np=valid_tokens_np,
                    num_steps=remaining,
                    base_iteration=resume_base_iteration,
                    snapshots=snapshots_resumed,
                    shard_batch=_shard_resume,
                    sync_gradients=_sync_resume,
                    reduce_metric=allreduce_mean,
                    world_size=1,
                    local_rank=0,
                    is_rank_zero=True,
                )

        return baseline, snapshots_resumed
    finally:
        if need_cleanup and dist.is_initialized():
            cleanup_process_group()


def run_single_step(bundle: TrainingBundle, *, seed: int | None = None) -> TrainingStepSnapshot:
    return run_training_steps(bundle, num_steps=1, seed=seed)[0]
