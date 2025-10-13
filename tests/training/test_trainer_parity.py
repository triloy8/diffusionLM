from __future__ import annotations

from typing import List

import pytest
import torch
import torch.distributed as dist

from tests.helpers import (
    TrainingStepSnapshot,
    run_training_steps,
    run_training_steps_ddp,
    run_training_with_checkpoint,
    run_training_with_checkpoint_ddp,
)


def summarize_mismatch(metric_name: str, step_idx: int, diff_tensor: torch.Tensor) -> str:
    diff_flat = diff_tensor.detach().abs().flatten()
    if diff_flat.numel() == 0:
        return f"{metric_name} mismatch at step {step_idx}: empty diff tensor"
    max_diff = float(diff_flat.max().item())
    mean_diff = float(diff_flat.mean().item())
    max_index = int(diff_flat.argmax().item())
    return (
        f"{metric_name} mismatch at step {step_idx}: "
        f"max_diff={max_diff:.3e}, mean_diff={mean_diff:.3e}, max_index={max_index}"
    )


def _assert_close_tensor(metric_name: str, step_idx: int, actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> None:
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        diff = (actual - expected).detach().cpu()
        raise AssertionError(summarize_mismatch(metric_name, step_idx, diff)) from exc


def _assert_optimizer_states_equal(step_idx: int, state_a: dict, state_b: dict, *, atol: float, rtol: float) -> None:
    assert state_a.keys() == state_b.keys(), f"Optimizer state keys differ at step {step_idx}"
    assert state_a["state"].keys() == state_b["state"].keys(), f"Optimizer param state keys differ at step {step_idx}"

    for key in state_a["state"]:
        entry_a = state_a["state"][key]
        entry_b = state_b["state"][key]
        assert entry_a.keys() == entry_b.keys(), f"Optimizer entry keys differ for param {key} at step {step_idx}"
        for sub_key in entry_a:
            val_a = entry_a[sub_key]
            val_b = entry_b[sub_key]
            if torch.is_tensor(val_a) and torch.is_tensor(val_b):
                _assert_close_tensor(f"optimizer.state[{key}][{sub_key}]", step_idx, val_a, val_b, atol=atol, rtol=rtol)
            else:
                assert val_a == val_b, f"Optimizer state mismatch for param {key} field {sub_key} at step {step_idx}"

    assert state_a["param_groups"] == state_b["param_groups"], f"Optimizer param_groups mismatch at step {step_idx}"


def _assert_snapshots_equal(
    expected: List[TrainingStepSnapshot],
    actual: List[TrainingStepSnapshot],
    *,
    param_atol: float = 1e-7,
    grad_atol: float = 1e-7,
    opt_atol: float = 1e-7,
    opt_rtol: float = 1e-6,
) -> None:
    assert len(expected) == len(actual)
    for snap_a, snap_b in zip(expected, actual):
        assert snap_a.step == snap_b.step
        _assert_close_tensor("loss", snap_a.step, torch.tensor(snap_a.loss), torch.tensor(snap_b.loss), atol=1e-8, rtol=1e-6)

        for (name_a, tensor_a), (name_b, tensor_b) in zip(snap_a.parameter_tensors, snap_b.parameter_tensors):
            assert name_a == name_b
            _assert_close_tensor(f"param[{name_a}]", snap_a.step, tensor_a, tensor_b, atol=param_atol, rtol=1e-6)

        for (name_a, tensor_a), (name_b, tensor_b) in zip(snap_a.gradient_tensors, snap_b.gradient_tensors):
            assert name_a == name_b
            _assert_close_tensor(f"grad[{name_a}]", snap_a.step, tensor_a, tensor_b, atol=grad_atol, rtol=1e-6)

        assert snap_a.gradient_norms.keys() == snap_b.gradient_norms.keys()
        for name in snap_a.gradient_norms:
            assert pytest.approx(snap_a.gradient_norms[name], rel=1e-6, abs=1e-8) == snap_b.gradient_norms[name]

        _assert_close_tensor("learning_rate", snap_a.step, torch.tensor(snap_a.learning_rate), torch.tensor(snap_b.learning_rate), atol=1e-12, rtol=0.0)

        _assert_optimizer_states_equal(snap_a.step, snap_a.optimizer_state, snap_b.optimizer_state, atol=opt_atol, rtol=opt_rtol)


@pytest.mark.cpu
def test_single_trainer_matches_reference(toy_training_bundle):
    steps_run_1 = run_training_steps(toy_training_bundle, num_steps=3)
    steps_run_2 = run_training_steps(toy_training_bundle, num_steps=3)

    _assert_snapshots_equal(steps_run_1, steps_run_2)


@pytest.mark.cpu
def test_single_vs_ddp_match_on_tiny_run(toy_training_bundle):
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    if dist.is_initialized() and dist.get_world_size() != 1:
        pytest.skip("Process group already initialized with world_size > 1")

    single_steps = run_training_steps(toy_training_bundle, num_steps=3)
    ddp_steps = run_training_steps_ddp(toy_training_bundle, num_steps=3)

    _assert_snapshots_equal(
        single_steps,
        ddp_steps,
        param_atol=1e-4,
        grad_atol=2e-5,
        opt_atol=2e-5,
    )


@pytest.mark.cpu
def test_checkpoint_resume_matches_baseline(toy_training_bundle):
    total_steps = 4
    checkpoint_step = 2

    baseline_single, resumed_single = run_training_with_checkpoint(
        toy_training_bundle,
        total_steps=total_steps,
        checkpoint_step=checkpoint_step,
    )
    _assert_snapshots_equal(baseline_single, resumed_single)

    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    if dist.is_initialized() and dist.get_world_size() != 1:
        pytest.skip("Process group already initialized with world_size > 1")

    baseline_ddp, resumed_ddp = run_training_with_checkpoint_ddp(
        toy_training_bundle,
        total_steps=total_steps,
        checkpoint_step=checkpoint_step,
    )
    _assert_snapshots_equal(
        baseline_ddp,
        resumed_ddp,
        param_atol=1e-4,
        grad_atol=2e-5,
        opt_atol=2e-5,
    )
