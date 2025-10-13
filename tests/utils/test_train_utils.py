import copy
import numpy as np
import torch
import pytest

from diffusionlm.models import TransformerLM
from diffusionlm.training import (
    cross_entropy,
    AdamW,
    lr_cosine_schedule,
    gradient_clipping,
    get_batch,
    save_checkpoint,
    load_checkpoint,
)


def test_cross_entropy_matches_torch(device):
    B, T, V = 2, 3, 5
    logits = torch.randn(B, T, V, device=device)
    targets = torch.randint(0, V, (B, T), device=device)

    ours = cross_entropy(logits, targets).mean()
    torch_ce = torch.nn.functional.cross_entropy(logits.view(-1, V), targets.view(-1), reduction="mean")
    assert torch.allclose(ours, torch_ce, atol=1e-6, rtol=1e-5)


def test_lr_cosine_schedule_boundaries():
    max_lr = 1.0
    min_lr = 0.1
    warmup = 10
    cycle = 110
    assert lr_cosine_schedule(0, max_lr, min_lr, warmup, cycle) == pytest.approx(0.0)
    assert lr_cosine_schedule(warmup, max_lr, min_lr, warmup, cycle) == pytest.approx(max_lr)
    assert lr_cosine_schedule(cycle, max_lr, min_lr, warmup, cycle) == pytest.approx(min_lr)
    assert lr_cosine_schedule(cycle + 50, max_lr, min_lr, warmup, cycle) == pytest.approx(min_lr)


def test_lr_cosine_schedule_warmup_zero():
    max_lr = 1.0
    min_lr = 0.1
    warmup = 0
    cycle = 10
    # At it=0 with warmup=0, we expect max_lr
    assert lr_cosine_schedule(0, max_lr, min_lr, warmup, cycle) == pytest.approx(max_lr)
    # Mid-cycle returns between min and max
    mid = lr_cosine_schedule(5, max_lr, min_lr, warmup, cycle)
    assert min_lr < mid < max_lr


def test_lr_cosine_schedule_cycle_zero_returns_min_beyond():
    max_lr = 1.0
    min_lr = 0.2
    warmup = 0
    cycle = 0
    # For iterations beyond cycle, lr should be min_lr
    assert lr_cosine_schedule(1, max_lr, min_lr, warmup, cycle) == pytest.approx(min_lr)


def test_gradient_clipping_caps_total_norm(device):
    p1 = torch.nn.Parameter(torch.ones(3, device=device), requires_grad=True)
    p2 = torch.nn.Parameter(torch.ones(4, device=device) * 2, requires_grad=True)
    # set gradients
    p1.grad = torch.ones_like(p1) * 3
    p2.grad = torch.ones_like(p2) * 4
    params = [p1, p2]

    # Compute current norm
    grads = [p.grad for p in params]
    current = torch.norm(torch.stack([g.detach().norm(2) for g in grads]))
    max_norm = current / 2
    gradient_clipping(params, float(max_norm))
    new = torch.norm(torch.stack([g.detach().norm(2) for g in grads]))
    assert new <= max_norm + 1e-6


def test_get_batch_shapes_and_targets(tmp_path, device):
    import random
    random.seed(0)
    arr = np.arange(50, dtype=np.int32)
    B, T = 2, 4
    x, y = get_batch(arr, batch_size=B, context_length=T, device=str(device))
    assert x.shape == (B, T)
    assert y.shape == (B, T)
    # For monotonic arr, targets equal inputs + 1
    assert torch.all(y == x + 1)


def test_get_batch_raises_on_too_small_dataset(device):
    import random
    random.seed(0)
    arr = np.arange(3, dtype=np.int32)  # too small for T=4
    with pytest.raises(ValueError):
        _ = get_batch(arr, batch_size=1, context_length=4, device=str(device))


def test_checkpointing_roundtrip(tmp_path, device):
    model = TransformerLM(
        vocab_size=8,
        context_length=4,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        rope_theta=10000.0,
        device=device,
        dtype=torch.float32,
    )
    opt = AdamW(model.parameters(), lr=1e-3)

    # Save
    ckpt_file = tmp_path / "ckpt.pt"
    save_checkpoint(model, opt, iteration=123, out=str(ckpt_file))

    # Mutate model
    orig_state = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    # Load
    it = load_checkpoint(str(ckpt_file), model, opt)
    assert it == 123

    # Compare states
    new_state = model.state_dict()
    for k in orig_state:
        assert torch.allclose(orig_state[k], new_state[k])


@pytest.mark.parametrize(
    "lr,betas,wd",
    [
        (1e-3, (0.9, 0.999), 0.01),
        (5e-4, (0.8, 0.99), 0.0),
        (1e-2, (0.95, 0.98), 0.1),
    ],
)
def test_adamw_single_step_matches_torch_paramized(device, lr, betas, wd):
    # Simple 1D parameter tensor with fixed gradient
    p_init = torch.tensor([1.0, -2.0, 3.0, -4.0], device=device)
    grad = torch.tensor([0.1, -0.2, 0.3, -0.4], device=device)

    # Our optimizer
    p1 = torch.nn.Parameter(p_init.clone(), requires_grad=True)
    p1.grad = grad.clone()
    opt1 = AdamW([p1], lr=lr, betas=betas, eps=1e-8, weight_decay=wd)
    opt1.step()

    # Torch AdamW
    p2 = torch.nn.Parameter(p_init.clone(), requires_grad=True)
    p2.grad = grad.clone()
    opt2 = torch.optim.AdamW([p2], lr=lr, betas=betas, eps=1e-8, weight_decay=wd)
    opt2.step()

    # Allow tiny numerical differences due to variant formulations of bias correction/decay
    assert torch.allclose(p1.data, p2.data, atol=1e-6, rtol=1e-5)
