import torch

from diffusionlm.models import TransformerLM
from diffusionlm.inference import top_p_filter, generate


def make_tiny_model(device):
    return TransformerLM(
        vocab_size=16,
        context_length=8,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        rope_theta=10_000.0,
        device=device,
        dtype=torch.float32,
    )


def test_forward_shape(device):
    model = make_tiny_model(device)
    B, T = 2, 4
    x = torch.randint(0, 16, (B, T), device=device)
    logits = model(x)
    assert logits.shape == (B, T, 16)


def test_generate_grows_by_steps_without_eos(device):
    model = make_tiny_model(device)
    B, T0 = 1, 3
    x0 = torch.randint(0, 16, (B, T0), device=device)

    steps = 5
    out = generate(model, in_indices=x0.clone(), steps=steps, temperature=1.0, p=0.8, eos_token_id=None)
    assert out.shape[1] == T0 + steps


def test_generate_deterministic_with_p_zero(device):
    model = make_tiny_model(device)
    B, T0 = 1, 3
    x0 = torch.randint(0, 16, (B, T0), device=device)

    steps = 4
    out1 = generate(model, in_indices=x0.clone(), steps=steps, temperature=1.0, p=0.0, eos_token_id=None)
    out2 = generate(model, in_indices=x0.clone(), steps=steps, temperature=1.0, p=0.0, eos_token_id=None)
    # With p=0, sampling reduces to argmax → deterministic
    assert torch.equal(out1, out2)


def test_generate_early_stop_on_eos_with_argmax(monkeypatch, device):
    model = make_tiny_model(device)
    eos_id = 0

    # Patch forward to always produce logits where eos_id is the largest at the last position
    def fake_forward(self, in_indices):
        B, T = in_indices.shape
        V = self.lm_head.weight.shape[0]
        logits = torch.zeros(B, T, V, device=in_indices.device)
        logits[:, -1, :] = -1.0
        logits[:, -1, eos_id] = 10.0
        return logits

    monkeypatch.setattr(TransformerLM, "forward", fake_forward, raising=True)

    x0 = torch.randint(0, 16, (1, 3), device=device)
    out = generate(model, in_indices=x0.clone(), steps=10, temperature=1.0, p=0.0, eos_token_id=eos_id)
    # Should stop after appending eos once
    assert out.shape[1] == x0.shape[1] + 1
    assert out[0, -1].item() == eos_id


def test_top_p_filter_properties(device):
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]], device=device)
    # p=0 -> argmax one-hot
    f0 = top_p_filter(probs, p=0.0)
    assert torch.allclose(f0.sum(dim=-1), torch.ones(1, device=device))
    assert torch.equal(f0.argmax(dim=-1), torch.tensor([0], device=device))

    # p>=1 -> unchanged (normalized)
    f1 = top_p_filter(probs, p=1.0)
    assert torch.allclose(f1, probs, atol=1e-7)

    # minimal set cumulative >= p
    f07 = top_p_filter(probs, p=0.7)
    # 0.4 + 0.3 >= 0.7, so exactly two non-zeros
    assert (f07 > 0).sum().item() == 2
    assert torch.allclose(f07.sum(dim=-1), torch.ones(1, device=device))

    f08 = top_p_filter(probs, p=0.8)
    # 0.4 + 0.3 < 0.8, include 0.2 as well → three
    assert (f08 > 0).sum().item() == 3
    assert torch.allclose(f08.sum(dim=-1), torch.ones(1, device=device))


def test_generate_p0_invariant_to_temperature(device):
    model = make_tiny_model(device)
    x0 = torch.randint(0, 16, (1, 3), device=device)
    steps = 4
    out_cold = generate(model, in_indices=x0.clone(), steps=steps, temperature=0.5, p=0.0, eos_token_id=None)
    out_hot = generate(model, in_indices=x0.clone(), steps=steps, temperature=2.0, p=0.0, eos_token_id=None)
    # Argmax is invariant to positive scaling, so sequences should match
    assert torch.equal(out_cold, out_hot)


def test_top_p_candidate_set_monotonic(device):
    probs = torch.tensor([[0.5, 0.2, 0.15, 0.1, 0.05]], device=device)
    ps = [0.1, 0.3, 0.6, 0.8, 0.95]
    counts = []
    for p in ps:
        f = top_p_filter(probs, p=p)
        counts.append(int((f > 0).sum().item()))
        # Always renormalizes
        assert torch.allclose(f.sum(dim=-1), torch.ones(1, device=device))
    # Candidate set size should be non-decreasing with p
    assert counts == sorted(counts)
