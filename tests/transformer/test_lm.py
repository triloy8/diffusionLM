import torch

from diffusionlm.models import TransformerLM


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
