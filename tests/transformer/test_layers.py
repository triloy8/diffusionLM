import torch

from diffusionlm.models import Linear, Embedding, RMSNorm, SwiGLU


def test_linear_matches_einsum(device):
    lin = Linear(in_features=3, out_features=2, device=device)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [0.0, -1.0, 1.0]], device=device))

    x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 1.0, 0.5]], device=device)
    y = lin(x)
    expected = torch.einsum("oi,bi->bo", lin.weight, x)
    assert torch.allclose(y, expected)


def test_embedding_indexing_and_backward(device):
    emb = Embedding(num_embeddings=5, embedding_dim=3, device=device)
    token_ids = torch.tensor([[1, 3, 1]], device=device)
    out = emb(token_ids)
    assert out.shape == (1, 3, 3)
    loss = out.sum()
    loss.backward()
    # Gradients only for indices 1 and 3
    used = {1, 3}
    with torch.no_grad():
        for idx in range(5):
            grad_row = emb.weight.grad[idx]
            if idx in used:
                assert torch.any(grad_row != 0)
            else:
                assert torch.all(grad_row == 0)


def test_rmsnorm_formula_and_dtype(device):
    x = torch.randn(2, 4, 6, device=device, dtype=torch.float32)
    rms = RMSNorm(d_model=6, device=device)
    y = rms(x)
    assert y.dtype == x.dtype

    # Manual computation with weight = 1
    with torch.no_grad():
        rms_val = torch.sqrt((x.float() ** 2).mean(dim=-1, keepdim=True) + rms.eps)
        expected = (x.float() / rms_val).to(x.dtype)
    assert torch.allclose(y, expected, atol=1e-5, rtol=1e-4)

    # Zero-variance edge case
    x0 = torch.zeros(1, 2, 6, device=device)
    y0 = rms(x0)
    assert torch.all(y0 == 0)


def test_swiglu_shapes_and_backward(device):
    # Ensure shapes propagate and backward works
    d_model, d_ff = 8, 16
    ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device)
    x = torch.randn(2, 3, d_model, device=device)
    y = ffn(x)
    assert y.shape == (2, 3, d_model)
    y.sum().backward()  # no error
