import torch
import pytest

from transformerlm.models import MultiheadSelfAttentionRoPE, scaled_dot_product_attention
from transformerlm.models import RotaryPositionalEmbedding
from transformerlm.inference import softmax


def test_causal_mask_invariance(device):
    # Test invariance using the core scaled_dot_product_attention function
    B, T, d_k = 1, 4, 6
    Q = torch.randn(B, T, d_k, device=device)
    K = torch.randn(B, T, d_k, device=device)
    V = torch.randn(B, T, d_k, device=device)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    out1 = scaled_dot_product_attention(Q, K, V, mask)

    # Modify future tokens (> i) and confirm outputs up to i unchanged
    i = 1
    Q2, K2, V2 = Q.clone(), K.clone(), V.clone()
    Q2[:, i + 1 :, :] += 1.0
    K2[:, i + 1 :, :] += 1.0
    V2[:, i + 1 :, :] += 1.0
    out2 = scaled_dot_product_attention(Q2, K2, V2, mask)

    assert torch.allclose(out1[:, : i + 1, :], out2[:, : i + 1, :], atol=1e-6, rtol=1e-5)


def test_softmax_matches_torch(device):
    logits = torch.tensor([[1000.0, 0.0, -1000.0], [0.2, -0.3, 0.1]], device=device)
    sx = softmax(logits, dim=-1)
    tx = torch.nn.functional.softmax(logits, dim=-1)
    assert torch.allclose(sx, tx, atol=1e-7, rtol=1e-6)


def test_rope_shapes_and_norm_preservation(device):
    # d_k is the full per-head feature dimension and must be even for pairwise rotation
    d_k = 4
    max_seq_len = 8
    rope = RotaryPositionalEmbedding(theta=10_000.0, d_k=d_k, max_seq_len=max_seq_len, device=device)

    B, T = 2, 5
    # rope.forward expects last dim to be even; it internally reshapes into pairs of 2
    # Provide last dim = d_k (full head dim)
    x = torch.randn(B, T, d_k, device=device)
    pos = torch.arange(T, dtype=torch.long, device=device)

    y = rope(x, pos)
    assert y.shape == x.shape
    assert y.device == x.device

    # Pairwise 2-norm preservation per (real, imag) pair
    x_pairs = x.view(B, T, d_k // 2, 2)
    y_pairs = y.view(B, T, d_k // 2, 2)
    x_norms = torch.linalg.vector_norm(x_pairs, dim=-1)
    y_norms = torch.linalg.vector_norm(y_pairs, dim=-1)
    assert torch.allclose(x_norms, y_norms, atol=1e-5, rtol=1e-4)


def test_end_to_end_causal_invariance_with_mhsa_rope(device):
    d_model = 8
    num_heads = 2  # ensures per-head dim is even
    max_seq_len = 8
    theta = 10_000.0
    attn = MultiheadSelfAttentionRoPE(d_model=d_model,
                                      num_heads=num_heads,
                                      max_seq_len=max_seq_len,
                                      theta=theta,
                                      device=device)

    T = 5
    x = torch.randn(1, T, d_model, device=device)
    tok_pos = torch.arange(T, dtype=torch.long, device=device)

    out1 = attn(x, tok_pos)

    i = 2
    x2 = x.clone()
    x2[:, i + 1 :, :] += 1.0
    out2 = attn(x2, tok_pos)

    assert torch.allclose(out1[:, : i + 1, :], out2[:, : i + 1, :], atol=1e-6, rtol=1e-5)


def test_head_split_merge_preserves_feature_dimension(device):
    d_model = 12
    num_heads = 3
    max_seq_len = 8
    theta = 10_000.0
    attn = MultiheadSelfAttentionRoPE(d_model=d_model,
                                      num_heads=num_heads,
                                      max_seq_len=max_seq_len,
                                      theta=theta,
                                      device=device)
    x = torch.randn(2, 4, d_model, device=device)
    pos = torch.arange(4, dtype=torch.long, device=device)
    y = attn(x, pos)
    assert y.shape == x.shape
