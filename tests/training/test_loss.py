import torch

from trainkit.objectives.loss import cross_entropy, diffusion_cross_entropy


def test_diffusion_cross_entropy_matches_weighted_average():
    logits = torch.log(torch.tensor([
        [[0.7, 0.3], [0.2, 0.8]],
    ], dtype=torch.float32))
    targets = torch.tensor([[0, 1]])
    mask = torch.tensor([[True, True]])
    p_mask = torch.tensor([[0.5, 1.0]])

    ce = cross_entropy(logits, targets, reduction="none")
    manual = ((ce[0, 0] / p_mask[0, 0]) + (ce[0, 1] / p_mask[0, 1])) / 2
    ours = diffusion_cross_entropy(logits, targets, mask, p_mask)
    assert torch.allclose(ours, manual)


def test_diffusion_cross_entropy_ignores_unmasked_tokens():
    logits = torch.log(torch.tensor([
        [[0.6, 0.4], [0.1, 0.9]],
    ], dtype=torch.float32))
    targets = torch.tensor([[0, 1]])
    mask = torch.tensor([[True, False]])
    p_mask = torch.tensor([[0.5, 0.5]])

    value = diffusion_cross_entropy(logits, targets, mask, p_mask)
    ce = cross_entropy(logits, targets, reduction="none")[0, 0]
    expected = (ce / p_mask[0, 0]) / (targets.numel())
    assert torch.allclose(value, expected)
