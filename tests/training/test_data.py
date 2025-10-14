import numpy as np
import torch

from diffusionlm.training.data import get_batch


def test_get_batch_metadata_mask_ratio_and_truncation(device):
    arr = np.arange(100, dtype=np.int32)
    generator = torch.Generator(device="cpu").manual_seed(42)

    batch = get_batch(
        arr,
        batch_size=2,
        context_length=8,
        device=str(device),
        mask_token_id=999,
        noise_epsilon=0.05,
        random_trunc_prob=0.0,
        generator=generator,
    )

    assert "mask_ratio" in batch.metadata
    assert 0.0 <= batch.metadata["mask_ratio"] <= 1.0
    assert batch.metadata["random_truncation_applied"] is False


def test_get_batch_random_truncation(device):
    arr = np.arange(100, dtype=np.int32)
    generator = torch.Generator(device="cpu").manual_seed(123)

    batch = get_batch(
        arr,
        batch_size=1,
        context_length=16,
        device=str(device),
        mask_token_id=999,
        noise_epsilon=0.1,
        random_trunc_prob=1.0,
        generator=generator,
    )

    assert batch.clean_targets.shape[1] <= 16
    assert batch.metadata["random_truncation_applied"] is True
