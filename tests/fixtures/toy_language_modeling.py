from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class ToyLanguageModelingDataset:
    """Deterministic, in-memory dataset for quick parity checks."""

    train_tokens: torch.Tensor
    valid_tokens: torch.Tensor
    vocab_size: int
    context_length: int

    def as_tuple(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (train_tokens, valid_tokens) for unpacking convenience."""
        return self.train_tokens, self.valid_tokens


def build_toy_language_modeling_dataset(
    *,
    device: torch.device,
    context_length: int,
    dtype: torch.dtype = torch.long,
) -> ToyLanguageModelingDataset:
    """Create a tiny dataset with fixed tokens for deterministic tests.

    The sequence is crafted to produce four distinct batches when
    `batch_size=2` and `context_length=2`. Validation tokens reuse the
    same pattern but reversed to exercise different slices.
    """
    base_tokens = torch.tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            0,
        ],
        dtype=dtype,
    )
    train_tokens = base_tokens.to(device=device)
    valid_tokens = torch.flip(base_tokens, dims=[0]).to(device=device)

    return ToyLanguageModelingDataset(
        train_tokens=train_tokens,
        valid_tokens=valid_tokens,
        vocab_size=int(base_tokens.max().item() + 1),
        context_length=context_length,
    )
