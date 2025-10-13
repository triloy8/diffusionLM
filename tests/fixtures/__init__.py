"""Test fixtures package for reusable parity-testing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Any

from config import TrainConfig

from .toy_language_modeling import ToyLanguageModelingDataset, build_toy_language_modeling_dataset


@dataclass(frozen=True)
class TrainingBundle:
    dataset: ToyLanguageModelingDataset
    model_factory: Callable[[], Any]
    optimizer_factory: Callable[[Iterable], Any]
    train_config: TrainConfig


__all__ = [
    "ToyLanguageModelingDataset",
    "TrainingBundle",
    "build_toy_language_modeling_dataset",
]
