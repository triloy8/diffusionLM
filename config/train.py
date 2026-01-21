from __future__ import annotations

from pathlib import Path

from .schemas import TrainConfig, TrainTokenizerConfig
from .io import _as_path, _load_toml


def load_train_config(path: Path | str) -> TrainConfig:
    data = _load_toml(_as_path(path))
    return TrainConfig.model_validate(data)


def load_train_tokenizer_config(path: Path | str) -> TrainTokenizerConfig:
    data = _load_toml(_as_path(path))
    return TrainTokenizerConfig.model_validate(data)
