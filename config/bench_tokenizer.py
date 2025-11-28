from __future__ import annotations

from pathlib import Path

from .schemas import BenchTokenizerConfig
from .io import _as_path, _load_toml


def load_bench_tokenizer_config(path: Path | str) -> BenchTokenizerConfig:
    data = _load_toml(_as_path(path))
    return BenchTokenizerConfig.model_validate(data)
