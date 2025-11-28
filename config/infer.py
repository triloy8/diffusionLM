from __future__ import annotations

from pathlib import Path

from .schemas import InferConfig
from .io import _as_path, _load_toml


def load_infer_config(path: Path | str) -> InferConfig:
    data = _load_toml(_as_path(path))
    return InferConfig.model_validate(data)
