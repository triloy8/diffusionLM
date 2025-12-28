from __future__ import annotations

from pathlib import Path

from .schemas import SweepInferConfig
from .io import _as_path, _load_toml


def load_sweep_infer_config(path: Path | str) -> SweepInferConfig:
    data = _load_toml(_as_path(path))
    return SweepInferConfig.model_validate(data)
