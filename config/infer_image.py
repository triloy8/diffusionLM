from __future__ import annotations

from pathlib import Path

from .schemas import ImageInferConfig
from .io import _as_path, _load_toml


def load_image_infer_config(path: Path | str) -> ImageInferConfig:
    data = _load_toml(_as_path(path))
    return ImageInferConfig.model_validate(data)
