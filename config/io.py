from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List
import tomllib
from importlib import resources as importlib_resources
from pydantic import BaseModel


def _as_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _load_toml(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def asdict_pretty(obj: Any) -> Dict[str, Any]:
    """Convert a Pydantic model (or dataclass) to a JSON-serializable dict with str Paths."""

    def normalize(value: Any):
        if isinstance(value, BaseModel):
            value = value.model_dump()
        elif is_dataclass(value):
            value = asdict(value)

        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        return value

    result = normalize(obj)
    if not isinstance(result, dict):
        raise TypeError("asdict_pretty expects model/dataclass producing a mapping")
    return result


# ===== Resources helpers =====

def _resources_root():
    return importlib_resources.files(__package__).joinpath("resources")


def list_examples() -> List[str]:
    root = _resources_root()
    return sorted([p.name for p in root.iterdir() if p.is_file() and p.suffix == ".toml"])


def open_example(name: str):
    if "/" in name or ".." in name:
        raise ValueError("name must be a base filename")
    root = _resources_root()
    path = root.joinpath(name)
    if not path.exists():
        raise FileNotFoundError(f"example not found: {name}")
    return path.open("r", encoding="utf-8")
