from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
import tomllib
from importlib import resources as importlib_resources


def _as_path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _load_toml(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def asdict_pretty(dc) -> Dict[str, Any]:
    # Convert dataclass to dict with stringified Paths for printing
    def _stringify(obj):
        if isinstance(obj, Path):
            return str(obj)
        return obj

    d = asdict(dc)

    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return _stringify(x)

    return walk(d)


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
