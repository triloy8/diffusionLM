from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from typing import Iterator, Optional


def _env_enabled() -> bool:
    val = os.getenv("TRANSFORMERLM_NVTX", "0").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _env_level() -> str:
    level = os.getenv("TRANSFORMERLM_NVTX_LEVEL", "coarse").strip().lower()
    return level if level in {"coarse", "fine", "verbose"} else "coarse"


_ENABLED = _env_enabled()
_LEVEL = _env_level()


def _level_enabled(requested: Optional[str]) -> bool:
    if not _ENABLED:
        return False
    if requested is None:
        return True
    order = {"coarse": 0, "fine": 1, "verbose": 2}
    return order[_LEVEL] >= order[requested]


# Backend resolution
_torch_nvtx = None
_py_nvtx = None
try:
    import torch  # type: ignore

    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        from torch.cuda import nvtx as _torch_nvtx  # type: ignore
except Exception:
    _torch_nvtx = None  # type: ignore

if _torch_nvtx is None:
    try:
        # Optional Python NVTX package (nvidia-nvtx-cu12 / nvtx)
        import nvtx as _py_nvtx  # type: ignore
    except Exception:
        _py_nvtx = None  # type: ignore


def enabled(level: Optional[str] = None) -> bool:
    """Return True if NVTX is globally enabled and the requested level is active.

    level can be one of: None, "coarse", "fine", "verbose".
    """
    if level is not None and level not in {"coarse", "fine", "verbose"}:
        # Unknown level -> treat as disabled
        return False
    return _level_enabled(level)


@contextmanager
def range(name: str, color: Optional[int] = None, category: Optional[str] = None) -> Iterator[None]:
    """Open an NVTX range if enabled; otherwise a no-op context manager.

    color and category are best-effort and only used by the Python nvtx backend.
    """
    if not _ENABLED:
        yield
        return

    if _torch_nvtx is not None:
        try:
            _torch_nvtx.range_push(name)
            try:
                yield
            finally:
                _torch_nvtx.range_pop()
            return
        except Exception:
            # Fall through to no-op on errors
            pass

    if _py_nvtx is not None:
        try:
            with _py_nvtx.annotate(message=name, color=color, category=category):
                yield
            return
        except Exception:
            pass

    # No backend available: no-op
    with nullcontext():
        yield


def mark(name: str) -> None:
    """Insert a point marker if enabled; otherwise no-op."""
    if not _ENABLED:
        return
    try:
        if _torch_nvtx is not None:
            _torch_nvtx.mark(name)
            return
    except Exception:
        pass
    try:
        if _py_nvtx is not None:
            _py_nvtx.mark(message=name)
            return
    except Exception:
        pass


__all__ = ["enabled", "range", "mark"]

