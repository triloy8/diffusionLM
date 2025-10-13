from __future__ import annotations

import time
from typing import Callable, Tuple, TypeVar

T = TypeVar("T")


def _cuda_synchronize_if_needed(device: str) -> None:
    if device == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass


def measure(device: str, fn: Callable[[], T], *, synchronize: bool = True) -> Tuple[T, float]:
    """Measure wall time (seconds) of a callable, optionally syncing CUDA."""
    if synchronize:
        _cuda_synchronize_if_needed(device)
    t0 = time.perf_counter()
    out = fn()
    if synchronize:
        _cuda_synchronize_if_needed(device)
    dt = time.perf_counter() - t0
    return out, dt


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def stddev(xs: list[float]) -> float:
    if not xs:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
