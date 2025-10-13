from __future__ import annotations

from typing import Any, Dict, Optional, Union

try:
    import torch
except Exception:  # pragma: no cover - defensive import guard
    torch = None  # type: ignore


DeviceLike = Union[str, "torch.device"]


def _cuda_available() -> bool:
    try:
        return bool(torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available())
    except Exception:
        return False


def _to_cuda_device(device: DeviceLike = "cuda") -> Optional["torch.device"]:
    """Normalize device to a CUDA device or return None if not CUDA/available.

    Accepts strings like "cuda", "cuda:0" or torch.device objects.
    Returns None if CUDA is not available or the device is non-CUDA.
    """
    if not _cuda_available():
        return None
    try:
        dev = torch.device(device) if not isinstance(device, torch.device) else device  # type: ignore[arg-type]
        if dev.type != "cuda":
            return None
        return dev
    except Exception:
        return None


def start_history(enabled: bool, record_context: bool = True) -> bool:
    """Enable/disable CUDA allocator history recording (private API).

    Returns True if history recording is active after the call; False otherwise.
    Safely no-ops on CPU-only or when the API is unavailable.
    """
    if not _cuda_available():
        return False
    try:
        rec = getattr(torch.cuda.memory, "_record_memory_history", None)  # type: ignore[attr-defined]
        if rec is None:
            return False
        rec(enabled=bool(enabled), record_context=bool(record_context))  # type: ignore[misc]
        return bool(enabled)
    except Exception:
        return False


def snapshot() -> Optional[Dict[str, Any]]:
    """Return an in-memory allocator snapshot (dict) if supported; else None."""
    if not _cuda_available():
        return None
    try:
        snap_fn = getattr(torch.cuda.memory, "_snapshot", None)  # type: ignore[attr-defined]
        if snap_fn is None:
            return None
        return snap_fn()  # type: ignore[misc]
    except Exception:
        return None


def dump_snapshot(path: str) -> bool:
    """Write allocator history snapshot to a JSON file if supported; returns success."""
    if not _cuda_available():
        return False
    try:
        dump_fn = getattr(torch.cuda.memory, "_dump_snapshot", None)  # type: ignore[attr-defined]
        if dump_fn is None:
            return False
        dump_fn(path)  # type: ignore[misc]
        return True
    except Exception:
        return False


def summary(device: DeviceLike = "cuda", abbreviated: bool = True) -> str:
    """Return torch.cuda.memory_summary string or empty string on unsupported setups."""
    dev = _to_cuda_device(device)
    if dev is None:
        return ""
    try:
        return torch.cuda.memory_summary(dev, abbreviated=bool(abbreviated))
    except Exception:
        return ""


def reset_peaks(device: DeviceLike = "cuda") -> None:
    """Reset peak allocator statistics for the given device (safe no-op on CPU)."""
    dev = _to_cuda_device(device)
    if dev is None:
        return
    try:
        torch.cuda.reset_peak_memory_stats(dev)
    except Exception:
        return


def peaks(device: DeviceLike = "cuda") -> Dict[str, int]:
    """Return current and peak allocator counters in bytes for the given device.

    Keys: allocated, reserved, max_allocated, max_reserved
    Returns zeros on unsupported setups.
    """
    dev = _to_cuda_device(device)
    if dev is None:
        return {"allocated": 0, "reserved": 0, "max_allocated": 0, "max_reserved": 0}
    try:
        allocated = int(torch.cuda.memory_allocated(dev))
        reserved = int(torch.cuda.memory_reserved(dev))
        max_alloc = int(torch.cuda.max_memory_allocated(dev))
        max_res = int(torch.cuda.max_memory_reserved(dev))
        return {
            "allocated": allocated,
            "reserved": reserved,
            "max_allocated": max_alloc,
            "max_reserved": max_res,
        }
    except Exception:
        return {"allocated": 0, "reserved": 0, "max_allocated": 0, "max_reserved": 0}


__all__ = [
    "start_history",
    "snapshot",
    "dump_snapshot",
    "summary",
    "reset_peaks",
    "peaks",
]

