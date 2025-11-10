from __future__ import annotations

import warnings

from .tokenizer import Tokenizer as PythonTokenizer
from .pretokenize import gpt2_bytes_to_unicode, PAT as PRETOKENIZE_PAT
from .io import find_chunk_boundaries, process_chunk_text

try:
    from diffusionlm.tokenizer_rust import Tokenizer as RustTokenizer  # type: ignore[attr-defined]

    Tokenizer = RustTokenizer
    USING_RUST_TOKENIZER = True
except Exception as exc:  # pragma: no cover - only triggered when extension missing
    warnings.warn(
        f"Falling back to Python tokenizer because the Rust extension is unavailable ({exc}).",
        RuntimeWarning,
        stacklevel=2,
    )
    Tokenizer = PythonTokenizer
    USING_RUST_TOKENIZER = False

__all__ = [
    "Tokenizer",
    "USING_RUST_TOKENIZER",
    "PythonTokenizer",
    "gpt2_bytes_to_unicode",
    "PRETOKENIZE_PAT",
    "find_chunk_boundaries",
    "process_chunk_text",
]
