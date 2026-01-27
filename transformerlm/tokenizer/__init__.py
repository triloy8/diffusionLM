from .tokenizer import Tokenizer as PythonTokenizer
from .pretokenize import gpt2_bytes_to_unicode, PAT as PRETOKENIZE_PAT
from .io import find_chunk_boundaries, process_chunk_text

Tokenizer = PythonTokenizer

try:  # pragma: no cover - optional dependency
    from transformerlm.tokenizer_rust import Tokenizer as RustTokenizer  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional dependency
    RustTokenizer = None

__all__ = [
    "Tokenizer",
    "PythonTokenizer",
    "RustTokenizer",
    "gpt2_bytes_to_unicode",
    "PRETOKENIZE_PAT",
    "find_chunk_boundaries",
    "process_chunk_text",
]
