import torch

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

def resolve_dtype(name: str) -> torch.dtype:
    try:
        return DTYPES[name]
    except KeyError as e:
        raise KeyError(f"Unknown dtype '{name}'. Expected one of: {sorted(DTYPES.keys())}") from e

