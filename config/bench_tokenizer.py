from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import (
    TokenizerConfig,
    BenchTokenizerInput,
    BenchTokenizerParams,
    BenchTokenizerConfig,
    LoggingConfig,
)
from .io import _as_path, _expect_keys, _load_toml


def load_bench_tokenizer_config(path: Path | str) -> BenchTokenizerConfig:
    cfg = _load_toml(_as_path(path))
    _expect_keys(cfg, "root", ["tokenizer", "input", "benchmark"])
    t: Dict[str, Any] = cfg["tokenizer"]
    i: Dict[str, Any] = cfg["input"]
    b: Dict[str, Any] = cfg["benchmark"]
    lg: Dict[str, Any] = cfg.get("logging", {})

    tokenizer = TokenizerConfig(
        merges_path=_as_path(t["merges_path"]),
        vocab_path=_as_path(t["vocab_path"]),
        special_tokens=list(t.get("special_tokens", [])),
    )
    input_cfg = BenchTokenizerInput(text_list=list(i["text_list"]))
    bench = BenchTokenizerParams(repeats=int(b.get("repeats", 5)))

    logging: Optional[LoggingConfig] = None
    if lg:
        logging = LoggingConfig(
            backend=lg.get("backend"),
            run_name=lg.get("run_name"),
            architecture=lg.get("architecture"),
            dataset=lg.get("dataset"),
        )

    return BenchTokenizerConfig(tokenizer=tokenizer, input=input_cfg, benchmark=bench, logging=logging)
