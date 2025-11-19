from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import (
    TokenizerConfig,
    ModelConfig,
    CheckpointConfig,
    InferenceConfig,
    BenchParams,
    BenchInferConfig,
    BenchDataConfig,
    LoggingConfig,
    OptimizerBenchConfig,
)
from .io import _as_path, _expect_keys, _load_toml
from .validate import (
    _validate_tokenizer,
    _validate_model,
    _validate_inference,
    _validate_bench_params,
    _validate_bench_data,
)


def load_bench_infer_config(path: Path | str) -> BenchInferConfig:
    cfg = _load_toml(_as_path(path))
    _expect_keys(cfg, "root", ["tokenizer", "model", "checkpoint", "inference", "benchmark"])

    tok: Dict[str, Any] = cfg["tokenizer"]
    m: Dict[str, Any] = cfg["model"]
    c: Dict[str, Any] = cfg["checkpoint"]
    i: Dict[str, Any] = cfg["inference"]
    b: Dict[str, Any] = cfg["benchmark"]
    lg: Dict[str, Any] = cfg.get("logging", {})
    opt_tbl: Dict[str, Any] = cfg.get("optimizer", {})
    data_tbl: Optional[Dict[str, Any]] = cfg.get("data")

    tokenizer = TokenizerConfig(
        merges_path=_as_path(tok["merges_path"]),
        vocab_path=_as_path(tok["vocab_path"]),
        special_tokens=list(tok.get("special_tokens", [])),
    )
    vocab_size = int(m["vocab_size"])
    model = ModelConfig(
        vocab_size=vocab_size,
        context_length=int(m["context_length"]),
        d_model=int(m["d_model"]),
        num_layers=int(m["num_layers"]),
        num_heads=int(m["num_heads"]),
        d_ff=int(m["d_ff"]),
        rope_theta=float(m["rope_theta"]),
        device=str(m["device"]),
        dtype=str(m["dtype"]),
        mask_token_id=int(m.get("mask_token_id", vocab_size - 1)),
        noise_epsilon=float(m.get("noise_epsilon", 1e-3)),
        random_trunc_prob=float(m.get("random_trunc_prob", 0.01)),
    )
    checkpoint = CheckpointConfig(ckpt_path=_as_path(c["ckpt_path"]))
    inference = InferenceConfig(
        prompt=str(i["prompt"]),
        steps=int(i["steps"]),
        total_length=int(i.get("total_length", model.context_length)),
        block_length=int(i["block_length"]),
        temperature=float(i.get("temperature", 1.0)),
        mask_id=int(i.get("mask_id", model.mask_token_id)),
    )
    benchmark = BenchParams(
        warmup=int(b.get("warmup", 2)),
        repeats=int(b.get("repeats", 5)),
        steps=int(b.get("steps", model.context_length)),
        synchronize=bool(b.get("synchronize", True)),
        backward=bool(b.get("backward", False)),
        optimizer_step=bool(b.get("optimizer_step", False)),
        perplexity_max_batches=int(b["perplexity_max_batches"]) if "perplexity_max_batches" in b else None,
        perplexity_batch_size=int(b["perplexity_batch_size"]) if "perplexity_batch_size" in b else None,
        perplexity_seed=int(b["perplexity_seed"]) if "perplexity_seed" in b else None,
    )

    data_config: Optional[BenchDataConfig] = None
    if data_tbl:
        _expect_keys(data_tbl, "data", ["dataset_name", "split", "text_field"])
        shuffle_seed_val = data_tbl.get("shuffle_seed")
        data_config = BenchDataConfig(
            dataset_name=str(data_tbl["dataset_name"]),
            dataset_config=(str(data_tbl["dataset_config"]) if data_tbl.get("dataset_config") is not None else None),
            split=str(data_tbl["split"]),
            text_field=str(data_tbl["text_field"]),
            shuffle_buffer_size=int(data_tbl.get("shuffle_buffer_size", 0)),
            shuffle_seed=(int(shuffle_seed_val) if shuffle_seed_val is not None else None),
        )

    _validate_tokenizer(tokenizer)
    _validate_model(model)
    _validate_inference(inference)
    _validate_bench_params(benchmark)
    if data_config is not None:
        _validate_bench_data(data_config)
    if not checkpoint.ckpt_path.exists():
        raise FileNotFoundError(f"ckpt_path not found: {checkpoint.ckpt_path}")

    logging: Optional[LoggingConfig] = None
    if lg:
        logging = LoggingConfig(
            backend=lg.get("backend"),
            run_name=lg.get("run_name"),
            architecture=lg.get("architecture"),
            dataset=lg.get("dataset"),
        )

    # Optimizer config (only needed if optimizer_step enabled; still safe to construct with defaults)
    optimizer: Optional[OptimizerBenchConfig] = None
    # Defaults emphasize stability (no weight updates) but include step overhead
    default_lr = 0.0
    default_betas = (0.9, 0.999)
    default_eps = 1e-8
    default_wd = 0.0
    default_clip = 0.0
    if benchmark.optimizer_step or opt_tbl:
        lr = float(opt_tbl.get("lr", default_lr))
        betas = opt_tbl.get("betas", list(default_betas))
        if isinstance(betas, tuple):
            betas_tuple = (float(betas[0]), float(betas[1]))
        else:
            betas_tuple = (float(betas[0]), float(betas[1]))
        eps = float(opt_tbl.get("eps", default_eps))
        wd = float(opt_tbl.get("weight_decay", default_wd))
        clip = float(opt_tbl.get("grad_clip_max_l2_norm", default_clip))
        optimizer = OptimizerBenchConfig(
            lr=lr,
            betas=betas_tuple,
            eps=eps,
            weight_decay=wd,
            grad_clip_max_l2_norm=clip,
        )
    return BenchInferConfig(
        tokenizer=tokenizer,
        model=model,
        checkpoint=checkpoint,
        inference=inference,
        benchmark=benchmark,
        data=data_config,
        logging=logging,
        optimizer=optimizer,
    )
