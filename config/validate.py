from __future__ import annotations

from pathlib import Path

from .schemas import (
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    DataConfig,
    InferenceConfig,
    TokenizerConfig,
    BenchParams,
    BenchDataConfig,
    BenchTokenizerInput,
)

ALLOWED_DTYPES = {"float32", "float16", "bfloat16"}
ALLOWED_DEVICES = {"cpu", "cuda"}


def _validate_model(m: ModelConfig) -> None:
    if m.d_model % m.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    if m.dtype not in ALLOWED_DTYPES:
        raise ValueError(f"dtype must be one of {sorted(ALLOWED_DTYPES)}")
    if m.device not in ALLOWED_DEVICES:
        raise ValueError(f"device must be one of {sorted(ALLOWED_DEVICES)}")
    for k in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff"):
        if getattr(m, k) <= 0:
            raise ValueError(f"{k} must be > 0")
    if m.rope_theta <= 0:
        raise ValueError("rope_theta must be > 0")


def _validate_optimizer(o: OptimizerConfig) -> None:
    if len(o.betas) != 2:
        raise ValueError("betas must have 2 elements")
    if not (0 <= o.betas[0] < 1 and 0 <= o.betas[1] < 1):
        raise ValueError("betas must be in [0,1)")
    for k in ("eps", "initial_learning_rate", "max_learning_rate", "min_learning_rate", "grad_clip_max_l2_norm"):
        if getattr(o, k) <= 0:
            raise ValueError(f"{k} must be > 0")
    for k in ("warmup_iters", "cosine_cycle_iters"):
        if getattr(o, k) < 0:
            raise ValueError(f"{k} must be >= 0")
    if o.min_learning_rate > o.max_learning_rate:
        raise ValueError("min_learning_rate must be <= max_learning_rate")
    if o.initial_learning_rate > o.max_learning_rate:
        raise ValueError("initial_learning_rate must be <= max_learning_rate")


def _validate_training(t: TrainingConfig) -> None:
    for k in ("batch_size", "max_train_iteration", "max_val_iteration", "val_freq_iteration", "ckpting_save_iter"):
        if getattr(t, k) <= 0:
            raise ValueError(f"{k} must be > 0")
    if t.seed is not None and t.seed < 0:
        raise ValueError("seed must be >= 0")


def _validate_data(d: DataConfig) -> None:
    # Inputs should exist; create of outputs handled by caller.
    if not d.np_dat_train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {d.np_dat_train_path}")
    if not d.np_dat_valid_path.exists():
        raise FileNotFoundError(f"Validation dataset not found: {d.np_dat_valid_path}")
    if d.total_train_tokens <= 0 or d.total_val_tokens <= 0:
        raise ValueError("total_*_tokens must be > 0")


def _validate_inference(i: InferenceConfig) -> None:
    if i.temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not (0 <= i.p <= 1):
        raise ValueError("p must be in [0, 1]")
    if i.eos_token_id < 0:
        raise ValueError("eos_token_id must be >= 0")
    if not i.text_list:
        raise ValueError("text_list must not be empty")


def _validate_tokenizer(tok: TokenizerConfig) -> None:
    if not tok.vocab_path.exists():
        raise FileNotFoundError(f"vocab_path not found: {tok.vocab_path}")
    if not tok.merges_path.exists():
        raise FileNotFoundError(f"merges_path not found: {tok.merges_path}")


def _validate_bench_params(b: BenchParams) -> None:
    if b.warmup < 0:
        raise ValueError("warmup must be >= 0")
    if b.repeats <= 0:
        raise ValueError("repeats must be > 0")
    if b.steps <= 0:
        raise ValueError("steps must be > 0")
    if b.perplexity_max_batches is not None and b.perplexity_max_batches <= 0:
        raise ValueError("perplexity_max_batches must be > 0 when provided")
    if b.perplexity_batch_size is not None and b.perplexity_batch_size <= 0:
        raise ValueError("perplexity_batch_size must be > 0 when provided")
    if b.perplexity_seed is not None and b.perplexity_seed < 0:
        raise ValueError("perplexity_seed must be >= 0 when provided")


def _validate_bench_data(d: BenchDataConfig) -> None:
    if not d.np_dat_valid_path.exists():
        raise FileNotFoundError(f"validation dataset not found: {d.np_dat_valid_path}")
    if d.total_val_tokens <= 0:
        raise ValueError("total_val_tokens must be > 0")


def _validate_tokenizer_bench_input(i: BenchTokenizerInput) -> None:
    if not i.text_list:
        raise ValueError("input.text_list must not be empty")
