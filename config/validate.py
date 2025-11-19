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
    MuonOptimizerConfig,
)

ALLOWED_DTYPES = {"float32", "float16", "bfloat16"}
ALLOWED_DEVICES = {"cpu", "cuda"}
ALLOWED_OPTIMIZERS = {"adamw", "muon"}


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
    if not (0 <= m.mask_token_id < m.vocab_size):
        raise ValueError("mask_token_id must be in [0, vocab_size)")
    if not (0.0 < m.noise_epsilon <= 1.0):
        raise ValueError("noise_epsilon must be in (0, 1]")
    if not (0.0 <= m.random_trunc_prob <= 1.0):
        raise ValueError("random_trunc_prob must be in [0, 1]")


def _validate_optimizer(o: OptimizerConfig) -> None:
    optimizer_name = o.optimizer_name.lower()
    if optimizer_name not in ALLOWED_OPTIMIZERS:
        raise ValueError(f"optimizer_name must be one of {sorted(ALLOWED_OPTIMIZERS)}")
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
    if optimizer_name == "muon":
        if o.muon is None:
            raise ValueError("Muon optimizer requires [optimizer.muon] configuration")
        _validate_muon_config(o.muon)


def _validate_muon_config(muon: MuonOptimizerConfig) -> None:
    def _check_lr_range(initial: float, min_lr: float, max_lr: float, label: str) -> None:
        if initial <= 0 or min_lr <= 0 or max_lr <= 0:
            raise ValueError(f"{label}: learning rates must be > 0")
        if min_lr > max_lr:
            raise ValueError(f"{label}: min_learning_rate must be <= max_learning_rate")
        if initial > max_lr or initial < min_lr:
            raise ValueError(f"{label}: initial_learning_rate must be within [min, max]")

    hidden = muon.hidden
    _check_lr_range(hidden.initial_learning_rate, hidden.min_learning_rate, hidden.max_learning_rate, "muon.hidden")
    if hidden.momentum <= 0 or hidden.momentum >= 1:
        raise ValueError("muon.hidden.momentum must be in (0, 1)")
    if hidden.weight_decay < 0:
        raise ValueError("muon.hidden.weight_decay must be >= 0")

    for key in ("head", "embed", "scalar"):
        group = getattr(muon, key)
        _check_lr_range(group.initial_learning_rate, group.min_learning_rate, group.max_learning_rate, f"muon.{key}")
        if not (0 <= group.betas[0] < 1 and 0 <= group.betas[1] < 1):
            raise ValueError(f"muon.{key}.betas must be in [0,1)")
        if len(group.betas) != 2:
            raise ValueError(f"muon.{key}.betas must have 2 elements")
        if group.eps <= 0:
            raise ValueError(f"muon.{key}.eps must be > 0")
        if group.weight_decay < 0:
            raise ValueError(f"muon.{key}.weight_decay must be >= 0")


def _validate_training(t: TrainingConfig) -> None:
    for k in ("batch_size", "max_train_iteration", "max_val_iteration", "val_freq_iteration", "ckpting_save_iter"):
        if getattr(t, k) <= 0:
            raise ValueError(f"{k} must be > 0")
    if t.seed is not None and t.seed < 0:
        raise ValueError("seed must be >= 0")


def _validate_data(d: DataConfig) -> None:
    if not d.dataset_name:
        raise ValueError("dataset_name must not be empty")
    if not d.train_split:
        raise ValueError("train_split must not be empty")
    if not d.val_split:
        raise ValueError("val_split must not be empty")
    if not d.text_field:
        raise ValueError("text_field must not be empty")
    if d.shuffle_buffer_size < 0:
        raise ValueError("shuffle_buffer_size must be >= 0")
    if d.shuffle_seed is not None and d.shuffle_seed < 0:
        raise ValueError("shuffle_seed must be >= 0 when provided")
    _validate_tokenizer(d.tokenizer)


def _validate_inference(i: InferenceConfig) -> None:
    if not i.prompt:
        raise ValueError("prompt must not be empty")
    if i.steps <= 0:
        raise ValueError("steps must be > 0")
    if i.total_length <= 0:
        raise ValueError("total_length must be > 0")
    if i.block_length <= 0:
        raise ValueError("block_length must be > 0")
    if i.temperature <= 0:
        raise ValueError("temperature must be > 0")
    if i.mask_id < 0:
        raise ValueError("mask_id must be >= 0")


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
