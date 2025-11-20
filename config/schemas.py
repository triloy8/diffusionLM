from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# ===== Dataclasses (Schemas) =====

@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    device: str
    dtype: str
    mask_token_id: int
    noise_epsilon: float = 1e-3
    random_trunc_prob: float = 0.01


@dataclass
class MuonHiddenConfig:
    initial_learning_rate: float
    max_learning_rate: float
    min_learning_rate: float
    momentum: float
    weight_decay: float


@dataclass
class MuonAdamGroupConfig:
    initial_learning_rate: float
    max_learning_rate: float
    min_learning_rate: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float


@dataclass
class MuonOptimizerConfig:
    hidden: MuonHiddenConfig
    head: MuonAdamGroupConfig
    embed: MuonAdamGroupConfig
    scalar: MuonAdamGroupConfig


@dataclass
class OptimizerConfig:
    optimizer_name: str
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    initial_learning_rate: float
    max_learning_rate: float
    min_learning_rate: float
    warmup_iters: int
    cosine_cycle_iters: int
    grad_clip_max_l2_norm: float
    muon: Optional[MuonOptimizerConfig] = None


@dataclass
class TrainingConfig:
    batch_size: int
    max_train_iteration: int
    max_val_iteration: int
    val_freq_iteration: int
    ckpting_save_iter: int
    seed: Optional[int] = None
    skip_validation: bool = False


@dataclass
class DataConfig:
    runs_path: Path
    dataset_name: str
    dataset_config: Optional[str]
    train_split: str
    val_split: str
    text_field: str
    tokenizer: "TokenizerConfig"
    shuffle_buffer_size: int = 0
    shuffle_seed: Optional[int] = None


@dataclass
class WandbConfig:
    entity: Optional[str] = None
    project: Optional[str] = None
    architecture: Optional[str] = None
    dataset: Optional[str] = None


@dataclass
class LoggingConfig:
    backend: Optional[str] = None  # "console" | "wandb" | "noop" | "jsonl"
    run_name: Optional[str] = None
    architecture: Optional[str] = None
    dataset: Optional[str] = None


@dataclass
class DdpConfig:
    backend: str = "nccl"
    num_nodes: int = 1
    node_rank: int = 0
    num_gpus_per_node: int = 1
    master_addr: str = "localhost"
    master_port: str = "29500"
    bucket_size_mb: int = 0


@dataclass
class TrainConfig:
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    data: DataConfig
    wandb: Optional[WandbConfig] = None
    logging: Optional[LoggingConfig] = None
    ddp: Optional[DdpConfig] = None


@dataclass
class TokenizerConfig:
    merges_path: Path
    vocab_path: Path
    special_tokens: List[str]


@dataclass
class CheckpointConfig:
    ckpt_path: Path


@dataclass
class InferenceConfig:
    prompt: str
    steps: int
    total_length: int
    block_length: int
    temperature: float
    mask_id: int


@dataclass
class InferConfig:
    tokenizer: TokenizerConfig
    model: ModelConfig
    checkpoint: CheckpointConfig
    inference: InferenceConfig
    logging: Optional[LoggingConfig] = None


@dataclass
class MakeDataInputConfig:
    input_filename: Path
    total_tokens: int


@dataclass
class MakeDataOutputConfig:
    output_filename: Path


@dataclass
class MakeDataConfig:
    input: MakeDataInputConfig
    output: MakeDataOutputConfig
    tokenizer: TokenizerConfig


@dataclass
class TrainTokenizerInputConfig:
    input_path: Path
    vocab_size: int
    special_tokens: List[str]


@dataclass
class TrainTokenizerOutputConfig:
    merges_path: Path
    vocab_path: Path


@dataclass
class TrainTokenizerConfig:
    input: TrainTokenizerInputConfig
    output: TrainTokenizerOutputConfig


# ===== Benchmark Schemas =====

@dataclass
class BenchParams:
    warmup: int
    repeats: int
    steps: int
    synchronize: bool = True
    backward: bool = False
    optimizer_step: bool = False
    perplexity_max_batches: Optional[int] = None
    perplexity_batch_size: Optional[int] = None
    perplexity_seed: Optional[int] = None


@dataclass
class BenchDataConfig:
    dataset_name: str
    dataset_config: Optional[str]
    split: str
    text_field: str
    shuffle_buffer_size: int = 0
    shuffle_seed: Optional[int] = None


@dataclass
class BenchInferConfig:
    tokenizer: TokenizerConfig
    model: ModelConfig
    checkpoint: CheckpointConfig
    inference: InferenceConfig
    benchmark: BenchParams
    data: Optional[BenchDataConfig] = None
    logging: Optional[LoggingConfig] = None
    optimizer: Optional["OptimizerBenchConfig"] = None


@dataclass
class OptimizerBenchConfig:
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    grad_clip_max_l2_norm: float = 0.0


@dataclass
class BenchTokenizerInput:
    text_list: List[str]


@dataclass
class BenchTokenizerParams:
    repeats: int


@dataclass
class BenchTokenizerConfig:
    tokenizer: TokenizerConfig
    input: BenchTokenizerInput
    benchmark: BenchTokenizerParams
    logging: Optional[LoggingConfig] = None
