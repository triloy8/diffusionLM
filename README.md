# Transformer Language Model from Scratch

## What Is This?

A minimal, from‑scratch Transformer language model implementation with a small, practical toolset (tokenizer, dataset builder, CLI, benchmarks, logging). The focus is on clarity and readability rather than feature breadth or scale.

## Overview

- From‑scratch model: decoder‑only Transformer LM (RMSNorm, SwiGLU, RoPE, SDPA/MHA), implemented directly with PyTorch modules.
- From‑scratch training: AdamW optimizer, cosine LR schedule, gradient clipping, checkpointing.
- From‑scratch tokenizer: byte‑level BPE training and IO, producing `vocab.json` and `merges.txt`.
- From‑scratch inference: temperature and top‑p sampling with optional EOS handling.
- Databuilder: memmap pipeline for large corpora (token counting and ID writing).
- CLI + TOML configs: consistent, simple entry points.
- Logging: console JSON or Weights & Biases.
- Distributed training (DDP): minimal wrapper with async gradient buckets, optimizer state sharding, rank-zero logging, and aggregated metrics.
- Benchmarking: tokenizer and inference throughput checks plus optional perplexity summaries.
- Profiling: small helpers for memory and runtime inspection.

## Installation

Requires Python 3.11–3.12 and PyTorch. Using [`uv`](https://github.com/astral-sh/uv) is recommended.

- Quick run without installing the package:

```bash
# Print an example resolved config
uv run transformerlm-train --config config/resources/train.toml --print-config
```

- Or install the package locally (editable):

```bash
uv pip install -e .
```

## Usage Examples

Entry points live in `cli/` and are driven by TOML configs in `config/resources/`.

- Train the tokenizer (BPE):

```bash
uv run transformerlm-train-tokenizer --config config/resources/train_tokenizer.toml
```

- Start model training:

```bash
uv run transformerlm-train --config config/resources/train.toml
```

- Train with DDP:

```bash
uv run transformerlm-train-ddp --config config/resources/train_ddp.toml
```

Notes:
- CPU uses `gloo`; CUDA uses `nccl`. Set `[model].device` and `[ddp].backend` accordingly.
- Prefer `[logging].backend = "console"` for local runs.
- Optimizer state sharding is enabled by default in the DDP entry point to reduce per-rank optimizer memory.

- Generate text:

```bash
uv run transformerlm-infer --config config/resources/infer.toml
```

- Build memmap datasets from raw text:

```bash
uv run transformerlm-make-data --config config/resources/make_data.toml
```

- Inspect effective configuration without running:

```bash
uv run transformerlm-train --config config/resources/train.toml --print-config
```

## Modules

- transformerlm.models
  - Purpose: Core Transformer components and the decoder‑only LM.
  - Key files: `transformerlm/models/transformer.py`, `transformerlm/models/attention.py`, `transformerlm/models/layers.py`.
  - Notes: dtype helpers under `transformerlm/utils/dtypes.py`.

- transformerlm.training
  - Purpose: Training loop, loss, optimizer, schedule, checkpointing, and batching over memmap data.
  - Key files: `transformerlm/training/trainer.py`, `transformerlm/training/loop.py`, `transformerlm/training/optim.py`, `transformerlm/training/schedule.py`, `transformerlm/training/checkpoint.py`, `transformerlm/training/data.py`.

- transformerlm.inference
  - Purpose: Sampling utilities and simple generation helpers.
  - Key files: `transformerlm/inference/generate.py`, `transformerlm/inference/sampling.py`, `transformerlm/inference/predictor.py`.

- transformerlm.tokenizer
  - Purpose: From‑scratch byte‑level BPE trainer and tokenizer IO.
  - Key files: `transformerlm/tokenizer/bpe_trainer.py`, `transformerlm/tokenizer/tokenizer.py`, `transformerlm/tokenizer/pretokenize.py`, `transformerlm/tokenizer/io.py`.
  - Artifacts: `vocab.json`, `merges.txt` (with optional special tokens).

- databuilder
  - Purpose: Dataset building helpers for large corpora (memmap writer, token counting).
  - Key files: `databuilder/dataset_builder.py`.
  - Usage: driven via `transformerlm-make-data` and `config/resources/make_data.toml`.

- cli
  - Purpose: Command‑line entry points wrapping configs and orchestration.
  - Key files: `cli/train.py`, `cli/infer.py`, `cli/make_data.py`, `cli/train_tokenizer.py`, `cli/utils.py`.
  - Scripts: exposed in `pyproject.toml` under `[project.scripts]`.

- logger
  - Purpose: Pluggable logging backends (console JSON and Weights & Biases).
  - Key files: `logger/base.py`, `logger/console_logger.py`, `logger/wandb_logger.py`, `logger/noop.py`, `logger/rank_zero.py`.

- ddp
  - Purpose: Minimal DDP wrapper, optimizer state sharding, and helpers (process group setup/cleanup, broadcast, all-reduce, metric reduction).
  - Key files: `ddp/ddp.py`, `ddp/optimizer_state_sharding.py`, `ddp/utils.py`.

- benchmarking
  - Purpose: Quick throughput checks for inference and tokenizer.
  - Key files: `benchmarking/bench_infer_latency.py`, `benchmarking/bench_tokenizer.py`.
  - Configs: `config/resources/bench_infer.toml`, `config/resources/bench_tokenizer.toml`.

- config
  - Purpose: Typed config schemas, loaders, validation, and example TOMLs.
  - Key files: `config/train.py`, `config/infer.py`, `config/bench_infer.py`, `config/bench_tokenizer.py`, `config/io.py`, `config/schemas.py`.
  - Examples: `config/resources/*.toml`.

- profiling
  - Purpose: Lightweight helpers for memory/runtime profiling, including NVTX ranges.
  - Key files: `profiling/memory.py`, `profiling/nvtx.py`.

- utils
  - Purpose: Small shared helpers.
  - Key files: `transformerlm/utils/dtypes.py`.

## Benchmarking

- Benchmarks live under `benchmarking/` and are TOML‑driven, similar to the CLI tools.
- Use the sample configs in `config/resources/` and run the scripts directly.
- Results are logged with the `ConsoleLogger` to stdout; no files are written.

- Inference latency:
  - Run: `python -m benchmarking.bench_infer_latency --config config/resources/bench_infer.toml`
  - Measures warmup and repeated decode steps (tokens/sec, latency). When the config includes a `[data]` section with `np_dat_valid_path`/`total_val_tokens`, the benchmark also samples forward-only batches from the validation memmap and emits a summary `metrics.perplexity` alongside the latency stats. Optional knobs (`perplexity_max_batches`, `perplexity_batch_size`, `perplexity_seed`) can be set under `[benchmark]` to bound evaluation cost and control sampling.

- Tokenizer throughput:
  - Run: `python -m benchmarking.bench_tokenizer --config config/resources/bench_tokenizer.toml`
  - Measures encode and decode throughput over given texts.

## Tests

- Run tests: `uv run pytest`
- Markers:
  - `slow`: long‑running tests. Deselect with `-m "not slow"`.
  - `gpu`: requires CUDA/GPU. Deselect with `-m "not gpu"`.
- Examples:
  - Quick CPU suite: `uv run pytest -m "not slow and not gpu"`
  - Select a file/test: `uv run pytest tests/tokenizer/test_tokenizer.py -q`
  - Filter by name: `uv run pytest -k tokenizer`

## Logging

- Backends:
  - `console` (default): prints structured JSON lines with metrics like `metrics.loss`, `metrics.lr`, `metrics.grad_l2_norm`, plus optional activation/weight norms.
  - `wandb`: logs to Weights & Biases and uploads artifacts (checkpoints, tokenizer files, optional inference outputs).
- Configure in `config/resources/train.toml` under `[logging]`:

```toml
[logging]
backend = "console"   # or "wandb"
run_name = ""         # optional; defaults to timestamp
architecture = "TransformerLM"
dataset = "TinyStoriesV2-GPT4"

# Optional if using Wandb
[wandb]
entity = "your-entity"
project = "your-project"
```

- Inference logs include sampling params and truncated text:
  - Keys: `params.temperature`, `params.p`, `params.eos_token_id`, `text.prompt`, `text.output`, `metrics.latency_ms`.
- Tip (console backend): Pipe to `jq` for readability:
  - `uv run transformerlm-train --config config/resources/train.toml | jq -r "."`

### DDP Policy
- Rank-zero logging: only rank 0 constructs a real logger and emits logs; other ranks are no-ops via `RankZeroLogger`.
- Aggregated metrics: scalar train/val metrics are all-reduced (mean) across ranks before logging.
- Synchronized run name: rank 0 generates the `run_name` and broadcasts it so all ranks agree on the checkpoint directory.
- Checkpoints: only rank 0 writes checkpoints and logs artifacts.
- Optimizer sharding: `OptimizerStateSharding` keeps optimizer state on the owning rank and re-broadcasts parameters after each step.

## Profiling

- Tools:
  - NVTX ranges for GPU timeline annotation (via `profiling/nvtx.py`).
  - CUDA memory helpers for summaries and allocator history (via `profiling/memory.py`).

- NVTX usage:
  - Enable globally with env vars:
    - `TRANSFORMERLM_NVTX=1` to turn on annotations
    - `TRANSFORMERLM_NVTX_LEVEL=coarse|fine|verbose` to control detail
  - Example (collect with Nsight Systems):

```bash
TRANSFORMERLM_NVTX=1 TRANSFORMERLM_NVTX_LEVEL=fine \
  nsys profile -o result \
  uv run transformerlm-train --config config/resources/train.toml
```

- Memory helpers (Python):

```python
from profiling import memory

# Optionally record allocator history (CUDA only)
memory.start_history(True)

# ... run training or inference ...

# Snapshot/summary (safe no-ops on CPU)
ok = memory.dump_snapshot("alloc_history.json")
print(memory.summary())
print(memory.peaks())
memory.reset_peaks()
```
