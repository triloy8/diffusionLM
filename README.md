# Diffusion Language Model

## What Is This?

This repo started as a from‑scratch diffusion language model inspired by the LLaDA guidelines and has since evolved into two pieces:
- `diffusionlm`: the model, tokenizer, and inference utilities.
- `trainkit`: a generic training stack (loop, DDP, checkpointing, logging, streaming).

See `trainkit/README.md` for training‑specific details.

## Overview

- From‑scratch model: decoder‑only Transformer LM (RMSNorm, SwiGLU, RoPE, SDPA/MHA), implemented directly with PyTorch modules.
- From‑scratch tokenizer: byte‑level BPE training and IO, producing `vocab.json` and `merges.txt`.
- Diffusion inference: bidirectional decoding with configurable mask scheduling (steps, block length, temperature).
- Training stack lives in `trainkit` (see `trainkit/README.md`).
- CLI + TOML configs: consistent entry points built around config schemas.
- Benchmarking + profiling: tokenizer/inference throughput checks and memory/runtime inspection.

## Installation

Requires Python 3.11–3.12 and PyTorch. Using [`uv`](https://github.com/astral-sh/uv) is recommended.

- Quick run without installing the package:

```bash
# Print an example resolved config
uv run diffusionlm-train --config config/resources/train.toml --print-config
```

## Usage Examples

Entry points live in `cli/` and are driven by TOML configs in `config/resources/`.

- Train the tokenizer (BPE):

```bash
uv run diffusionlm-train-tokenizer --config config/resources/train_tokenizer.toml
```

- Start model training (DDP works for single GPU too):

```bash
uv run diffusionlm-train --config config/resources/train.toml
```

Notes:
- See `trainkit/README.md` for training/runtime behavior (DDP backend, logging, optimizer options, validation, accumulation).

- Generate text:

```bash
uv run diffusionlm-infer --config config/resources/infer.toml
```

_Note:_ `inference.total_length` should be the final sequence length (prompt + generated tokens) and must not exceed `model.context_length`; the generated span is computed automatically as `total_length - prompt_tokens`.

- Inspect effective configuration without running:

```bash
uv run diffusionlm-train --config config/resources/train.toml --print-config
```
> Config loaders use Pydantic validation, so missing/extra keys or bad values raise `pydantic.ValidationError` with detailed paths.

## Datasets & Tokenizer Assets

Training now streams data directly from Hugging Face Datasets. To prepare your environment:

1. **Grab tokenizer files** (if you don’t already have them). Run `scripts/fetch_data.sh <output_dir>` to download `gpt2_vocab.json` and `gpt2_merges.txt` into `data/` (or another path you supply). Point `[data.tokenizer]` in your configs to those files.
2. **Choose a dataset** available via `datasets.load_dataset`. Configure `[data]` with `dataset_name`, optional `dataset_config`, `train_split`, `val_split`, and the text field to tokenize. Streaming supports both public Hub datasets and local files (e.g., `load_dataset("json", data_files=...)`).
3. **Cache for offline runs**: set `HF_DATASETS_CACHE=/path/to/cache` and run training/benchmarking once with network access (or use `huggingface-cli download ...`). After the cache is populated, set `HF_DATASETS_OFFLINE=1` to force offline mode. Private datasets require `huggingface-cli login` or an `HF_TOKEN` in the environment.
4. **Shuffling**: `[data].shuffle_buffer_size` controls the streaming shuffle window; bump it up (e.g., 10_000) for better randomization, or leave at 0 to read the dataset order as-is. `shuffle_seed` seeds the buffer RNG; DDP adds the rank to keep shards deterministic and independent.
5. **Row boundaries**: each streamed row is tokenized, appended with `eot_token_id`, and packed into fixed-length `context_length` blocks with rollover (no padding).
6. **Pipeline mode**: set `[data].pipeline_mode = "packed"` (default) to concatenate rows, or `"rows"` to keep each row as its own sequence with padding. Row mode requires `[data].pad_token_id` and applies an attention mask so padded tokens are ignored.

## Remote Orchestration

The `Justfile` plus helper scripts under `scripts/` provide a thin remote control plane focused on Prime Intellect hosts; set `PRIME_HOST`/`REMOTE_ROOT` to point at the target machine and path. All available recipes:
- `just bootstrap-remote` runs `scripts/bootstrap_remote.sh` over SSH to install uv/just/tmux, clone the repo, and prepare `data/`, `runs/`, and `env/` directories on the remote.
- `just data-remote` executes `scripts/fetch_data.sh` remotely to download the GPT‑2 tokenizer vocab/merges into the remote `data/` directory (streaming datasets are fetched via Hugging Face at runtime).
- `just build-remote` syncs dependencies by running `uv sync` (with a frozen lockfile fallback) inside the remote checkout.
- `just train config=<toml> extra="<args>"` launches `scripts/run_train_remote.sh` inside a tmux session (`diffusionlm-train`) after validating Weights & Biases credentials.
- `just infer command="<cmd>" args="<extra>"` calls `scripts/run_infer_remote.sh` to execute arbitrary inference or benchmarking commands; defaults derive from `CMD_INFER`.
- `just nvitop` opens `nvitop` on the remote box for lightweight GPU monitoring (assumes it is installed by the bootstrap step).
- `just attach-train` attaches your terminal to the `diffusionlm-train` tmux session, while `just kill-train` tears it down if you need to restart.
- `just fetch any_file=<path>` pulls any file or directory from `${REMOTE_ROOT}/<path>` into the current working directory; `just list-runs` prints the remote run directory names.
- `just sync-env` uploads your local `env/wandb.env` (copy `env/wandb.env.example` and populate `WANDB_API_KEY`) so the remote training session can authenticate with W&B.
- `just auto-train` runs the full bootstrap + data + env sync + training workflow (`bootstrap-remote`, `data-remote`, `sync-env`, `train`) in one shot.

## Modules

- diffusionlm.models
  - Purpose: Core Transformer components with bidirectional self-attention for diffusion-style language modelling.
  - Key files: `diffusionlm/models/transformer.py`, `diffusionlm/models/attention.py`, `diffusionlm/models/layers.py`.
    (Attention is unmasked/bidirectional by default.)
  - Notes: dtype helpers under `diffusionlm/utils/dtypes.py`.

- diffusionlm.inference
  - Purpose: Sampling utilities and simple generation helpers.
  - Key files: `diffusionlm/inference/generate.py`, `diffusionlm/inference/sampling.py`, `diffusionlm/inference/predictor.py`.
  - Notes: Inference configs provide `prompt`, `steps`, `total_length`, `block_length`, `temperature`, and `mask_id` for diffusion decoding.

- diffusionlm.tokenizer
  - Purpose: From‑scratch byte‑level BPE trainer and tokenizer IO.
  - Key files: `diffusionlm/tokenizer/bpe_trainer.py`, `diffusionlm/tokenizer/tokenizer.py`, `diffusionlm/tokenizer/pretokenize.py`, `diffusionlm/tokenizer/io.py`.
  - Artifacts: `vocab.json`, `merges.txt` (with optional special tokens).


- cli
  - Purpose: Command‑line entry points wrapping configs and orchestration.
  - Key files: `cli/train.py`, `cli/infer.py`, `cli/train_tokenizer.py`, `cli/utils.py`.
  - Scripts: exposed in `pyproject.toml` under `[project.scripts]`.

- benchmarking
  - Purpose: Quick throughput checks for inference and tokenizer.
  - Key files: `benchmarking/bench_infer_latency.py`.
  - Configs: `config/resources/bench_infer.toml`.

- config
  - Purpose: Typed config schemas, loaders, validation, and example TOMLs.
  - Key files: `config/train.py`, `config/infer.py`, `config/bench_infer.py`, `config/bench_tokenizer.py`, `config/io.py`, `config/schemas.py`.
  - Examples: `config/resources/*.toml`.
  - Validation: Schemas are implemented with Pydantic v2 (`extra="forbid"`), so malformed or misspelled fields raise `pydantic.ValidationError` with structured error paths; CLI `--print-config` output comes from `model_dump()` with file paths stringified.

- profiling
  - Purpose: Lightweight helpers for memory/runtime profiling, including NVTX ranges.
  - Key files: `profiling/memory.py`, `profiling/nvtx.py`.

- utils
  - Purpose: Small shared helpers.
  - Key files: `diffusionlm/utils/dtypes.py`.

## Benchmarking

- Benchmarks live under `benchmarking/` and are TOML‑driven, similar to the CLI tools.
- Use the sample configs in `config/resources/` and run the scripts directly.
- Results are logged with the `ConsoleLogger` to stdout; no files are written.

- Inference latency:
- Run: `uv run diffusionlm-bench-infer --config config/resources/bench_infer.toml`
  - Measures warmup and repeated diffusion reverse passes, logging latency, tokens/sec, and diffusion-specific metrics (steps, block length, average mask ratio). When the config includes a `[data]` section with streaming fields (`dataset_name`, `split`, `text_field`, optional shuffle settings), the benchmark can optionally stream validation tokens from Hugging Face Datasets for backward/perplexity summaries (`perplexity_*` knobs under `[benchmark]`).


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

Training logging details live in `trainkit/README.md`. Inference logs include sampling params
and truncated text (keys: `params.temperature`, `params.p`, `params.eos_token_id`,
`text.prompt`, `text.output`, `metrics.latency_ms`).

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
  uv run diffusionlm-train --config config/resources/train.toml
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
