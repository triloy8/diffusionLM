# Checkpointing Module Overview

## Purpose
This package provides a manifest-driven checkpointing system with deterministic per-rank state capture, local/S3 storage, and resumable training.

## Key Concepts
- **Run**: A training session with a unique `run_id`, containing many checkpoint versions.
- **Version**: A snapshot at a specific step, stored under `runs/<run_id>/versions/vXXXXXX/`.
- **Manifest**: JSON metadata describing a version (per-version manifest) and run index (run manifest).
- **Aliases**: `latest` and `best` pointers for convenience.

## Module Layout
- `storage.py`: Local/S3 helpers, hashing, and path/key normalization.
- `state.py`: RNG and batcher state capture/restore.
- `manifest.py`: Manifest/alias creation, persistence, and resolution.
- `manager.py`: High-level orchestration for prepare, resume, and checkpoint callback creation.

## Typical Flow
1. `CheckpointManager` is created with config + run name.
2. `prepare_run()` writes run config snapshots and initializes run manifest.
3. `maybe_resume()` loads model/optimizer/RNG/batcher state if configured.
4. `make_checkpoint_callback()` is passed to the training loop and saves versions.

## Storage Backends
- Local filesystem is always supported.
- Optional S3-compatible storage is supported via `S3ConfigData` and `S3Uploader`.

## Manifest Artifacts
Each checkpoint version contains:
- `model.safetensors`
- `opt_shard_rankXXXX.bin`
- `rng_rankXXXX.json`
- `manifest.json`

Aliases live at `runs/<run_id>/aliases/latest.json` and `runs/<run_id>/aliases/best.json`.

## Resume Semantics
- Resume is driven by a version manifest or alias.
- Exact resume is possible when RNG/batcher state is present for all ranks.
