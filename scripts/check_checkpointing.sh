#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-}"
RUNS_PATH="${2:-./runs}"
VERSION_ID="${3:-}"

if [[ -z "$RUN_ID" ]]; then
  echo "usage: $0 <run_id> [runs_path] [version_id]" >&2
  exit 2
fi

RUN_DIR="$RUNS_PATH/$RUN_ID"
MANIFEST="$RUN_DIR/manifest.json"
LATEST_ALIAS="$RUN_DIR/aliases/latest.json"
BEST_ALIAS="$RUN_DIR/aliases/best.json"

if [[ ! -f "$MANIFEST" ]]; then
  echo "missing run manifest: $MANIFEST" >&2
  exit 1
fi
if [[ ! -f "$LATEST_ALIAS" ]]; then
  echo "missing latest alias: $LATEST_ALIAS" >&2
  exit 1
fi
if [[ ! -f "$BEST_ALIAS" ]]; then
  echo "missing best alias: $BEST_ALIAS" >&2
  exit 1
fi

python - <<'PY' "$MANIFEST" "$LATEST_ALIAS" "$BEST_ALIAS" "$VERSION_ID"
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
latest_path = Path(sys.argv[2])
best_path = Path(sys.argv[3])
version_id = sys.argv[4] or None
run_dir = manifest_path.parent
root_parent = run_dir.parent

manifest = json.loads(manifest_path.read_text())
latest = json.loads(latest_path.read_text())
best = json.loads(best_path.read_text())

for key in ("schema_version", "run_id", "paths", "config", "versions"):
    if key not in manifest:
        raise SystemExit(f"run manifest missing key: {key}")

if latest.get("alias") != "latest":
    raise SystemExit("latest alias missing or invalid")
if best.get("alias") != "best":
    raise SystemExit("best alias missing or invalid")

if not version_id:
    version_id = latest.get("version_id")

if not version_id:
    raise SystemExit("no version_id resolved from latest alias")

version_dir = manifest_path.parent / "versions" / version_id
version_manifest = version_dir / "manifest.json"

if not version_manifest.exists():
    raise SystemExit(f"missing version manifest: {version_manifest}")

v = json.loads(version_manifest.read_text())
for key in ("model", "optimizer", "rng", "resume"):
    if key not in v:
        raise SystemExit(f"version manifest missing key: {key}")

def _resolve(key: str) -> Path:
    return root_parent / key

model_key = v["model"]["key"]
model_path = _resolve(model_key)
if not model_path.exists():
    raise SystemExit(f"missing model artifact: {model_path}")

for shard in v["optimizer"]["shards"]:
    shard_key = shard.get("key")
    if not shard_key:
        raise SystemExit("optimizer shard missing key")
    shard_path = _resolve(shard_key)
    if not shard_path.exists():
        raise SystemExit(f"missing optimizer shard: {shard_path}")

for entry in v["rng"]["keys"]:
    rng_key = entry.get("key")
    if not rng_key:
        raise SystemExit("rng entry missing key")
    rng_path = _resolve(rng_key)
    if not rng_path.exists():
        raise SystemExit(f"missing rng artifact: {rng_path}")

print(f"ok: run_id={manifest['run_id']} version_id={version_id}")
PY

echo "checkpointing looks sane for run_id=$RUN_ID"
