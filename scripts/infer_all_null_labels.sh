#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config/resources/infer_mnist.toml}"

BASE_OUTPUT_DIR="$(
python - "$CONFIG" <<'PY'
import re
import sys
from pathlib import Path

cfg = Path(sys.argv[1])
text = cfg.read_text()
m = re.search(r'^output_dir\s*=\s*"([^"]+)"', text, re.M)
if not m:
    raise SystemExit("output_dir not found in config")
print(m.group(1))
PY
)"

for null_label_id in {0..9}; do
  run_output_dir="${BASE_OUTPUT_DIR}/null_label_${null_label_id}"
  tmp="$(mktemp "/tmp/infer_mnist_null_${null_label_id}.XXXXXX.toml")"
  sed -E \
    -e "s/^null_label_id = .*/null_label_id = ${null_label_id}/" \
    -e "s|^output_dir = .*|output_dir = \"${run_output_dir}\"|" \
    "$CONFIG" > "$tmp"

  echo "Running null_label_id=${null_label_id} output_dir=${run_output_dir}"
  uv run transformerlm-infer-image --config "$tmp"
  rm -f "$tmp"
done
