#!/usr/bin/env bash
set -euo pipefail

# Sweep cfg_scale, temperature, and labels for image inference.
# Usage:
#   bash infer_sweep_cfg_temp_labels.sh [config_path]
#
# Customize lists below.

CONFIG="${1:-config/resources/infer_mnist.toml}"

CFG_SCALES=(1.0 1.5 2.0 2.5 3.0)
TEMPS=(0.6)
LABELS=({0..9})

if [ ! -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

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

for temp in "${TEMPS[@]}"; do
  for cfg_scale in "${CFG_SCALES[@]}"; do
    for label in "${LABELS[@]}"; do
      run_output_dir="${BASE_OUTPUT_DIR}/temp_${temp}/cfg_${cfg_scale}"
      tmp="$(mktemp "/tmp/infer_mnist_cfg_${cfg_scale}_temp_${temp}_label_${label}.XXXXXX.toml")"
      sed -E \
        -e "s/^cfg_scale = .*/cfg_scale = ${cfg_scale}/" \
        -e "s/^temperature = .*/temperature = ${temp}/" \
        -e "s/^label = .*/label = ${label}/" \
        -e "s|^output_dir = .*|output_dir = \"${run_output_dir}\"|" \
        "$CONFIG" > "$tmp"

      echo "temp=${temp} cfg_scale=${cfg_scale} label=${label}"
      uv run transformerlm-infer-image --config "$tmp"
      rm -f "$tmp"
    done
  done
done
