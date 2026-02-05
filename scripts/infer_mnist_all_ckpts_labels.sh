#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config/resources/infer_mnist.toml}"

if [ ! -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

# Derive versions directory from ckpt_path in config
versions_dir="$(
python - <<'PY'
import re
from pathlib import Path
text = Path("config/resources/infer_mnist.toml").read_text()
# allow CONFIG override via env
import os
cfg = Path(os.environ.get("CONFIG", "config/resources/infer_mnist.toml"))
text = cfg.read_text()
m = re.search(r'^ckpt_path\s*=\s*"([^"]+)"', text, re.M)
if not m:
    raise SystemExit("ckpt_path not found in config")
ckpt = Path(m.group(1))
print(str(ckpt.parent.parent))
PY
)"

if [ ! -d "$versions_dir" ]; then
  echo "Versions dir not found: $versions_dir" >&2
  exit 1
fi

out_base="runs/2026-02-03_23-55-01/infer_images/mnist_all_ckpts"
mkdir -p "$out_base"

for vdir in $(ls -d "$versions_dir"/v* 2>/dev/null | sort); do
  vname="$(basename "$vdir")"
  ckpt_path="$vdir/model.safetensors"
  if [ ! -f "$ckpt_path" ]; then
    echo "Skipping $vname (no model.safetensors)" >&2
    continue
  fi
  out_dir="$out_base/$vname"
  mkdir -p "$out_dir"

  for label in {0..9}; do
    img_path="$out_dir/label_${label}_sample_0.png"
    if [ -f "$img_path" ]; then
      echo "[$vname] label=$label exists, skipping"
      continue
    fi
    tmp="$(mktemp "/tmp/infer_mnist_${vname}_label_${label}.XXXXXX.toml")"
    sed -E \
      -e "s|^ckpt_path = .*|ckpt_path = \"$ckpt_path\"|" \
      -e "s|^label = .*|label = $label|" \
      -e "s|^num_samples = .*|num_samples = 1|" \
      -e "s|^output_dir = .*|output_dir = \"$out_dir\"|" \
      "$CONFIG" > "$tmp"

    echo "[$vname] label=$label"
    uv run transformerlm-infer-image --config "$tmp"
    rm -f "$tmp"
  done

done

# Build GIFs per label across versions
python - <<'PY'
from pathlib import Path
from PIL import Image

out_base = Path("runs/2026-02-03_23-55-01/infer_images/mnist_all_ckpts")
versions = sorted([p for p in out_base.iterdir() if p.is_dir() and p.name.startswith("v")])

for label in range(10):
    frames = []
    missing = []
    for vdir in versions:
        p = vdir / f"label_{label}_sample_0.png"
        if not p.exists():
            missing.append(p)
            continue
        frames.append(Image.open(p).convert("L"))
    if not frames:
        print(f"label {label}: no frames, skipping")
        continue
    gif_path = out_base / f"label_{label}_across_versions.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0,
    )
    if missing:
        print(f"label {label}: wrote {gif_path} (missing {len(missing)} frames)")
    else:
        print(f"label {label}: wrote {gif_path}")
PY

echo "Done. Outputs in $out_base"
