#!/usr/bin/env bash
set -euo pipefail

err() {
    echo "bootstrap_remote: $*" >&2
}

if ! command -v docker >/dev/null 2>&1; then
    err "Docker is required on prime-node"
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    err "Unable to communicate with Docker daemon"
    exit 1
fi

if ! docker info 2>/dev/null | grep -qi "nvidia"; then
    err "NVIDIA Docker runtime missing; install nvidia-container-toolkit"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if ! command -v nvitop >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
        uv tool install nvitop
    else
        python3 -m pip install --user nvitop
    fi
fi

repo_root="${HOME}/diffusionLM"
mkdir -p "${repo_root}/data" "${repo_root}/runs" "${repo_root}/env"

train_env="${repo_root}/env/train.env"
if [ ! -f "${train_env}" ]; then
    cat > "${train_env}" <<'EOF'
# Reserved for optional non-secret environment variables.
EOF
fi

echo "Bootstrap complete: directories ready and tooling present."
