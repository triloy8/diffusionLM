#!/usr/bin/env bash
set -euo pipefail

err() {
    echo "bootstrap_remote: $*" >&2
}

if ! command -v docker >/dev/null 2>&1; then
    err "Docker is required on prime-node"
    exit 1
fi

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

apt_updated=false
apt_install() {
    if [ "${apt_updated}" = false ]; then
        ${SUDO} apt-get update
        apt_updated=true
    fi
    ${SUDO} apt-get install -y "$@"
}

if ! command -v curl >/dev/null 2>&1; then
    apt_install curl
fi

if ! command -v git >/dev/null 2>&1; then
    apt_install git
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

if ! command -v just >/dev/null 2>&1; then
    curl -fsSL https://just.systems/install.sh | ${SUDO} bash -s -- --to /usr/local/bin
fi

if ! command -v nvitop >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
        uv tool install nvitop
    else
        python3 -m pip install --user nvitop
    fi
fi

repo_root="${HOME}/diffusionLM"

if [ ! -d "${repo_root}/.git" ]; then
    if [ -d "${repo_root}" ] && [ "$(ls -A "${repo_root}")" ]; then
        err "${repo_root} exists but is not a git repo; aborting clone"
        exit 1
    fi
    rm -rf "${repo_root}"
    git clone -b feat/deploy https://github.com/triloy8/diffusionLM.git "${repo_root}"
fi

mkdir -p "${repo_root}/data" "${repo_root}/runs" "${repo_root}/env"

train_env="${repo_root}/env/train.env"
if [ ! -f "${train_env}" ]; then
    cat > "${train_env}" <<'EOF'
# Reserved for optional non-secret environment variables.
EOF
fi

echo "Bootstrap complete: directories ready and tooling present."
