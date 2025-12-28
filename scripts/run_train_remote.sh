#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC2034
CONFIG="${1:-${CONFIG:-config/resources/train.toml}}"
# shellcheck disable=SC2034
EXTRA_ARGS="${2:-${EXTRA_ARGS:-}}"

export PATH="${HOME}/.local/bin:${PATH}"

if [ ! -f env/wandb.env ]; then
	echo "Missing env/wandb.env; run \`just sync-env\` first" >&2
	exit 1
fi

WANDB_API_KEY=$(
	grep -Em1 '^[[:space:]]*(export[[:space:]]+)?WANDB_API_KEY=' env/wandb.env \
		| sed -E 's/^[[:space:]]*(export[[:space:]]+)?WANDB_API_KEY=//' \
		| tr -d '\r\n'
)

if [ -z "${WANDB_API_KEY:-}" ]; then
	echo "WANDB_API_KEY is empty" >&2
	exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
	echo "tmux not available on remote host" >&2
	exit 1
fi

SESSION="diffusionlm-train"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
	tmux kill-session -t "${SESSION}"
fi

CMD="uv run diffusionlm-train --config \"${CONFIG}\" ${EXTRA_ARGS}"
tmux new -d -s "${SESSION}" "WANDB_API_KEY=${WANDB_API_KEY} ${CMD}"
echo "Started tmux session ${SESSION}"
