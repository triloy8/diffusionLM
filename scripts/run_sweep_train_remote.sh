#!/usr/bin/env bash
set -euo pipefail

# shellcheck disable=SC2034
CONFIG="${1:-${CONFIG:-config/resources/wandb/train_sweep.yaml}}"
# shellcheck disable=SC2034
EXTRA_ARGS="${2:-${EXTRA_ARGS:-}}"

export PATH="${HOME}/.local/bin:${PATH}"

if [ ! -f env/wandb.env ]; then
	echo "Missing env/wandb.env; run \`just sync-env\` first" >&2
	exit 1
fi

read_env_value() {
	local var_name="$1"
	local file_path="$2"
	grep -Em1 "^[[:space:]]*(export[[:space:]]+)?${var_name}=" "$file_path" \
		| sed -E "s/^[[:space:]]*(export[[:space:]]+)?${var_name}=//" \
		| tr -d '\r\n'
}

WANDB_API_KEY=$(
	read_env_value "WANDB_API_KEY" "env/wandb.env"
)

if [ -z "${WANDB_API_KEY:-}" ]; then
	echo "WANDB_API_KEY is empty" >&2
	exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
	echo "tmux not available on remote host" >&2
	exit 1
fi

SWEEP_OUT=$(WANDB_API_KEY=${WANDB_API_KEY} uv run wandb sweep "${CONFIG}" 2>&1)
AGENT_CMD=$(printf "%s\n" "${SWEEP_OUT}" | grep -Eo "wandb agent [^[:space:]]+" | tail -n1)
if [ -z "${AGENT_CMD}" ]; then
	echo "Failed to parse wandb agent command" >&2
	exit 1
fi
if [ -n "${EXTRA_ARGS}" ]; then
	AGENT_CMD="${AGENT_CMD} ${EXTRA_ARGS}"
fi
echo "${AGENT_CMD}"
AGENT_CMD="${AGENT_CMD/wandb/uv run wandb}"

SESSION="transformerlm-sweep-train"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
	tmux kill-session -t "${SESSION}"
fi

tmux new -d -s "${SESSION}" "bash -lc 'export WANDB_API_KEY=${WANDB_API_KEY}; ${AGENT_CMD}'"
echo "Started tmux session ${SESSION}"
