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

if [ -f env/checkpointing.env ]; then
	CHECKPOINTING_S3_BUCKET="$(read_env_value "CHECKPOINTING_S3_BUCKET" "env/checkpointing.env")"
	CHECKPOINTING_S3_PREFIX="$(read_env_value "CHECKPOINTING_S3_PREFIX" "env/checkpointing.env")"
	CHECKPOINTING_S3_ENDPOINT_URL="$(read_env_value "CHECKPOINTING_S3_ENDPOINT_URL" "env/checkpointing.env")"
	CHECKPOINTING_S3_REGION="$(read_env_value "CHECKPOINTING_S3_REGION" "env/checkpointing.env")"
	CHECKPOINTING_S3_ACCESS_KEY_ID="$(read_env_value "CHECKPOINTING_S3_ACCESS_KEY_ID" "env/checkpointing.env")"
	CHECKPOINTING_S3_SECRET_ACCESS_KEY="$(read_env_value "CHECKPOINTING_S3_SECRET_ACCESS_KEY" "env/checkpointing.env")"
	CHECKPOINTING_S3_SESSION_TOKEN="$(read_env_value "CHECKPOINTING_S3_SESSION_TOKEN" "env/checkpointing.env")"
fi

if ! command -v tmux >/dev/null 2>&1; then
	echo "tmux not available on remote host" >&2
	exit 1
fi

SESSION="transformerlm-train"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
	tmux kill-session -t "${SESSION}"
fi

CMD="uv run transformerlm-train --config \"${CONFIG}\" ${EXTRA_ARGS}"
tmux new -d -s "${SESSION}" \
	"WANDB_API_KEY=${WANDB_API_KEY} \
CHECKPOINTING_S3_BUCKET=${CHECKPOINTING_S3_BUCKET:-} \
CHECKPOINTING_S3_PREFIX=${CHECKPOINTING_S3_PREFIX:-} \
CHECKPOINTING_S3_ENDPOINT_URL=${CHECKPOINTING_S3_ENDPOINT_URL:-} \
CHECKPOINTING_S3_REGION=${CHECKPOINTING_S3_REGION:-} \
CHECKPOINTING_S3_ACCESS_KEY_ID=${CHECKPOINTING_S3_ACCESS_KEY_ID:-} \
CHECKPOINTING_S3_SECRET_ACCESS_KEY=${CHECKPOINTING_S3_SECRET_ACCESS_KEY:-} \
CHECKPOINTING_S3_SESSION_TOKEN=${CHECKPOINTING_S3_SESSION_TOKEN:-} \
${CMD}"
echo "Started tmux session ${SESSION}"
