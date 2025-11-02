#!/usr/bin/env bash
set -euo pipefail

COMMAND_INPUT="${1:-${COMMAND:-}}"
ARGS_INPUT="${2:-${EXTRA_ARGS:-}}"

if [ -z "${COMMAND_INPUT}" ]; then
	echo "infer: command is empty; provide one via argument or COMMAND env var" >&2
	exit 1
fi

export PATH="${HOME}/.local/bin:${PATH}"

CMD_STRING="${COMMAND_INPUT}"
if [ -n "${ARGS_INPUT}" ]; then
	CMD_STRING="${CMD_STRING} ${ARGS_INPUT}"
fi

eval "${CMD_STRING}"
