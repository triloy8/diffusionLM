set shell := ["bash", "-euo", "pipefail", "-c"]

prime_host := env_var_or_default("PRIME_HOST", "prime-node")
remote_root := env_var_or_default("REMOTE_ROOT", "~/diffusionLM")
infer_command_default := env_var_or_default("CMD_INFER", "uv run diffusionlm-bench-infer --config config/resources/bench_infer.toml")

bootstrap-remote:
	ssh {{prime_host}} 'bash -s' < scripts/bootstrap_remote.sh

data-remote:
	ssh {{prime_host}} 'bash -s -- {{remote_root}}/data' < scripts/fetch_data.sh

build-remote:
	ssh {{prime_host}} "cd {{remote_root}} && export PATH=\"\\$HOME/.local/bin:\\$PATH\" && (uv sync --frozen || uv sync)"

train config="config/resources/train_ddp.toml" extra="":
	CONFIG="{{config}}" EXTRA_ARGS="{{extra}}" ssh {{prime_host}} 'bash -s' <<'EOS'
	set -euo pipefail
	cd {{remote_root}}
	export PATH="${HOME}/.local/bin:${PATH}"
	if [ ! -f env/wandb.env ]; then
		echo "Missing env/wandb.env; run `just sync-env` first" >&2
		exit 1
	fi
	set -a
	. env/wandb.env
	set +a
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
	CMD="uv run diffusionlm-train-ddp --config \"${CONFIG}\" ${EXTRA_ARGS}"
	tmux new -d -s "${SESSION}" "${CMD}"
	echo "Started tmux session ${SESSION}"
EOS

infer command="{{infer_command_default}}" args="":
	COMMAND="{{command}}" EXTRA_ARGS="{{args}}" ssh {{prime_host}} 'bash -s' <<'EOS'
	set -euo pipefail
	cd {{remote_root}}
	export PATH="${HOME}/.local/bin:${PATH}"
	${COMMAND} ${EXTRA_ARGS}
EOS

nvitop:
	ssh -t {{prime_host}} 'export PATH="$HOME/.local/bin:$PATH"; nvitop'

attach-train:
	ssh -t {{prime_host}} 'tmux attach -t diffusionlm-train'

kill-train:
	ssh {{prime_host}} 'tmux kill-session -t diffusionlm-train 2>/dev/null || true'

fetch run_dir:
	mkdir -p runs
	echo "Fetching run directory ${run_dir} from {{prime_host}}"
	scp -r {{prime_host}}:{{remote_root}}/runs/${run_dir} runs/

list-runs:
	ssh {{prime_host}} "ls -1 {{remote_root}}/runs"

sync-env:
	if [ ! -f env/wandb.env ]; then
	echo "Missing env/wandb.env; copy env/wandb.env.example and fill WANDB_API_KEY" >&2
	exit 1
	fi
	scp env/wandb.env {{prime_host}}:{{remote_root}}/env/wandb.env
