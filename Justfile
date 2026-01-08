set shell := ["bash", "-euo", "pipefail", "-c"]

prime_host := env_var_or_default("PRIME_HOST", "prime-node")
remote_root := env_var_or_default("REMOTE_ROOT", "~/diffusionLM")
infer_command_default := env_var_or_default("CMD_INFER", "uv run diffusionlm-bench-infer --config config/resources/bench_infer.toml")

bootstrap-remote:
	ssh {{prime_host}} 'bash -s' < scripts/bootstrap_remote.sh

data-remote:
	ssh {{prime_host}} "cd {{remote_root}} && bash -s" < scripts/fetch_data.sh

build-remote:
	ssh {{prime_host}} "cd {{remote_root}} && export PATH=\"\\$HOME/.local/bin:\\$PATH\" && (uv sync --frozen || uv sync)"

train config="config/resources/train.toml" extra="":
	ssh {{prime_host}} "cd {{remote_root}} && bash scripts/run_train_remote.sh $(printf '%q' '{{config}}') $(printf '%q' '{{extra}}')"

sweep-train config="config/resources/wandb/train_sweep.yaml" extra="":
	ssh {{prime_host}} "cd {{remote_root}} && bash scripts/run_sweep_train_remote.sh $(printf '%q' '{{config}}') $(printf '%q' '{{extra}}')"

infer command="{{infer_command_default}}" args="":
	ssh {{prime_host}} "cd {{remote_root}} && bash scripts/run_infer_remote.sh $(printf '%q' '{{command}}') $(printf '%q' '{{args}}')"

nvitop:
	ssh -t {{prime_host}} 'export PATH="$HOME/.local/bin:$PATH"; uvx nvitop'

attach-train:
	ssh -t {{prime_host}} 'tmux attach -t diffusionlm-train'

attach-sweep:
	ssh -t {{prime_host}} 'tmux attach -t diffusionlm-sweep-train'

kill-train:
	ssh {{prime_host}} 'tmux kill-session -t diffusionlm-train 2>/dev/null || true'

kill-sweep:
	ssh {{prime_host}} 'tmux kill-session -t diffusionlm-sweep-train 2>/dev/null || true'

fetch any_file:
	echo "Fetching {{any_file}} from {{prime_host}}"
	scp -r {{prime_host}}:{{remote_root}}/{{any_file}} {{any_file}}

list-runs:
	ssh {{prime_host}} "ls -1 {{remote_root}}/runs"

sync-env:
	if [ ! -f env/wandb.env ]; then echo "Missing env/wandb.env; copy env/wandb.env.example and fill WANDB_API_KEY" >&2; exit 1; fi
	scp env/wandb.env {{prime_host}}:{{remote_root}}/env/wandb.env
	if [ -f env/checkpointing.env ]; then scp env/checkpointing.env {{prime_host}}:{{remote_root}}/env/checkpointing.env; else echo "Skipping env/checkpointing.env (optional)"; fi

auto-train: bootstrap-remote data-remote sync-env train

auto-sweep-train: bootstrap-remote data-remote sync-env sweep-train
