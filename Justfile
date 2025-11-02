set shell := ["bash", "-euo", "pipefail", "-c"]

prime_host := env_var("PRIME_HOST", "prime-node")
remote_root := env_var("REMOTE_ROOT", "~/diffusionLM")
image_name := env_var("IMAGE_NAME", "diffusionlm")
infer_command_default := env_var("CMD_INFER", "uv run diffusionlm-bench-infer --config config/resources/bench_infer.toml")

bootstrap-remote:
	ssh {{prime_host}} 'bash -s' < scripts/bootstrap_remote.sh

data-remote:
	ssh {{prime_host}} 'bash -s -- {{remote_root}}/data' < scripts/fetch_data.sh

build-remote:
	tag=${IMAGE_TAG:-$(git rev-parse --short HEAD)}
	echo "Building {{image_name}}:${tag} on {{prime_host}}"
	ssh {{prime_host}} "cd {{remote_root}} && docker build -f docker/Dockerfile -t {{image_name}}:${tag} ."

train config="config/resources/train_ddp.toml" extra="":
	tag=${IMAGE_TAG:-$(git rev-parse --short HEAD)}
	TAG="${tag}" CONFIG="{{config}}" EXTRA_ARGS="${extra}" ssh {{prime_host}} 'bash -s' <<'EOS'
	set -euo pipefail
	cd {{remote_root}}
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
	repo_root=$(pwd)
	docker run --rm --gpus all \
	    -v "${repo_root}/data:/opt/diffusionLM/data" \
	    -v "${repo_root}/runs:/opt/diffusionLM/runs" \
	    -e WANDB_API_KEY="${WANDB_API_KEY}" \
	    {{image_name}}:"${TAG}" \
	    diffusionlm-train-ddp --config "${CONFIG}" ${EXTRA_ARGS}
EOS

infer command="{{infer_command_default}}" args="":
	tag=${IMAGE_TAG:-$(git rev-parse --short HEAD)}
	TAG="${tag}" COMMAND="{{command}}" EXTRA_ARGS="${args}" ssh {{prime_host}} 'bash -s' <<'EOS'
	set -euo pipefail
	cd {{remote_root}}
	repo_root=$(pwd)
	docker run --rm --gpus all \
	    -v "${repo_root}/data:/opt/diffusionLM/data" \
	    -v "${repo_root}/runs:/opt/diffusionLM/runs" \
	    {{image_name}}:"${TAG}" \
	    ${COMMAND} ${EXTRA_ARGS}
EOS

nvitop:
	ssh -t {{prime_host}} 'nvitop'

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
