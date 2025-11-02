FROM nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && /root/.local/bin/uv python install 3.11

ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /opt/diffusionLM

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .

RUN mkdir -p /opt/diffusionLM/data /opt/diffusionLM/runs \
 && bash scripts/fetch_data.sh /opt/diffusionLM/data

ENV WANDB_DIR=/opt/diffusionLM/runs \
    DATA_DIR=/opt/diffusionLM/data \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["uv", "run"]
CMD ["diffusionlm-train-ddp", "--config", "config/resources/train_ddp.toml"]
