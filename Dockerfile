FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

ENV UV_SYSTEM_PYTHON=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    openssh-server \
    pkg-config \
    libssl-dev \
  && rm -rf /var/lib/apt/lists/*

RUN conda install -y python=3.11 \
  && conda clean -ya

RUN python -m pip install --no-cache-dir uv

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

RUN mkdir -p /var/run/sshd \
  && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
  && sed -i 's@session\\s*required\\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd \
  && ssh-keygen -A

RUN mkdir -p /root/.ssh \
  && chmod 700 /root/.ssh

COPY pyproject.toml uv.lock ./
COPY . .
