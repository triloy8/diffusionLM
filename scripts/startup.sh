#!/usr/bin/env bash
set -euo pipefail

mkdir -p /root/.ssh
chmod 700 /root/.ssh

if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

if [ -n "${SSH_PORT:-}" ]; then
    sed -i '/^#*Port /d' /etc/ssh/sshd_config
    echo "Port ${SSH_PORT}" >> /etc/ssh/sshd_config
fi

echo "Starting SSH server on port ${SSH_PORT:-22}"
/usr/sbin/sshd

if [ -x /app/scripts/fetch_data.sh ]; then
    /app/scripts/fetch_data.sh /app/data
else
    bash /app/scripts/fetch_data.sh /app/data
fi

exec uv run diffusionlm-train --config config/resources/train.toml
