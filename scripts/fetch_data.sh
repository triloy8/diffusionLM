#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [destination_dir]" >&2
}

data_dir="${1:-$(pwd)/data}"

mkdir -p "${data_dir}"

download() {
    local url="$1"
    local dest="${data_dir}/$(basename "${url}")"

    if [[ -f "${dest}" ]]; then
        echo "Already present: ${dest}"
        return 0
    fi

    local tmp="${dest}.tmp.$$"
    echo "Downloading ${url}"
    curl -L --fail --retry 3 -o "${tmp}" "${url}"
    mv "${tmp}" "${dest}"
}

download "https://huggingface.co/datasets/trixyL/tiny-story/resolve/main/gpt2_merges.txt"
download "https://huggingface.co/datasets/trixyL/tiny-story/resolve/main/gpt2_vocab.json"
