from __future__ import annotations

import argparse
from typing import List

from cli.utils import add_config_args, load_config_or_print
from config import load_bench_tokenizer_config
from diffusionlm.tokenizer import Tokenizer
from logger import ConsoleLogger

from .common import measure, mean


def _parse_only_config():
    parser = argparse.ArgumentParser(description="Benchmark: tokenizer throughput via config.", allow_abbrev=False)
    add_config_args(parser, type_=str)
    return parser.parse_args()


def main():
    args_cfg = _parse_only_config()
    cfg = load_config_or_print(load_bench_tokenizer_config, args_cfg.config, args_cfg.print_config)
    if cfg is None:
        return

    logger = ConsoleLogger()

    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(cfg.tokenizer.vocab_path),
        merges_filepath=str(cfg.tokenizer.merges_path),
        special_tokens=cfg.tokenizer.special_tokens,
    )
    texts: List[str] = list(cfg.input.text_list)

    run_config = {
        "benchmark": "tokenizer_throughput",
        "texts": len(texts),
        "repeats": cfg.benchmark.repeats,
    }
    logger.start_run(run_config)

    # Precompute token ids once for decode benchmark
    encoded_once = [tokenizer.encode(t) for t in texts]
    token_counts = [len(x) for x in encoded_once]
    total_tokens = int(sum(token_counts))

    # Encode benchmark
    encode_lat_ms: List[float] = []
    encode_tps: List[float] = []
    for r in range(cfg.benchmark.repeats):
        def _encode_all():
            out = []
            for t in texts:
                out.append(tokenizer.encode(t))
            return out

        out, dt = measure("cpu", _encode_all)  # tokenizer is CPU
        toks = int(sum(len(x) for x in out))
        lat_ms = dt * 1000.0
        tps = (float(toks) / dt) if dt > 0 else 0.0
        encode_lat_ms.append(lat_ms)
        encode_tps.append(tps)
        logger.log({
            "phase": "bench_tokenizer",
            "op": "encode",
            "metrics.latency_ms": lat_ms,
            "metrics.tokens_sec": tps,
            "metrics.tokens": toks,
        }, step=r)

    # Decode benchmark
    decode_lat_ms: List[float] = []
    decode_tps: List[float] = []
    for r in range(cfg.benchmark.repeats):
        def _decode_all():
            out = []
            for ids in encoded_once:
                out.append(tokenizer.decode(ids))
            return out

        out, dt = measure("cpu", _decode_all)
        lat_ms = dt * 1000.0
        tps = (float(total_tokens) / dt) if dt > 0 else 0.0
        decode_lat_ms.append(lat_ms)
        decode_tps.append(tps)
        logger.log({
            "phase": "bench_tokenizer",
            "op": "decode",
            "metrics.latency_ms": lat_ms,
            "metrics.tokens_sec": tps,
            "metrics.tokens": total_tokens,
        }, step=r)

    # Summary
    logger.log({
        "phase": "bench_tokenizer",
        "event": "summary",
        "metrics.encode.latency_ms.mean": mean(encode_lat_ms),
        "metrics.encode.tokens_sec.mean": mean(encode_tps),
        "metrics.decode.latency_ms.mean": mean(decode_lat_ms),
        "metrics.decode.tokens_sec.mean": mean(decode_tps),
    })

    logger.finish()


if __name__ == "__main__":
    main()
