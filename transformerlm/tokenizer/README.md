# Tokenizer Toolkit

This directory houses the byte-level GPT-2 tokenizer implementation (pure Python), the optional PyO3-backed Rust tokenizer, and the tooling around them (training, IO helpers, benchmarks).

## Benchmarks

Tokenizer throughput lives under `benchmarking/bench_tokenizer.py`. It reads `config/resources/bench_tokenizer.toml` (or any compatible TOML) and runs encode/decode benchmarks for each available implementation.

```bash
uv run transformerlm-bench-tokenizer --config config/resources/bench_tokenizer.toml
```

- The script always benchmarks the pure-Python tokenizer.
- If the Rust extension is installed, it also benchmarks the Rust implementation and logs both sets of metrics side by side (latency and tokens/sec for encode/decode).
- Results stream to stdout via `ConsoleLogger`.

## Building the Rust Tokenizer

The PyO3 tokenizer lives in `transformerlm/tokenizer_rust/`. To build/install it into your current environment:

```bash
uvx maturin develop
```

This compiles the Rust crate and installs `transformerlm.tokenizer_rust` so Python can import `RustTokenizer`. Re-run the command whenever you change the Rust sources.

## Layout

- `tokenizer.py`: Pure Python GPT-2 byte-pair tokenizer (default implementation).
- `tokenizer_rust/`: Rust implementation + PyO3 bindings.
- `pretokenize.py`: Regex helpers / byte encoders shared by both versions.
- `benchmarking/bench_tokenizer.py`: Shared benchmark entry point.
- `config/resources/bench_tokenizer.toml`: Sample benchmark config.

Use `PythonTokenizer` for streaming workloads (e.g., dataset building). `RustTokenizer` is available for experimentation/benchmarks once the PyO3 module is built.
