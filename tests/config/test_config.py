import json
import textwrap
from copy import deepcopy
from pathlib import Path

import pytest
import tomllib
from pydantic import ValidationError

from config import (
    load_train_config,
    load_infer_config,
    load_make_data_config,
    load_train_tokenizer_config,
    asdict_pretty,
    TrainConfig,
    InferConfig,
    MakeDataConfig,
    TrainTokenizerConfig,
    BenchInferConfig,
    BenchTokenizerConfig,
)

RESOURCE_ROOT = Path("config/resources")


def write(path: Path, content: str):
    path.write_text(textwrap.dedent(content))


def test_train_config_happy_and_validation(tmp_path: Path):
    # Create dummy tokenizer files
    vocab = tmp_path / "vocab.json"
    merges = tmp_path / "merges.txt"
    vocab.write_text("{}")
    merges.write_text("")

    cfg_path = tmp_path / "train.toml"
    write(cfg_path, f"""
    [model]
    vocab_size = 32
    context_length = 128
    d_model = 8
    num_layers = 1
    num_heads = 2
    d_ff = 16
    rope_theta = 10000.0
    device = "cpu"
    dtype = "float32"
    mask_token_id = 31
    noise_epsilon = 0.01
    random_trunc_prob = 0.0

    [optimizer]
    betas = [0.9, 0.95]
    eps = 1e-8
    weight_decay = 0.01
    initial_learning_rate = 0.001
    max_learning_rate = 0.001
    min_learning_rate = 0.0001
    warmup_iters = 10
    cosine_cycle_iters = 100
    grad_clip_max_l2_norm = 1.0

    [training]
    batch_size = 2
    max_train_iteration = 2
    max_val_iteration = 1
    val_freq_iteration = 1
    ckpting_save_iter = 2
    seed = 42

    [data]
    runs_path = "{tmp_path.as_posix()}"
    dataset_name = "example/dataset"
    dataset_config = "default"
    train_split = "train"
    val_split = "validation"
    text_field = "text"
    shuffle_buffer_size = 1000
    shuffle_seed = 123

    [data.tokenizer]
    vocab_path = "{vocab.as_posix()}"
    merges_path = "{merges.as_posix()}"
    special_tokens = ["<|endoftext|>", "<|mask|>"]
    """)

    cfg = load_train_config(cfg_path)
    assert cfg.data.tokenizer.vocab_path.exists()
    assert cfg.data.shuffle_buffer_size == 1000
    assert cfg.optimizer.initial_learning_rate == pytest.approx(0.001)
    assert cfg.training.seed == 42
    dump = cfg.model_dump()
    assert dump["model"]["mask_token_id"] == 31
    assert dump["optimizer"]["initial_learning_rate"] == dump["optimizer"]["max_learning_rate"]
    # pretty dict stringifies paths
    pretty = asdict_pretty(cfg)
    assert isinstance(pretty["data"]["tokenizer"]["vocab_path"], str)

    # Validation error: d_model % num_heads != 0
    bad_cfg = tmp_path / "bad_train.toml"
    write(bad_cfg, cfg_path.read_text().replace("num_heads = 2", "num_heads = 3"))
    with pytest.raises(ValidationError) as exc:
        load_train_config(bad_cfg)
    assert "d_model must be divisible by num_heads" in str(exc.value)


def test_infer_config_happy_and_errors(tmp_path: Path):
    merges = tmp_path / "merges.txt"
    vocab = tmp_path / "vocab.json"
    ckpt = tmp_path / "model.ckpt"
    merges.write_text("")
    vocab.write_text("{}")
    ckpt.write_bytes(b"\0\1")

    cfg_path = tmp_path / "infer.toml"
    write(cfg_path, f"""
    [tokenizer]
    merges_path = "{merges.as_posix()}"
    vocab_path = "{vocab.as_posix()}"
    special_tokens = ["<|eot|>"]

    [model]
    vocab_size = 32
    context_length = 128
    d_model = 8
    num_layers = 1
    num_heads = 2
    d_ff = 16
    rope_theta = 10000.0
    device = "cpu"
    dtype = "float32"

    [checkpoint]
    ckpt_path = "{ckpt.as_posix()}"

    [inference]
    prompt = "hello"
    steps = 32
    total_length = 64
    block_length = 8
    temperature = 1.0
    mask_id = 31
    """)
    cfg = load_infer_config(cfg_path)
    assert cfg.checkpoint.ckpt_path.exists()
    assert cfg.inference.total_length == 64

    # Invalid block_length (must be > 0)
    bad = tmp_path / "infer_bad.toml"
    write(bad, cfg_path.read_text().replace("block_length = 8", "block_length = 0"))
    with pytest.raises(ValidationError) as exc:
        load_infer_config(bad)
    assert "block_length must be > 0" in str(exc.value)

    # Invalid temperature
    bad_t = tmp_path / "infer_bad_t.toml"
    write(bad_t, cfg_path.read_text().replace("temperature = 1.0", "temperature = 0.0"))
    with pytest.raises(ValidationError) as exc:
        load_infer_config(bad_t)
    assert "temperature must be > 0" in str(exc.value)

    # Extra key should be rejected
    bad_extra = tmp_path / "infer_extra.toml"
    write(bad_extra, cfg_path.read_text().replace('[model]\n', '[model]\nunknown = 123\n', 1))
    with pytest.raises(ValidationError) as exc:
        load_infer_config(bad_extra)
    # errors() contains path ('model', 'unknown')
    assert any(err["loc"] == ("model", "unknown") for err in exc.value.errors())


def test_make_data_and_train_tokenizer_loaders(tmp_path: Path):
    merges = tmp_path / "merges.txt"
    vocab = tmp_path / "vocab.json"
    merges.write_text("")
    vocab.write_text("{}")

    # make-data
    input_txt = tmp_path / "input.txt"
    input_txt.write_text("hello")
    out_bin = tmp_path / "out.bin"
    make_cfg = tmp_path / "make.toml"
    write(make_cfg, f"""
    [input]
    input_filename = "{input_txt.as_posix()}"
    total_tokens = 10

    [output]
    output_filename = "{out_bin.as_posix()}"

    [tokenizer]
    merges_path = "{merges.as_posix()}"
    vocab_path = "{vocab.as_posix()}"
    special_tokens = []
    """)
    cfg_mk = load_make_data_config(make_cfg)
    assert cfg_mk.input.input_filename.exists()

    # train-tokenizer
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello")
    tt_cfg = tmp_path / "train_tok.toml"
    write(tt_cfg, f"""
    [input]
    input_path = "{corpus.as_posix()}"
    vocab_size = 32
    special_tokens = ["<|eot|>"]

    [output]
    merges_path = "{merges.as_posix()}"
    vocab_path = "{vocab.as_posix()}"
    """)
    cfg_tt = load_train_tokenizer_config(tt_cfg)
    assert cfg_tt.input.input_path.exists()


def test_optimizer_initial_lr_defaults_to_max(tmp_path: Path):
    vocab = tmp_path / "vocab.json"
    merges = tmp_path / "merges.txt"
    vocab.write_text("{}")
    merges.write_text("")
    cfg_path = tmp_path / "train_defaults.toml"
    write(cfg_path, f"""
    [model]
    vocab_size = 16
    context_length = 8
    d_model = 8
    num_layers = 1
    num_heads = 2
    d_ff = 16
    rope_theta = 10000.0
    device = "cpu"
    dtype = "float32"

    [optimizer]
    eps = 1e-8
    weight_decay = 0.0
    max_learning_rate = 0.01
    min_learning_rate = 0.001
    warmup_iters = 0
    cosine_cycle_iters = 10
    grad_clip_max_l2_norm = 1.0

    [training]
    batch_size = 2
    max_train_iteration = 2
    max_val_iteration = 1
    val_freq_iteration = 1
    ckpting_save_iter = 2

    [data]
    runs_path = "{tmp_path.as_posix()}"
    dataset_name = "example/dataset"
    train_split = "train"
    val_split = "validation"
    text_field = "text"

    [data.tokenizer]
    vocab_path = "{vocab.as_posix()}"
    merges_path = "{merges.as_posix()}"
    """)
    cfg = load_train_config(cfg_path)
    assert cfg.optimizer.initial_learning_rate == pytest.approx(cfg.optimizer.max_learning_rate)


def _write_text(path: Path, content: str = "") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return str(path)


def _write_bytes(path: Path, data: bytes = b"") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def _patch_tokenizer(tbl: dict, tmp_path: Path) -> None:
    tbl["vocab_path"] = _write_text(tmp_path / "vocab.json", "{}")
    tbl["merges_path"] = _write_text(tmp_path / "merges.txt", "")


def _patch_train_like(cfg: dict, tmp_path: Path) -> dict:
    _patch_tokenizer(cfg["data"]["tokenizer"], tmp_path)
    return cfg


def _patch_infer_like(cfg: dict, tmp_path: Path) -> dict:
    _patch_tokenizer(cfg["tokenizer"], tmp_path)
    cfg["checkpoint"]["ckpt_path"] = _write_bytes(tmp_path / "ckpt.bin", b"\0\1")
    return cfg


def _patch_bench_infer(cfg: dict, tmp_path: Path) -> dict:
    return _patch_infer_like(cfg, tmp_path)


def _patch_make_data(cfg: dict, tmp_path: Path) -> dict:
    cfg["input"]["input_filename"] = _write_text(tmp_path / "input.txt", "hello")
    _patch_tokenizer(cfg["tokenizer"], tmp_path)
    return cfg


def _patch_train_tokenizer(cfg: dict, tmp_path: Path) -> dict:
    cfg["input"]["input_path"] = _write_text(tmp_path / "corpus.txt", "hello")
    return cfg


def _patch_bench_tokenizer(cfg: dict, tmp_path: Path) -> dict:
    _patch_tokenizer(cfg["tokenizer"], tmp_path)
    return cfg


RESOURCE_CASES = [
    ("train.toml", TrainConfig, _patch_train_like),
    ("train_ddp.toml", TrainConfig, _patch_train_like),
    ("infer.toml", InferConfig, _patch_infer_like),
    ("bench_infer.toml", BenchInferConfig, _patch_bench_infer),
    ("bench_tokenizer.toml", BenchTokenizerConfig, _patch_bench_tokenizer),
    ("make_data.toml", MakeDataConfig, _patch_make_data),
    ("train_tokenizer.toml", TrainTokenizerConfig, _patch_train_tokenizer),
]


@pytest.mark.parametrize(("filename", "schema", "patcher"), RESOURCE_CASES)
def test_resource_configs_validate(filename: str, schema, patcher, tmp_path: Path):
    raw = tomllib.load((RESOURCE_ROOT / filename).open("rb"))
    patched = patcher(deepcopy(raw), tmp_path)
    cfg = schema.model_validate(patched)
    assert cfg is not None
