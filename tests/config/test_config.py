import json
from pathlib import Path
import textwrap
import pytest

from config import (
    load_train_config,
    load_infer_config,
    load_make_data_config,
    load_train_tokenizer_config,
    asdict_pretty,
)


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
    # pretty dict stringifies paths
    pretty = asdict_pretty(cfg)
    assert isinstance(pretty["data"]["tokenizer"]["vocab_path"], str)

    # Validation error: d_model % num_heads != 0
    bad_cfg = tmp_path / "bad_train.toml"
    write(bad_cfg, cfg_path.read_text().replace("num_heads = 2", "num_heads = 3"))
    with pytest.raises(ValueError):
        load_train_config(bad_cfg)


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
    with pytest.raises(ValueError):
        load_infer_config(bad)

    # Invalid temperature
    bad_t = tmp_path / "infer_bad_t.toml"
    write(bad_t, cfg_path.read_text().replace("temperature = 1.0", "temperature = 0.0"))
    with pytest.raises(ValueError):
        load_infer_config(bad_t)


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
