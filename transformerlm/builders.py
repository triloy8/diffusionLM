from __future__ import annotations

import torch

from transformerlm.models import TransformerLM, TransformerImage, Linear
from transformerlm.models.attention import set_sdp_backend
from transformerlm.tokenizer.tokenizer import Tokenizer
from transformerlm.utils.dtypes import DTYPES


class _DummyTokenizer:
    def encode(self, _text: str) -> list[int]:
        raise NotImplementedError("encode is not available for this pipeline")

    def decode(self, _tokens: list[int]) -> str:
        raise NotImplementedError("decode is not available for this pipeline")


def build_model(cfg) -> torch.nn.Module:
    set_sdp_backend(getattr(cfg, "attention_sdp_backend", "auto"))
    model_type = str(getattr(cfg, "model_type", "lm")).lower()
    if model_type == "image":
        return TransformerImage(
            vocab_size=cfg.vocab_size,
            context_length=cfg.context_length,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            d_ff=cfg.d_ff,
            rope_theta=cfg.rope_theta,
            label_vocab_size=int(getattr(cfg, "label_vocab_size")),
            attention_backend=cfg.attention_backend,
            device=cfg.device,
            dtype=DTYPES[cfg.dtype],
        )
    return TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
        attention_backend=cfg.attention_backend,
        device=cfg.device,
        dtype=DTYPES[cfg.dtype],
    )


def build_tokenizer(args) -> Tokenizer:
    pipeline_mode = str(getattr(args, "pipeline_mode", "")).lower()
    if pipeline_mode == "mnist":
        return _DummyTokenizer()
    if args.tokenizer_vocab_path is None or args.tokenizer_merges_path is None or args.tokenizer_special_tokens_path is None:
        return _DummyTokenizer()
    return Tokenizer.from_files(
        str(args.tokenizer_vocab_path),
        str(args.tokenizer_merges_path),
        str(args.tokenizer_special_tokens_path),
    )


def _is_linear_module(module) -> bool:
    return isinstance(module, Linear)


def build_activation_filter():
    return _is_linear_module
