from __future__ import annotations

import torch

from diffusionlm.models import TransformerLM, Linear
from diffusionlm.models.attention import set_sdp_backend
from diffusionlm.tokenizer.tokenizer import Tokenizer
from diffusionlm.utils.dtypes import DTYPES


def build_model(cfg) -> torch.nn.Module:
    set_sdp_backend(getattr(cfg, "attention_sdp_backend", "auto"))
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
    return Tokenizer.from_files(
        str(args.tokenizer_vocab_path),
        str(args.tokenizer_merges_path),
        str(args.tokenizer_special_tokens_path),
    )


def _is_linear_module(module) -> bool:
    return isinstance(module, Linear)


def build_activation_filter():
    return _is_linear_module

