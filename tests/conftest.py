import random
from pathlib import Path
from typing import Callable, Iterable

import pytest
import numpy as np
import torch

from transformerlm.models import TransformerLM
from transformerlm.training.optim import AdamW
from transformerlm.utils.dtypes import DTYPES

from tests.fixtures import TrainingBundle, build_toy_language_modeling_dataset

from config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig as RuntimeTrainingConfig,
    TrainConfig as RuntimeTrainConfig,
)


@pytest.fixture(scope="session", autouse=True)
def set_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def tiny_dims():
    return {
        "vocab_size": 16,
        "T": 4,
        "d_model": 8,
        "num_heads": 2,
        "num_layers": 1,
        "d_ff": 16,
    }


@pytest.fixture(scope="session")
def toy_training_bundle(device) -> TrainingBundle:
    """Return deterministic dataset, factories, and parsed train config."""
    dataset = build_toy_language_modeling_dataset(device=device, context_length=2)

    model_cfg = ModelConfig(
        vocab_size=dataset.vocab_size,
        context_length=dataset.context_length,
        d_model=16,
        num_layers=2,
        num_heads=2,
        d_ff=32,
        rope_theta=10000.0,
        device=str(device),
        dtype="float32",
    )

    optimizer_cfg = OptimizerConfig(
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        initial_learning_rate=1e-3,
        max_learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_iters=0,
        cosine_cycle_iters=10,
        grad_clip_max_l2_norm=1.0,
    )

    training_cfg = RuntimeTrainingConfig(
        batch_size=2,
        max_train_iteration=4,
        max_val_iteration=1,
        val_freq_iteration=1,
        ckpting_save_iter=10,
        seed=123,
    )

    data_cfg = DataConfig(
        runs_path=Path("./runs"),
        np_dat_train_path=Path("./toy_train.dat"),
        total_train_tokens=int(dataset.train_tokens.numel()),
        np_dat_valid_path=Path("./toy_valid.dat"),
        total_val_tokens=int(dataset.valid_tokens.numel()),
    )

    train_cfg = RuntimeTrainConfig(
        model=model_cfg,
        optimizer=optimizer_cfg,
        training=training_cfg,
        data=data_cfg,
        wandb=None,
        logging=None,
        ddp=None,
    )

    model_dtype = DTYPES[model_cfg.dtype]

    def model_factory() -> TransformerLM:
        torch.manual_seed(0)
        model = TransformerLM(
            vocab_size=model_cfg.vocab_size,
            context_length=model_cfg.context_length,
            d_model=model_cfg.d_model,
            num_layers=model_cfg.num_layers,
            num_heads=model_cfg.num_heads,
            d_ff=model_cfg.d_ff,
            rope_theta=model_cfg.rope_theta,
            device=model_cfg.device,
            dtype=model_dtype,
        )
        return model.to(device)

    def optimizer_factory(parameters: Iterable[torch.nn.Parameter]) -> AdamW:
        return AdamW(
            parameters,
            lr=optimizer_cfg.initial_learning_rate,
            betas=optimizer_cfg.betas,
            eps=float(optimizer_cfg.eps),
            weight_decay=optimizer_cfg.weight_decay,
        )

    return TrainingBundle(
        dataset=dataset,
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        train_config=train_cfg,
    )
