import argparse
import json
import os
from pathlib import Path

import torch.multiprocessing as mp

from config import TrainConfig, asdict_pretty, load_train_config
from diffusionlm.training.trainer import train_transformer_ddp
from cli.train import build_train_namespace
from cli.utils import add_config_args, load_config_or_print


def _parse_only_config():
    parser = argparse.ArgumentParser(description="Training with W&B sweep overrides.", allow_abbrev=False)
    add_config_args(parser, type_=Path)
    return parser.parse_args()


def _apply_overrides(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if key.startswith("_") or key in {"wandb_version"} or value is None:
            continue
        if not isinstance(key, str) or not key:
            continue
        parts = key.split(".")
        cursor = base
        for idx, part in enumerate(parts):
            if not isinstance(cursor, dict) or part not in cursor:
                raise ValueError(f"Unknown config path '{key}'")
            if idx == len(parts) - 1:
                cursor[part] = value
            else:
                cursor = cursor[part]
    return base


def main():
    args_cfg = _parse_only_config()
    cfg_dc = load_config_or_print(load_train_config, args_cfg.config, args_cfg.print_config)
    if cfg_dc is None:
        return
    if cfg_dc.ddp is None:
        raise ValueError("DDP config is required when using diffusionlm-sweep-train")

    config_json = os.environ.get("WANDB_CONFIG_JSON")
    if not config_json:
        raise ValueError("WANDB_CONFIG_JSON not set; run via `wandb agent` for sweeps.")
    overrides = json.loads(config_json)
    cfg_dict = _apply_overrides(asdict_pretty(cfg_dc), overrides)
    cfg_dc = TrainConfig.model_validate(cfg_dict)

    ns = build_train_namespace(cfg_dc, str(args_cfg.config))
    nprocs = max(1, int(cfg_dc.ddp.num_gpus_per_node))
    mp.spawn(train_transformer_ddp, args=(ns, cfg_dc), nprocs=nprocs, join=True)


if __name__ == "__main__":
    main()
