import argparse
from pathlib import Path

import torch.multiprocessing as mp

from config import load_train_config
from diffusionlm.training.trainer import train_transformer_ddp
from cli.utils import add_config_args, load_config_or_print


def _parse_only_config():
    parser = argparse.ArgumentParser(description="Training via config file only.", allow_abbrev=False)
    add_config_args(parser, type_=Path)
    return parser.parse_args()


def main():
    # Config-only entry point
    args_cfg = _parse_only_config()
    cfg_dc = load_config_or_print(load_train_config, args_cfg.config, args_cfg.print_config)
    if cfg_dc is None:
        return
    if cfg_dc.ddp is None:
        raise ValueError("DDP config is required when using train_ddp.py")

    # Build an argparse-like namespace expected by existing code
    ns = argparse.Namespace(
        # optimizer
        optimizer_name=cfg_dc.optimizer.optimizer_name,
        betas=(cfg_dc.optimizer.betas[0], cfg_dc.optimizer.betas[1]),
        eps=cfg_dc.optimizer.eps,
        weight_decay=cfg_dc.optimizer.weight_decay,
        initial_learning_rate=cfg_dc.optimizer.initial_learning_rate,
        max_learning_rate=cfg_dc.optimizer.max_learning_rate,
        min_learning_rate=cfg_dc.optimizer.min_learning_rate,
        warmup_iters=cfg_dc.optimizer.warmup_iters,
        cosine_cycle_iters=cfg_dc.optimizer.cosine_cycle_iters,
        grad_clip_max_l2_norm=cfg_dc.optimizer.grad_clip_max_l2_norm,
        # model
        vocab_size=cfg_dc.model.vocab_size,
        context_length=cfg_dc.model.context_length,
        d_model=cfg_dc.model.d_model,
        num_layers=cfg_dc.model.num_layers,
        num_heads=cfg_dc.model.num_heads,
        d_ff=cfg_dc.model.d_ff,
        rope_theta=cfg_dc.model.rope_theta,
        mask_token_id=cfg_dc.model.mask_token_id,
        noise_epsilon=cfg_dc.model.noise_epsilon,
        random_trunc_prob=cfg_dc.model.random_trunc_prob,
        # global
        device=cfg_dc.model.device,
        dtype=cfg_dc.model.dtype,
        max_iteration=None,
        ckpting_save_iter=cfg_dc.training.ckpting_save_iter,
        batch_size=cfg_dc.training.batch_size,
        max_train_iteration=cfg_dc.training.max_train_iteration,
        max_val_iteration=cfg_dc.training.max_val_iteration,
        val_freq_iteration=cfg_dc.training.val_freq_iteration,
        # data/paths
        runs_path=cfg_dc.data.runs_path,
        np_dat_train_path=cfg_dc.data.np_dat_train_path,
        total_train_tokens=cfg_dc.data.total_train_tokens,
        np_dat_valid_path=cfg_dc.data.np_dat_valid_path,
        total_val_tokens=cfg_dc.data.total_val_tokens,
        rng_seed=cfg_dc.training.seed,
        muon_cfg=cfg_dc.optimizer.muon,
        # ddp
        backend=cfg_dc.ddp.backend,
        num_nodes=cfg_dc.ddp.num_nodes,
        num_gpus_per_node=cfg_dc.ddp.num_gpus_per_node,
        node_rank=cfg_dc.ddp.node_rank,
        master_addr=cfg_dc.ddp.master_addr,
        master_port=cfg_dc.ddp.master_port,
        bucket_size_mb=cfg_dc.ddp.bucket_size_mb,
    )

    nprocs = max(1, int(cfg_dc.ddp.num_gpus_per_node))
    mp.spawn(train_transformer_ddp, args=(ns, cfg_dc), nprocs=nprocs, join=True)


if __name__ == "__main__":
    main()
