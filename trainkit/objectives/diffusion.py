from __future__ import annotations

from typing import Optional
import torch

from trainkit.inference.generate import autoregressive_generate, diffusion_generate
from trainkit.objectives.base import Objective
from trainkit.objectives.data import DiffusionBatch, get_batch, get_megadlm_diffusion_batch
from trainkit.objectives.loss import cross_entropy, diffusion_cross_entropy


class DiffusionObjective(Objective):
    def __init__(self, cfg, tokenizer) -> None:
        super().__init__("diffusion")
        self._tokenizer = tokenizer
        self.mask_token_id = int(getattr(cfg, "mask_token_id", cfg.vocab_size - 1))
        self.noise_epsilon = float(getattr(cfg, "noise_epsilon", 1e-3))
        self.random_trunc_prob = float(getattr(cfg, "random_trunc_prob", 0.01))
        self.p_mask_override = getattr(cfg, "p_mask_override", None)
        self.deterministic_mask = bool(getattr(cfg, "deterministic_mask", False))
        self.p_mask_bucket_edges = getattr(cfg, "p_mask_bucket_edges", None)

    def get_batch(self, *, dataset, batch_size: int, context_length: int, device: str, generator=None):
        return get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            mask_token_id=self.mask_token_id,
            noise_epsilon=self.noise_epsilon,
            random_trunc_prob=self.random_trunc_prob,
            p_mask_override=self.p_mask_override,
            deterministic_mask=self.deterministic_mask,
            generator=generator,
        )

    def model_inputs(self, batch: DiffusionBatch):
        labels = getattr(batch, "labels", None)
        if labels is None:
            return batch.noisy_inputs
        return batch.noisy_inputs, labels

    def attention_mask(self, batch: DiffusionBatch):
        return batch.attention_mask

    def compute_loss(self, logits: torch.Tensor, batch: DiffusionBatch) -> torch.Tensor:
        return diffusion_cross_entropy(
            logits,
            batch.clean_targets,
            batch.mask,
            batch.p_mask,
            loss_mask=batch.loss_mask,
        )

    def extra_metrics(self, logits: torch.Tensor, batch: DiffusionBatch, reduce_metric):
        edges = self.p_mask_bucket_edges or [i / 10.0 for i in range(11)]
        p_mask = batch.p_mask
        mask = batch.mask
        targets = batch.clean_targets
        loss_mask = getattr(batch, "loss_mask", None)
        cleaned = sorted({float(e) for e in edges})
        if len(cleaned) < 2:
            cleaned = [0.0, 1.0]
        with torch.no_grad():
            per_token = cross_entropy(logits, targets, reduction="none")
            mask_f = mask.to(per_token.dtype)
            if loss_mask is not None:
                loss_mask_f = loss_mask.to(per_token.dtype)
                mask_f = mask_f * loss_mask_f
            else:
                loss_mask_f = None
            weighted = (per_token * mask_f) / p_mask
            if loss_mask_f is not None:
                denom = loss_mask_f.sum(dim=1)
            else:
                denom = torch.full(
                    (targets.shape[0],),
                    targets.shape[1],
                    device=per_token.device,
                    dtype=per_token.dtype,
                )
            per_example_loss = weighted.sum(dim=1) / denom.clamp_min(1)
            p_mask_vals = p_mask.view(-1)
            if len(cleaned) > 2:
                boundaries = torch.tensor(cleaned[1:-1], device=p_mask_vals.device, dtype=p_mask_vals.dtype)
                bucket_ids = torch.bucketize(p_mask_vals, boundaries)
            else:
                bucket_ids = torch.zeros_like(p_mask_vals, dtype=torch.long)
            payload = {}
            for i in range(len(cleaned) - 1):
                in_bucket = bucket_ids == i
                count = int(in_bucket.sum().item())
                if count == 0:
                    continue
                mean_val = float(per_example_loss[in_bucket].mean().item())
                if reduce_metric is not None:
                    mean_val = float(reduce_metric(mean_val))
                label = f"{cleaned[i]:.2f}-{cleaned[i + 1]:.2f}"
                payload[f"metrics.p_mask_bucket_loss/{label}"] = mean_val
                payload[f"metrics.p_mask_bucket_count/{label}"] = count
            return payload if payload else None

    def val_samples(self, inputs: torch.Tensor, logits: torch.Tensor, batch: DiffusionBatch, max_samples: int):
        if max_samples <= 0:
            return None
        count = min(int(max_samples), int(inputs.shape[0]))
        if count <= 0:
            return None
        targets = batch.clean_targets
        inputs_list = inputs[:count].detach().cpu().tolist()
        preds_list = logits[:count].argmax(dim=-1).detach().cpu().tolist()
        targets_list = targets[:count].detach().cpu().tolist()
        return [
            {"inputs": inputs_list[i], "predictions": preds_list[i], "targets": targets_list[i]}
            for i in range(count)
        ]

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    def generate(self, model, prompt_indices: torch.Tensor, **kwargs) -> torch.Tensor:
        generation_mode = str(kwargs.get("generation_mode", "diffusion")).lower()
        if generation_mode == "ar":
            return autoregressive_generate(
                model,
                prompt_indices,
                gen_length=int(kwargs.get("gen_length", 0)),
                temperature=float(kwargs.get("temperature", 0.0)),
                top_p=kwargs.get("top_p"),
                eos_token_id=kwargs.get("eos_token_id"),
                logits_eos_inf=bool(kwargs.get("logits_eos_inf", False)),
                generator=kwargs.get("generator"),
            )
        return diffusion_generate(
            model,
            prompt_indices,
            mask_id=int(kwargs.get("mask_id")),
            eos_token_id=kwargs.get("eos_token_id"),
            steps=int(kwargs.get("steps", 0)),
            gen_length=int(kwargs.get("gen_length", 0)),
            block_length=int(kwargs.get("block_length", 0)),
            temperature=float(kwargs.get("temperature", 0.0)),
            top_p=kwargs.get("top_p"),
            cfg_scale=float(kwargs.get("cfg_scale", 0.0)),
            remasking=str(kwargs.get("remasking", "random")),
            logits_eos_inf=bool(kwargs.get("logits_eos_inf", False)),
            confidence_eos_eot_inf=bool(kwargs.get("confidence_eos_eot_inf", False)),
            generator=kwargs.get("generator"),
        )


class MegaDlmDiffusionObjective(Objective):
    def __init__(self, cfg, tokenizer) -> None:
        super().__init__("megadlm-diffusion")
        self._tokenizer = tokenizer
        self.mask_token_id = int(getattr(cfg, "mask_token_id", cfg.vocab_size - 1))
        self.eot_token_id = getattr(cfg, "eot_token_id", None)
        if self.eot_token_id is not None:
            self.eot_token_id = int(self.eot_token_id)
        self.eot_mask_loss = bool(getattr(cfg, "eot_mask_loss", False))
        self.random_trunc_prob = float(getattr(cfg, "random_trunc_prob", 0.01))
        self.p_mask_bucket_edges = getattr(cfg, "p_mask_bucket_edges", None)

    def get_batch(self, *, dataset, batch_size: int, context_length: int, device: str, generator=None):
        return get_megadlm_diffusion_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            mask_token_id=self.mask_token_id,
            eot_token_id=self.eot_token_id,
            eot_mask_loss=self.eot_mask_loss,
            random_trunc_prob=self.random_trunc_prob,
            generator=generator,
        )

    def model_inputs(self, batch: DiffusionBatch) -> torch.Tensor:
        return batch.noisy_inputs

    def attention_mask(self, batch: DiffusionBatch):
        return batch.attention_mask

    def compute_loss(self, logits: torch.Tensor, batch: DiffusionBatch) -> torch.Tensor:
        return diffusion_cross_entropy(
            logits,
            batch.clean_targets,
            batch.mask,
            batch.p_mask,
            loss_mask=batch.loss_mask,
        )

    def extra_metrics(self, logits: torch.Tensor, batch: DiffusionBatch, reduce_metric):
        edges = self.p_mask_bucket_edges or [i / 10.0 for i in range(11)]
        p_mask = batch.p_mask
        mask = batch.mask
        targets = batch.clean_targets
        loss_mask = getattr(batch, "loss_mask", None)
        cleaned = sorted({float(e) for e in edges})
        if len(cleaned) < 2:
            cleaned = [0.0, 1.0]
        with torch.no_grad():
            per_token = cross_entropy(logits, targets, reduction="none")
            mask_f = mask.to(per_token.dtype)
            if loss_mask is not None:
                loss_mask_f = loss_mask.to(per_token.dtype)
                mask_f = mask_f * loss_mask_f
            else:
                loss_mask_f = None
            weighted = (per_token * mask_f) / p_mask
            if loss_mask_f is not None:
                denom = loss_mask_f.sum(dim=1)
            else:
                denom = torch.full(
                    (targets.shape[0],),
                    targets.shape[1],
                    device=per_token.device,
                    dtype=per_token.dtype,
                )
            per_example_loss = weighted.sum(dim=1) / denom.clamp_min(1)
            p_mask_vals = p_mask.view(-1)
            if len(cleaned) > 2:
                boundaries = torch.tensor(cleaned[1:-1], device=p_mask_vals.device, dtype=p_mask_vals.dtype)
                bucket_ids = torch.bucketize(p_mask_vals, boundaries)
            else:
                bucket_ids = torch.zeros_like(p_mask_vals, dtype=torch.long)
            payload = {}
            for i in range(len(cleaned) - 1):
                in_bucket = bucket_ids == i
                count = int(in_bucket.sum().item())
                if count == 0:
                    continue
                mean_val = float(per_example_loss[in_bucket].mean().item())
                if reduce_metric is not None:
                    mean_val = float(reduce_metric(mean_val))
                label = f"{cleaned[i]:.2f}-{cleaned[i + 1]:.2f}"
                payload[f"metrics.p_mask_bucket_loss/{label}"] = mean_val
                payload[f"metrics.p_mask_bucket_count/{label}"] = count
            return payload if payload else None

    def val_samples(self, inputs: torch.Tensor, logits: torch.Tensor, batch: DiffusionBatch, max_samples: int):
        if max_samples <= 0:
            return None
        count = min(int(max_samples), int(inputs.shape[0]))
        if count <= 0:
            return None
        targets = batch.clean_targets
        inputs_list = inputs[:count].detach().cpu().tolist()
        preds_list = logits[:count].argmax(dim=-1).detach().cpu().tolist()
        targets_list = targets[:count].detach().cpu().tolist()
        return [
            {"inputs": inputs_list[i], "predictions": preds_list[i], "targets": targets_list[i]}
            for i in range(count)
        ]

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    def generate(self, model, prompt_indices: torch.Tensor, **kwargs) -> torch.Tensor:
        generation_mode = str(kwargs.get("generation_mode", "diffusion")).lower()
        if generation_mode == "ar":
            return autoregressive_generate(
                model,
                prompt_indices,
                gen_length=int(kwargs.get("gen_length", 0)),
                temperature=float(kwargs.get("temperature", 0.0)),
                top_p=kwargs.get("top_p"),
                eos_token_id=kwargs.get("eos_token_id"),
                logits_eos_inf=bool(kwargs.get("logits_eos_inf", False)),
                generator=kwargs.get("generator"),
            )
        return diffusion_generate(
            model,
            prompt_indices,
            mask_id=int(kwargs.get("mask_id")),
            eos_token_id=kwargs.get("eos_token_id"),
            steps=int(kwargs.get("steps", 0)),
            gen_length=int(kwargs.get("gen_length", 0)),
            block_length=int(kwargs.get("block_length", 0)),
            temperature=float(kwargs.get("temperature", 0.0)),
            top_p=kwargs.get("top_p"),
            cfg_scale=float(kwargs.get("cfg_scale", 0.0)),
            remasking=str(kwargs.get("remasking", "random")),
            logits_eos_inf=bool(kwargs.get("logits_eos_inf", False)),
            confidence_eos_eot_inf=bool(kwargs.get("confidence_eos_eot_inf", False)),
            generator=kwargs.get("generator"),
        )


__all__ = ["DiffusionObjective", "MegaDlmDiffusionObjective"]
