from __future__ import annotations

import torch
import torch.nn.functional as F

from trainkit.inference.generate import autoregressive_generate, diffusion_generate, semicat_flow_generate
from trainkit.objectives.base import Objective
from trainkit.objectives.data import DiffusionBatch, get_batch
from trainkit.objectives.loss import diffusion_cross_entropy


class SemicatObjective(Objective):
    """Semicat-inspired objective adapted to token-id models."""

    def __init__(self, cfg, tokenizer) -> None:
        super().__init__("semicat")
        self._tokenizer = tokenizer
        self.mask_token_id = int(getattr(cfg, "mask_token_id", cfg.vocab_size - 1))
        self.random_trunc_prob = float(getattr(cfg, "random_trunc_prob", 0.01))
        self.noise_epsilon = float(getattr(cfg, "noise_epsilon", 1e-3))
        self.sd_prop = float(getattr(cfg, "semicat_sd_prop", 0.25))
        self.sd_lambda = float(getattr(cfg, "semicat_sd_lambda", 1.0))
        self.sd_type = str(getattr(cfg, "semicat_sd_type", "lag")).lower()
        self.label_smoothing = float(getattr(cfg, "semicat_label_smoothing", 0.0))
        if not (0.0 <= self.sd_prop <= 1.0):
            raise ValueError("semicat_sd_prop must be in [0, 1]")
        if self.sd_lambda < 0.0:
            raise ValueError("semicat_sd_lambda must be >= 0")
        if self.sd_type not in {"lag"}:
            raise ValueError("semicat_sd_type must be one of: lag")
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError("semicat_label_smoothing must be in [0, 1)")

    def get_batch(self, *, dataset, batch_size: int, context_length: int, device: str, generator=None):
        # Reuse existing batch/dataset plumbing; SemicatObjective builds its own corruption.
        return get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            mask_token_id=self.mask_token_id,
            noise_epsilon=self.noise_epsilon,
            random_trunc_prob=self.random_trunc_prob,
            p_mask_override=0.5,
            deterministic_mask=False,
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
        # Fallback path only; train_loop should call forward_with_model for this objective.
        return diffusion_cross_entropy(
            logits,
            batch.clean_targets,
            batch.mask,
            batch.p_mask,
            loss_mask=batch.loss_mask,
        )

    def _draw_mask(
        self,
        shape: tuple[int, int],
        p: torch.Tensor,
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        rand = torch.rand(shape, device=device, generator=generator)
        return rand < p

    def _corrupt_with_t(
        self,
        clean_targets: torch.Tensor,
        t: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = clean_targets.shape
        mask = self._draw_mask(
            (batch_size, seq_len),
            t.view(-1, 1),
            device=clean_targets.device,
            generator=generator,
        )
        if attention_mask is not None:
            mask = mask & attention_mask
        mask_token_tensor = torch.full_like(clean_targets, fill_value=self.mask_token_id)
        noisy = torch.where(mask, mask_token_tensor, clean_targets)
        return noisy, mask

    def _forward_model(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            if labels is not None:
                return model(inputs, attention_mask=attention_mask, context=labels)
            return model(inputs, attention_mask=attention_mask)
        if labels is not None:
            return model(inputs, context=labels)
        return model(inputs)

    def _weighted_masked_ce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        mask: torch.Tensor,
        p_mask: torch.Tensor,
        loss_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
            label_smoothing=self.label_smoothing,
        ).reshape_as(targets)
        weights = mask.to(losses.dtype) / p_mask
        if loss_mask is not None:
            weights = weights * loss_mask.to(losses.dtype)
            denom = loss_mask.to(losses.dtype).sum()
        else:
            denom = torch.tensor(float(targets.numel()), device=targets.device, dtype=losses.dtype)
        return (losses * weights).sum() / denom.clamp_min(1.0)

    def _sd_lag_loss(
        self,
        model: torch.nn.Module,
        clean_targets: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = clean_targets.shape
        device = clean_targets.device
        t = torch.rand((batch_size,), device=device, generator=generator).clamp_min(1e-4)
        s = torch.rand((batch_size,), device=device, generator=generator) * t

        rand = torch.rand((batch_size, seq_len), device=device, generator=generator)
        xs_mask = rand < s.view(-1, 1)
        xt_mask = rand < t.view(-1, 1)
        if attention_mask is not None:
            xs_mask = xs_mask & attention_mask
            xt_mask = xt_mask & attention_mask

        mask_token_tensor = torch.full_like(clean_targets, fill_value=self.mask_token_id)
        xs = torch.where(xs_mask, mask_token_tensor, clean_targets)
        xt = torch.where(xt_mask, mask_token_tensor, clean_targets)

        student_logits = self._forward_model(
            model,
            xs,
            attention_mask=attention_mask,
            labels=labels,
        )
        with torch.no_grad():
            teacher_logits = self._forward_model(
                model,
                xt,
                attention_mask=attention_mask,
                labels=labels,
            )
            teacher_probs = F.softmax(teacher_logits, dim=-1)

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        per_token = -(teacher_probs * student_log_probs).sum(dim=-1)
        target_mask = xt_mask & (~xs_mask)
        if not bool(target_mask.any().item()):
            target_mask = xt_mask
        if attention_mask is not None:
            target_mask = target_mask & attention_mask
        delta = ((t - s) / (1.0 - s + 1e-8)).view(-1, 1)
        weights = target_mask.to(per_token.dtype) * delta
        denom = weights.sum().clamp_min(1.0)
        return (per_token * weights).sum() / denom

    def forward_with_model(self, model: torch.nn.Module, batch: DiffusionBatch) -> dict:
        clean_targets = batch.clean_targets
        attention_mask = batch.attention_mask
        loss_mask = batch.loss_mask
        labels = getattr(batch, "labels", None)
        generator = None

        batch_size = clean_targets.shape[0]
        sd_split = int(self.sd_prop * batch_size)
        sd_split = max(0, min(sd_split, batch_size))

        vf_start = sd_split
        if vf_start >= batch_size:
            vf_start = 0

        vf_clean = clean_targets[vf_start:]
        vf_attn = attention_mask[vf_start:] if attention_mask is not None else None
        vf_loss_mask = loss_mask[vf_start:] if loss_mask is not None else None
        vf_labels = labels[vf_start:] if labels is not None else None
        vf_batch = vf_clean.shape[0]

        t = torch.rand((vf_batch,), device=vf_clean.device, generator=generator).clamp_min(1e-4)
        vf_inputs, vf_mask = self._corrupt_with_t(
            vf_clean,
            t,
            attention_mask=vf_attn,
            generator=generator,
        )
        vf_logits = self._forward_model(
            model,
            vf_inputs,
            attention_mask=vf_attn,
            labels=vf_labels,
        )
        vf_loss = self._weighted_masked_ce(
            vf_logits,
            vf_clean,
            mask=vf_mask,
            p_mask=t.view(-1, 1),
            loss_mask=vf_loss_mask,
        )

        sd_loss = vf_loss.new_zeros(())
        if sd_split > 0 and self.sd_lambda > 0.0:
            sd_clean = clean_targets[:sd_split]
            sd_attn = attention_mask[:sd_split] if attention_mask is not None else None
            sd_labels = labels[:sd_split] if labels is not None else None
            if self.sd_type == "lag":
                sd_loss = self._sd_lag_loss(
                    model,
                    sd_clean,
                    attention_mask=sd_attn,
                    labels=sd_labels,
                    generator=generator,
                )
            else:  # pragma: no cover - guarded by __init__
                raise ValueError(f"Unsupported semicat sd_type: {self.sd_type}")

        total_loss = vf_loss + self.sd_lambda * sd_loss
        return {
            "loss": total_loss,
            "logits": vf_logits,
            "inputs": vf_inputs,
            "batch": DiffusionBatch(
                noisy_inputs=vf_inputs,
                clean_targets=vf_clean,
                mask=vf_mask,
                p_mask=t.view(-1, 1),
                attention_mask=vf_attn,
                loss_mask=vf_loss_mask,
                metadata=batch.metadata,
                labels=vf_labels,
            ),
            "metrics": {
                "metrics.train_loss_semicat_vf": float(vf_loss.detach().item()),
                "metrics.train_loss_semicat_sd": float(sd_loss.detach().item()),
                "metrics.train_loss_semicat_total": float(total_loss.detach().item()),
            },
        }

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
        if generation_mode == "semicat_flow":
            return semicat_flow_generate(
                model,
                prompt_indices,
                mask_id=int(kwargs.get("mask_id")),
                steps=int(kwargs.get("steps", 0)),
                gen_length=int(kwargs.get("gen_length", 0)),
                temperature=float(kwargs.get("temperature", 0.0)),
                top_p=kwargs.get("top_p"),
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


__all__ = ["SemicatObjective"]
