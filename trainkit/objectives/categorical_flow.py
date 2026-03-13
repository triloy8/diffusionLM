from __future__ import annotations

from typing import Optional

import torch

from trainkit.inference.generate import categorical_flow_image_generate
from trainkit.objectives.base import Objective
from trainkit.objectives.data import CategoricalFlowBatch, get_categorical_flow_batch
from trainkit.objectives.loss import cross_entropy


def _masked_mean(values: torch.Tensor, loss_mask: torch.Tensor | None) -> torch.Tensor:
    if loss_mask is None:
        return values.mean()
    weights = loss_mask.to(values.dtype)
    denom = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denom


def _soft_cross_entropy(student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1)


class CategoricalFlowObjective(Objective):
    def __init__(self, cfg, tokenizer) -> None:
        super().__init__("categorical-flow")
        self._tokenizer = tokenizer
        self.vocab_size = int(getattr(cfg, "vocab_size"))
        self.random_trunc_prob = float(getattr(cfg, "random_trunc_prob", 0.0))
        self.null_label_id = getattr(cfg, "null_label_id", None)
        self.uncond_label_dropout_prob = float(getattr(cfg, "uncond_label_dropout_prob", 0.0))
        self.inf_weight = float(getattr(cfg, "categorical_flow_inf_weight", 1.0))
        self.ec_weight = float(getattr(cfg, "categorical_flow_ec_weight", 1.0))
        if self.uncond_label_dropout_prob > 0 and self.null_label_id is None:
            raise ValueError("uncond_label_dropout_prob > 0 requires null_label_id")
        if self.inf_weight < 0:
            raise ValueError("categorical_flow_inf_weight must be >= 0")
        if self.ec_weight < 0:
            raise ValueError("categorical_flow_ec_weight must be >= 0")

    def get_batch(self, *, dataset, batch_size: int, context_length: int, device: str, generator=None):
        batch = get_categorical_flow_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
            vocab_size=self.vocab_size,
            random_trunc_prob=self.random_trunc_prob,
            generator=generator,
        )
        labels = getattr(batch, "labels", None)
        if labels is not None and self.uncond_label_dropout_prob > 0:
            keep = torch.rand(labels.shape, device=labels.device, generator=generator) >= self.uncond_label_dropout_prob
            null_labels = torch.full_like(labels, int(self.null_label_id))
            batch.labels = torch.where(keep, labels, null_labels)
        return batch

    def model_inputs(self, batch: CategoricalFlowBatch):
        return batch.x_s

    def attention_mask(self, batch: CategoricalFlowBatch):
        return None

    def compute_loss(self, logits: torch.Tensor, batch: CategoricalFlowBatch) -> torch.Tensor:
        del logits
        raise RuntimeError("CategoricalFlowObjective requires forward_with_model")

    def forward_with_model(self, model: torch.nn.Module, batch: CategoricalFlowBatch) -> Optional[dict]:
        labels = getattr(batch, "labels", None)
        if labels is None:
            raise ValueError("categorical flow objective requires labels for class conditioning")

        x_s = batch.x_s
        x_t = batch.x_t
        s = batch.s_timesteps
        t = batch.t_timesteps
        targets = batch.clean_targets
        loss_mask = batch.loss_mask

        inf_logits = model(x_t, t, t, context=labels)
        inf_loss = _masked_mean(cross_entropy(inf_logits, targets, reduction="none"), loss_mask)

        student_logits = model(x_s, s, t, context=labels)
        student_probs = torch.softmax(student_logits, dim=-1)
        gamma = ((t - s) / (1.0 - s).clamp_min(1e-6)).unsqueeze(-1)
        transported = x_s + gamma * (student_probs - x_s)
        transported = transported.clamp_min(0.0)
        transported = transported / transported.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        with torch.no_grad():
            teacher_logits = model(transported, t, t, context=labels)
            teacher_probs = torch.softmax(teacher_logits, dim=-1)

        ec_per_token = _soft_cross_entropy(student_logits, teacher_probs)
        ec_loss = _masked_mean(ec_per_token, loss_mask)
        total_loss = self.inf_weight * inf_loss + self.ec_weight * ec_loss

        return {
            "loss": total_loss,
            "logits": inf_logits,
            "inputs": x_t,
            "metrics": {
                "metrics.train_loss_categorical_flow_inf": float(inf_loss.detach().item()),
                "metrics.train_loss_categorical_flow_ec": float(ec_loss.detach().item()),
                "metrics.train_loss_categorical_flow_total": float(total_loss.detach().item()),
            },
        }

    def val_samples(self, inputs: torch.Tensor, logits: torch.Tensor, batch: CategoricalFlowBatch, max_samples: int):
        if max_samples <= 0:
            return None
        count = min(int(max_samples), int(inputs.shape[0]))
        if count <= 0:
            return None
        targets = batch.clean_targets
        preds_list = logits[:count].argmax(dim=-1).detach().cpu().tolist()
        targets_list = targets[:count].detach().cpu().tolist()
        return [{"predictions": preds_list[i], "targets": targets_list[i]} for i in range(count)]

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    def generate(self, model, prompt_indices: torch.Tensor, **kwargs) -> torch.Tensor:
        context = kwargs.get("context")
        if context is None:
            raise ValueError("categorical flow generation requires context labels")
        return categorical_flow_image_generate(
            model,
            prompt_indices,
            context=context,
            steps=int(kwargs.get("steps", 0)),
            temperature=float(kwargs.get("temperature", 0.0)),
            top_p=kwargs.get("top_p"),
            cfg_scale=float(kwargs.get("cfg_scale", 0.0)),
            uncond_context=kwargs.get("uncond_context"),
            generator=kwargs.get("generator"),
        )


__all__ = ["CategoricalFlowObjective"]
