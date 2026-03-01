from __future__ import annotations

from types import SimpleNamespace

import torch

from trainkit.objectives.semicat import SemicatObjective


class _DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [0]

    def decode(self, tokens: list[int]) -> str:
        return ""


class _DummyDataset:
    def __init__(self, vocab_size: int, label_vocab_size: int, device: torch.device):
        self.vocab_size = vocab_size
        self.label_vocab_size = label_vocab_size
        self.device = device

    def draw(self, batch_size: int, context_length: int):
        tokens = torch.randint(0, self.vocab_size - 1, (batch_size, context_length), device=self.device)
        labels = torch.randint(0, self.label_vocab_size, (batch_size,), device=self.device)
        return SimpleNamespace(tokens=tokens, labels=labels)


class _TinyImageModel(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int, label_vocab_size: int):
        super().__init__()
        self.context_length = 32
        self.token_embed = torch.nn.Embedding(vocab_size, d_model)
        self.label_embed = torch.nn.Embedding(label_vocab_size, d_model)
        self.proj = torch.nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_embed(input_ids)
        if context is not None:
            x = x + self.label_embed(context).unsqueeze(1)
        return self.proj(x)


def test_semicat_objective_forward_backward_cpu():
    device = torch.device("cpu")
    cfg = SimpleNamespace(
        vocab_size=33,
        mask_token_id=32,
        noise_epsilon=1e-3,
        random_trunc_prob=0.0,
        semicat_sd_prop=0.5,
        semicat_sd_lambda=0.5,
        semicat_sd_type="lag",
        semicat_label_smoothing=0.0,
    )
    objective = SemicatObjective(cfg, _DummyTokenizer())
    dataset = _DummyDataset(vocab_size=33, label_vocab_size=11, device=device)
    model = _TinyImageModel(vocab_size=33, d_model=32, label_vocab_size=11).to(device)

    batch = objective.get_batch(
        dataset=dataset,
        batch_size=8,
        context_length=16,
        device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(123),
    )
    out = objective.forward_with_model(model, batch)
    loss = out["loss"]
    assert torch.isfinite(loss).item()
    loss.backward()
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += float(p.grad.norm().item())
    assert grad_norm > 0.0
