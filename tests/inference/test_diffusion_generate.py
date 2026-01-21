import torch

from trainkit.inference.generate import diffusion_generate
from trainkit.inference.sampling import compute_transfer_schedule, add_gumbel_noise


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, target_token: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.target_token = target_token

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size, device=input_ids.device)
        logits[..., self.target_token] = 1.0
        return logits


def test_compute_transfer_schedule_basic():
    mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
    schedule = compute_transfer_schedule(mask, steps=2)
    assert schedule.shape == (2, 2)
    assert schedule[0].tolist() == [1, 1]
    assert schedule[1].tolist() == [0, 0]


def test_add_gumbel_noise_temperature_zero():
    logits = torch.randn(2, 3, 5)
    perturbed = add_gumbel_noise(logits, temperature=0.0)
    assert torch.allclose(logits, perturbed)


def test_diffusion_generate_fills_mask_tokens():
    device = torch.device("cpu")
    vocab_size = 6
    context_length = 8
    prompt = torch.tensor([[1, 2]], device=device)
    model = DummyModel(vocab_size=vocab_size, context_length=context_length, target_token=3).to(device)

    output = diffusion_generate(
        model,
        prompt,
        mask_id=5,
        steps=4,
        gen_length=4,
        block_length=2,
        temperature=0.0,
    )

    assert output.shape == (1, 6)
    # Prompt should stay intact
    assert torch.equal(output[:, : prompt.shape[1]], prompt)
    # Newly generated tokens should equal target token from dummy model
    assert torch.all(output[:, prompt.shape[1]:] == 3)
