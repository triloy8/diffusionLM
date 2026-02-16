from types import SimpleNamespace

from trainkit.objectives import build_objective, JointMntpAutoregressiveObjective


class _DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [0]

    def decode(self, tokens: list[int]) -> str:
        return ""


def test_build_objective_joint_mntp_ar():
    cfg = SimpleNamespace(
        training_objective="joint-mntp-ar",
        vocab_size=16,
        mask_token_id=15,
        noise_epsilon=1e-3,
        random_trunc_prob=0.0,
        p_mask_override=None,
        deterministic_mask=False,
        joint_diffusion_alpha=0.3,
        joint_diffusion_alpha_end=None,
        joint_alpha_schedule="constant",
        joint_alpha_schedule_start=0.0,
        joint_alpha_schedule_end=1.0,
        max_train_iteration=10,
        p_mask_schedule="none",
        p_mask_start=None,
        p_mask_end=None,
        p_mask_schedule_start=0.0,
        p_mask_schedule_end=1.0,
    )
    objective = build_objective(cfg, _DummyTokenizer())
    assert isinstance(objective, JointMntpAutoregressiveObjective)
