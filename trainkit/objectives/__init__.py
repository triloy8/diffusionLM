from trainkit.objectives.base import Objective
from trainkit.objectives.data import DiffusionBatch, AutoregressiveBatch, get_batch, get_autoregressive_batch
from trainkit.objectives.loss import cross_entropy, diffusion_cross_entropy, mntp_cross_entropy, autoregressive_cross_entropy
from trainkit.objectives.diffusion import DiffusionObjective, MegaDlmDiffusionObjective
from trainkit.objectives.autoregressive import AutoregressiveObjective
from trainkit.objectives.joint import JointDiffusionAutoregressiveObjective, JointMntpAutoregressiveObjective


def build_objective(cfg, tokenizer) -> Objective:
    name = str(getattr(cfg, "training_objective", "diffusion")).lower()
    if name == "ar":
        return AutoregressiveObjective(cfg, tokenizer)
    if name == "megadlm-diffusion":
        return MegaDlmDiffusionObjective(cfg, tokenizer)
    if name == "joint-diffusion-ar":
        return JointDiffusionAutoregressiveObjective(cfg, tokenizer)
    if name == "joint-mntp-ar":
        return JointMntpAutoregressiveObjective(cfg, tokenizer)
    return DiffusionObjective(cfg, tokenizer)


__all__ = [
    "Objective",
    "DiffusionObjective",
    "MegaDlmDiffusionObjective",
    "AutoregressiveObjective",
    "JointDiffusionAutoregressiveObjective",
    "JointMntpAutoregressiveObjective",
    "build_objective",
    "DiffusionBatch",
    "AutoregressiveBatch",
    "get_batch",
    "get_autoregressive_batch",
    "cross_entropy",
    "diffusion_cross_entropy",
    "mntp_cross_entropy",
    "autoregressive_cross_entropy",
]
