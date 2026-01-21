from trainkit.objectives.base import Objective
from trainkit.objectives.data import DiffusionBatch, AutoregressiveBatch, get_batch, get_autoregressive_batch
from trainkit.objectives.loss import cross_entropy, diffusion_cross_entropy, autoregressive_cross_entropy
from trainkit.objectives.diffusion import DiffusionObjective
from trainkit.objectives.autoregressive import AutoregressiveObjective


def build_objective(cfg, tokenizer) -> Objective:
    name = str(getattr(cfg, "training_objective", "diffusion")).lower()
    if name == "ar":
        return AutoregressiveObjective(cfg, tokenizer)
    return DiffusionObjective(cfg, tokenizer)


__all__ = [
    "Objective",
    "DiffusionObjective",
    "AutoregressiveObjective",
    "build_objective",
    "DiffusionBatch",
    "AutoregressiveBatch",
    "get_batch",
    "get_autoregressive_batch",
    "cross_entropy",
    "diffusion_cross_entropy",
    "autoregressive_cross_entropy",
]
