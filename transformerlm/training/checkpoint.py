import typing
import os
import torch


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    ckpt_dict = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "iteration": iteration,
    }
    torch.save(ckpt_dict, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    ckpt_dict = torch.load(src)
    model.load_state_dict(ckpt_dict["model_state_dict"])
    optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
    iteration = ckpt_dict["iteration"]
    return iteration

__all__ = ["save_checkpoint", "load_checkpoint"]
