from collections.abc import Callable, Iterable
from typing import Optional
import typing
import numpy.typing as npt
import torch
import math
import random
import os

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs_max = inputs.max(dim=-1, keepdim=True).values
    inputs_stable = inputs - inputs_max
    exp_inputs_stable = torch.exp(inputs_stable)
    log_sum_exp_inputs_stable = torch.log(exp_inputs_stable.sum(dim=-1, keepdim=True))

    indices = targets.long().unsqueeze(-1)
    gathered_inputs_stable = torch.gather(inputs_stable, dim=-1, index=indices)

    l = -gathered_inputs_stable + log_sum_exp_inputs_stable

    return l.mean(dim=0)

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            betas = group["betas"] # Get the betas.
            eps = group["eps"] # Get the eps.
            weight_decay = group["weight_decay"] # Get the weight decay.
            
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get state
                state = self.state[p] # Get state associated with p. 
                m = state.get("m", torch.zeros(p.shape, device=p.grad.device))
                v = state.get("v", torch.zeros(p.shape, device=p.grad.device))
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.

                # Compute weight update
                m = betas[0] * m + (1 - betas[0]) * grad # Compute first moment
                v = betas[1] * v + (1 - betas[1]) * grad**2 # Compute second moment

                lr_t = lr * (math.sqrt(1 - betas[1]**(t+1)) / (1 - betas[0]**(t+1))) # Compute lr_t
                 
                p.data -= lr_t * (m / (torch.sqrt(v) + eps)) # Update weight tensor in-place.
                p.data -= lr * weight_decay * p.data # Apply Weight Decay.
                
                # Update state
                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m # Update first moment
                state["v"] = v # Update second moment

        return loss

def lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        lr = (it / warmup_iters) * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        lr = min_learning_rate + (1/2)*(1 + math.cos(((it - warmup_iters)/(cosine_cycle_iters - warmup_iters))*math.pi)) * (max_learning_rate - min_learning_rate)
    elif cosine_cycle_iters < it:
        lr = min_learning_rate
    else: 
        raise ValueError(f"Invalid learning rate: {lr}")

    return lr

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6
    grads = [p.grad for p in parameters if p.grad is not None]
    
    l2_norm = torch.norm(
        torch.stack([g.detach().norm(2) for g in grads])
    )

    if max_l2_norm < l2_norm:
        for g in grads:
            g.mul_(max_l2_norm / (l2_norm + eps))

def get_batch(dataset: npt.NDArray, batch_size: int, max_sequence_length: int, device: str):
    dataset_len = dataset.shape[0]
    sampled_sequence_stack = []
    for _ in range(batch_size):
        starting_index = random.randint(0, dataset_len - max_sequence_length - 1)
        
        np_sampled_sequence = dataset[starting_index:starting_index+max_sequence_length]
        sampled_sequence =  torch.from_numpy(np_sampled_sequence).to(device)
        sampled_sequence_stack.append(sampled_sequence)
        
    batch_sampled_sequence = torch.stack(sampled_sequence_stack, dim=0)

    return batch_sampled_sequence

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