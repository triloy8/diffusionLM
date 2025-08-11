# adapted from https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
from tokenizer import Tokenizer

import torch
from modeling_llada import LLaDAModel
import numpy as np

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

mlp_ratio = 4
d_model = 512
n_heads = 16
residual_dropout = 0.1
attention_dropout = 0.1
rope_theta = 10000.0
max_sequence_length = 1024
vocab_size = 50304
embedding_dropout = 0.1
n_layers = 4
device = "cuda"

model = LLaDAModel(
            mlp_ratio = mlp_ratio,
            d_model = d_model,
            n_heads = n_heads,
            residual_dropout = residual_dropout,
            attention_dropout = attention_dropout,
            rope_theta = rope_theta,
            max_sequence_length = max_sequence_length,
            vocab_size = vocab_size,
            embedding_dropout = embedding_dropout,
            n_layers = n_layers,
            device = device,
        )

ckpt_path = "./runs/2025-08-10_23-02-31_isgighbq/3000.ckpt"
ckpt_dict = torch.load(ckpt_path)

model.load_state_dict(ckpt_dict["model_state_dict"])

vocab_path = "./data/gpt2_vocab.json"
merges_path = "./data/gpt2_merges.txt"
special_tokens = ["<|endoftext|>"]
tokenizer = Tokenizer.from_files(vocab_filepath=vocab_path, merges_filepath=merges_path, special_tokens=special_tokens)

input_ids = [tokenizer.encode(text) for text in ["Oh wow, it's Judy!",]]
input_ids = torch.tensor(input_ids).to(device)

prompt = input_ids
mask_id = 50257
steps=256 
gen_length=256
block_length=128
temperature=0.
cfg_scale=0.
remasking = "random"

x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
x[:, :prompt.shape[1]] = prompt.clone()

prompt_index = (x != mask_id)

assert gen_length % block_length == 0
num_blocks = gen_length // block_length

assert steps % num_blocks == 0
steps = steps // num_blocks

for num_block in range(num_blocks):
    block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
    num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

    for i in range(steps):
        print(f"block: {num_block}")
        print(f"step: {i}")

        mask_index = (x == mask_id)
        if cfg_scale == 0.:
            model.eval()
            with torch.no_grad():
                logits = model(x)

        # gumbel noise for added perplexity
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

        if remasking == 'random':
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

        x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
            transfer_index[j, select_index] = True
        x[transfer_index] = x0[transfer_index]

output_strings = []
for out_indices_ in x: 
    out_indices_list = out_indices_.tolist()

    output_string = tokenizer.decode(out_indices_list)
    output_strings.append(output_string)

print(output_strings)
