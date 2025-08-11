# adapted from https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
from tokenizer import Tokenizer
from modeling_llada import LLaDAModel

import torch
import numpy as np
import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(
        description="Training script hyperparameters and options."
    )

    # ===== MODEL =====
    parser.add_argument("--mlp_ratio", type=int, default=4, help="MLP ratio for transformer feed-forward layers")
    parser.add_argument("--d_model", type=int, default=512, help="Transformer hidden dimension size")
    parser.add_argument("--n_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--residual_dropout", type=float, default=0.1, help="Dropout rate for residual connections")
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="Dropout rate for attention weights")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE base frequency")
    parser.add_argument("--max_sequence_length", type=int, default=1024, help="Maximum sequence length for the model")
    parser.add_argument("--vocab_size", type=int, default=50304, help="Vocabulary size")
    parser.add_argument("--embedding_dropout", type=float, default=0.1, help="Dropout rate for token embeddings")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")

    # ===== MODEL FILES =====
    parser.add_argument("--ckpt_path", type=str, default="./runs/2025-08-10_23-02-31_isgighbq/3000.ckpt", help="Path to model checkpoint")
    parser.add_argument("--vocab_path", type=str, default="./data/gpt2_vocab.json", help="Path to GPT-2 vocabulary JSON")
    parser.add_argument("--merges_path", type=str, default="./data/gpt2_merges.txt", help="Path to GPT-2 merges file")
    parser.add_argument("--special_tokens", type=str, nargs="+", default=["<|endoftext|>"], help="List of special tokens")

    # ===== GENERATION =====
    parser.add_argument("--prompt", type=str, default="Oh wow, it's Judy!", help="Prompt text for generation")
    parser.add_argument("--mask_id", type=int, default=50257, help="Token ID to use as mask")
    parser.add_argument("--steps", type=int, default=256, help="Number of generation steps")
    parser.add_argument("--gen_length", type=int, default=256, help="Total length of generated text")
    parser.add_argument("--block_length", type=int, default=128, help="Block length for generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")

    return parser.parse_args()

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

def generate_llada(args):
    model = LLaDAModel(
                mlp_ratio = args.mlp_ratio,
                d_model = args.d_model,
                n_heads = args.n_heads,
                residual_dropout = args.residual_dropout,
                attention_dropout = args.attention_dropout,
                rope_theta = args.rope_theta,
                max_sequence_length = args.max_sequence_length,
                vocab_size = args.vocab_size,
                embedding_dropout = args.embedding_dropout,
                n_layers = args.n_layers,
                device = args.device,
            )
    
    ckpt_dict = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt_dict["model_state_dict"])

    tokenizer = Tokenizer.from_files(vocab_filepath=args.vocab_path, merges_filepath=args.merges_path, special_tokens=args.special_tokens)

    input_ids = tokenizer.encode(args.prompt)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(torch.device(args.device))

    x = torch.full((1, input_ids.shape[1] + args.gen_length), args.mask_id, dtype=torch.long).to(model.device)
    x[:, :input_ids.shape[1]] = input_ids.clone()

    assert args.gen_length % args.block_length == 0
    num_blocks = args.gen_length // args.block_length

    assert args.steps % num_blocks == 0
    steps = args.steps // num_blocks

    for num_block in tqdm(range(num_blocks), desc="Blocks", unit="block", position=0):
        block_mask_index = (x[:, input_ids.shape[1] + num_block * args.block_length: input_ids.shape[1] + (num_block + 1) * args.block_length:] == args.mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in tqdm(range(steps), desc="Steps", unit="step", position=1, leave=False):

            mask_index = (x == args.mask_id)
            # cfg scale = 0
            model.eval()
            with torch.no_grad():
                logits = model(x)

            # gumbel noise for added perplexity
            logits_with_noise = add_gumbel_noise(logits, temperature=args.temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            # random remasking
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

            x0_p[:, input_ids.shape[1] + (num_block + 1) * args.block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    output_string = tokenizer.decode(x[0].tolist())

    return output_string

if __name__ == "__main__":
    args = get_args()
    output_string = generate_llada(args)
    print(output_string)

