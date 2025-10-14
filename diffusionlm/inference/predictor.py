from diffusionlm.tokenizer.tokenizer import Tokenizer
from diffusionlm.models import TransformerLM
from diffusionlm.inference.generate import diffusion_generate
import torch
import time
from diffusionlm.utils.dtypes import DTYPES
from typing import Optional
from logger import Logger


def infer_transformer(args, *, logger: Optional[Logger] = None, artifact_path: Optional[str] = None):
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path, merges_filepath=args.merges_path, special_tokens=args.special_tokens
    )
    prompt_text = args.prompt
    ids = [tokenizer.encode(prompt_text)]

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=DTYPES[args.dtype],
    )

    ckpt_dict = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt_dict["model_state_dict"])

    in_indices = torch.tensor(ids, device=args.device)

    # Log static sampling parameters once
    if logger is not None:
        logger.log(
            {
                "phase": "infer",
                "params.temperature": float(args.temperature),
                "params.steps": int(args.steps),
                "params.gen_length": int(args.gen_length),
                "params.block_length": int(args.block_length),
                "params.mask_id": int(args.mask_id),
            }
        )

    t0 = time.time()
    out_indices = diffusion_generate(
        model,
        in_indices,
        mask_id=int(args.mask_id),
        steps=int(args.steps),
        gen_length=int(args.gen_length),
        block_length=int(args.block_length),
        temperature=float(args.temperature),
    )
    elapsed = time.time() - t0

    out_indices_list = out_indices[0].tolist()
    output_string = tokenizer.decode(out_indices_list)
    if logger is not None:
        logger.log(
            {
                "phase": "infer",
                "text.prompt": (prompt_text[:200] + ("…" if len(prompt_text) > 200 else "")),
                "text.output": (output_string[:200] + ("…" if len(output_string) > 200 else "")),
                "metrics.prompt_len": int(len(ids[0])),
                "metrics.output_len": int(len(out_indices_list)),
                "metrics.latency_ms": float(elapsed * 1000.0),
            }
        )

    # Optional artifact logging of full predictions
    if logger is not None and artifact_path:
        try:
            import json
            with open(artifact_path, "w", encoding="utf-8") as f:
                rec = {
                    "prompt": prompt_text,
                    "output": output_string,
                    "temperature": args.temperature,
                    "steps": args.steps,
                    "gen_length": args.gen_length,
                    "block_length": args.block_length,
                    "mask_id": args.mask_id,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.log_artifact(artifact_path, name=artifact_path, type_="inference_outputs")
        except Exception:
            pass

    return [output_string]
