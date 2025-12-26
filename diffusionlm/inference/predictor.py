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
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens_path=args.special_tokens_path,
    )
    prompt_text = args.prompt
    ids = [tokenizer.encode(prompt_text)]
    prompt_len = len(ids[0])
    total_length = int(args.total_length)
    if total_length < prompt_len:
        raise ValueError("total_length must be >= prompt token length")
    gen_length = total_length - prompt_len
    if total_length > args.context_length:
        raise ValueError("total_length must be <= model context_length")

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
    eos_token_id = getattr(args, "eos_token_id", None)
    cfg_scale = float(getattr(args, "cfg_scale", 0.0))
    remasking = getattr(args, "remasking", "random")
    logits_eos_inf = bool(getattr(args, "logits_eos_inf", False))
    confidence_eos_eot_inf = bool(getattr(args, "confidence_eos_eot_inf", False))
    seed = getattr(args, "seed", None)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=args.device)
        generator.manual_seed(int(seed))

    # Log static sampling parameters once
    if logger is not None:
        logger.log(
            {
                "phase": "infer",
                "params.temperature": float(args.temperature),
                "params.steps": int(args.steps),
                "params.total_length": int(args.total_length),
                "params.gen_length": int(gen_length),
                "params.block_length": int(args.block_length),
                "params.mask_id": int(args.mask_id),
                "params.seed": (None if seed is None else int(seed)),
                "params.eos_token_id": (None if eos_token_id is None else int(eos_token_id)),
                "params.cfg_scale": float(cfg_scale),
                "params.remasking": str(remasking),
                "params.logits_eos_inf": bool(logits_eos_inf),
                "params.confidence_eos_eot_inf": bool(confidence_eos_eot_inf),
            }
        )

    t0 = time.time()
    if gen_length > 0:
        out_indices = diffusion_generate(
            model,
            in_indices,
            mask_id=int(args.mask_id),
            eos_token_id=(None if eos_token_id is None else int(eos_token_id)),
            steps=int(args.steps),
            gen_length=int(gen_length),
            block_length=int(args.block_length),
            temperature=float(args.temperature),
            cfg_scale=float(cfg_scale),
            remasking=str(remasking),
            logits_eos_inf=bool(logits_eos_inf),
            confidence_eos_eot_inf=bool(confidence_eos_eot_inf),
            generator=generator,
        )
    else:
        out_indices = in_indices
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
        print(output_string)

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
                    "total_length": args.total_length,
                    "gen_length": gen_length,
                    "block_length": args.block_length,
                    "mask_id": args.mask_id,
                    "seed": seed,
                    "eos_token_id": eos_token_id,
                    "cfg_scale": cfg_scale,
                    "remasking": remasking,
                    "logits_eos_inf": logits_eos_inf,
                    "confidence_eos_eot_inf": confidence_eos_eot_inf,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.log_artifact(artifact_path, name=artifact_path, type_="inference_outputs")
        except Exception:
            pass

    return [output_string]
