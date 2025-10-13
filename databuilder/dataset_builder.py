import time
import numpy as np
from typing import Optional
from logger import Logger


def count_total_tokens(tokenizer, input_filename, *, logger: Optional[Logger] = None):
    with open(input_filename, 'r') as f:
        total_tokens = 0
        start_time = time.time()
        last_time = start_time

        for token_id in tokenizer.encode_iterable(f):
            total_tokens += 1
            if total_tokens % 1_000_000 == 0:
                now = time.time()
                elapsed = now - start_time
                interval = now - last_time
                if logger is not None:
                    logger.log(
                        {
                            "phase": "data",
                            "metrics.tokens_processed": int(total_tokens),
                            "metrics.tokens_per_sec": float(1_000_000 / interval) if interval > 0 else float("inf"),
                        }
                    )
                last_time = now

        total_elapsed = time.time() - start_time
        if logger is not None:
            logger.log(
                {
                    "phase": "data",
                    "event": "finalize",
                    "metrics.total_tokens": int(total_tokens),
                    "metrics.duration_s": float(total_elapsed),
                }
            )


def write_token_ids_to_memmap(tokenizer, input_filename, total_tokens, output_filename, dtype=np.int32, *, logger: Optional[Logger] = None):
    arr = np.memmap(output_filename, dtype=dtype, mode='w+', shape=(total_tokens,))
    with open(input_filename, 'r') as f:
        start_time = time.time()
        last_time = start_time
        for i, token_id in enumerate(tokenizer.encode_iterable(f)):
            arr[i] = token_id
            if i > 0 and i % 1_000_000 == 0:
                now = time.time()
                elapsed = now - start_time
                interval = now - last_time
                if logger is not None:
                    logger.log(
                        {
                            "phase": "data",
                            "metrics.tokens_processed": int(i),
                            "metrics.tokens_per_sec": float(1_000_000 / interval) if interval > 0 else float("inf"),
                            "params.dtype": str(dtype),
                        }
                    )
                last_time = now
    arr.flush()
    total_time = time.time() - start_time
    if logger is not None:
        logger.log(
            {
                "phase": "data",
                "event": "finalize",
                "metrics.total_tokens": int(total_tokens),
                "metrics.duration_s": float(total_time),
                "params.dtype": str(dtype),
                "params.shape": f"({total_tokens},)",
            }
        )

