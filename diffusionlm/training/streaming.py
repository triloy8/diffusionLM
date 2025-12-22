from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Iterator, Optional
import time

from datasets import load_dataset
from datasets.utils import logging as hf_logging
import torch

from diffusionlm.tokenizer.tokenizer import Tokenizer
from logger import Logger


class HFTokenIteratorFactory:
    """Constructs token streams from Hugging Face streaming datasets."""

    def __init__(
        self,
        *,
        dataset_name: str,
        dataset_config: Optional[str],
        split: str,
        text_field: str,
        tokenizer: Tokenizer,
        shuffle_buffer_size: int = 0,
        shuffle_seed: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
        pad_newline: bool = True,
        logger: Optional[Logger] = None,
        hf_debug_logging: bool = False,
        slow_row_s: float = 10.0,
        slow_encode_s: float = 10.0,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.text_field = text_field
        self.tokenizer = tokenizer
        self.shuffle_buffer_size = max(0, shuffle_buffer_size)
        self.shuffle_seed = shuffle_seed if shuffle_seed is not None else 0
        self.world_size = max(1, world_size)
        self.rank = max(0, min(rank, self.world_size - 1))
        self.pad_newline = pad_newline
        self._epoch = 0
        self._logger = logger
        self._slow_row_s = float(slow_row_s)
        self._slow_encode_s = float(slow_encode_s)
        if hf_debug_logging:
            hf_logging.set_verbosity_debug()

    def _load_dataset(self):
        if self.dataset_config:
            return load_dataset(self.dataset_name, self.dataset_config, split=self.split, streaming=True)
        return load_dataset(self.dataset_name, split=self.split, streaming=True)

    def _apply_shuffle(self, dataset):
        if self.shuffle_buffer_size <= 0:
            return dataset
        seed = self.shuffle_seed + self._epoch
        return dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=seed)

    def _manual_shard(self, iterable: Iterable) -> Iterator:
        for idx, example in enumerate(iterable):
            if idx % self.world_size == self.rank:
                yield example

    def _apply_shard(self, dataset) -> Iterator:
        if self.world_size <= 1:
            return iter(dataset)
        try:
            sharded = dataset.shard(num_shards=self.world_size, index=self.rank)
            num_shards = getattr(sharded, "n_shards", None) or getattr(sharded, "num_shards", None)
            if num_shards is not None and num_shards >= self.world_size:
                return iter(sharded)
            # Fall back when HF dataset cannot provide enough shards.
            return self._manual_shard(iter(dataset))
        except Exception:
            return self._manual_shard(iter(dataset))

    def __call__(self) -> Iterator[int]:
        dataset = self._load_dataset()
        dataset = self._apply_shuffle(dataset)
        rows = iter(self._apply_shard(dataset))
        self._epoch += 1
        while True:
            row_fetch_start = time.monotonic()
            try:
                row = next(rows)
            except StopIteration:
                break
            row_fetch_elapsed = time.monotonic() - row_fetch_start
            if self._logger is not None and row_fetch_elapsed >= self._slow_row_s:
                self._logger.log(
                    {
                        "metrics.streaming_row_fetch/elapsed_s": float(row_fetch_elapsed),
                        "metrics.streaming_row_fetch/epoch": int(self._epoch),
                        "metrics.streaming_row_fetch/rank": int(self.rank),
                        "metrics.streaming_row_fetch/world_size": int(self.world_size),
                    }
                )

            if row is None:
                continue
            text_value = row.get(self.text_field) if isinstance(row, dict) else None
            if text_value is None:
                continue
            if not isinstance(text_value, str):
                text_value = str(text_value)
            normalized = text_value
            if self.pad_newline and not normalized.endswith("\n"):
                normalized = f"{normalized}\n"
            encode_start = time.monotonic()
            token_ids = self.tokenizer.encode(normalized)
            encode_elapsed = time.monotonic() - encode_start
            if self._logger is not None and encode_elapsed >= self._slow_encode_s:
                self._logger.log(
                    {
                        "metrics.streaming_encode/elapsed_s": float(encode_elapsed),
                        "metrics.streaming_encode/epoch": int(self._epoch),
                        "metrics.streaming_encode/rank": int(self.rank),
                        "metrics.streaming_encode/world_size": int(self.world_size),
                    }
                )
            for token_id in token_ids:
                yield int(token_id)


class StreamingBatcher:
    """Rolling token buffer that emits fixed-length sequences for batching."""

    def __init__(
        self,
        iterator_factory: HFTokenIteratorFactory,
        *,
        device: str | torch.device,
        logger: Optional[Logger] = None,
        stall_warn_s: float = 10.0,
        stall_repeat_s: float = 30.0,
    ) -> None:
        self.iterator_factory = iterator_factory
        self.device = torch.device(device)
        self._buffer: Deque[int] = deque()
        self._iterator: Iterator[int] = self.iterator_factory()
        self._logger = logger
        self._stall_warn_s = float(stall_warn_s)
        self._stall_repeat_s = float(stall_repeat_s)

    def _next_token(self) -> int:
        while True:
            try:
                return next(self._iterator)
            except StopIteration:
                self._iterator = self.iterator_factory()

    def _ensure_tokens(self, count: int) -> None:
        start = time.monotonic()
        last_log = start
        while len(self._buffer) < count:
            self._buffer.append(self._next_token())
            if self._logger is None:
                continue
            now = time.monotonic()
            elapsed = now - start
            if elapsed < self._stall_warn_s or (now - last_log) < self._stall_repeat_s:
                continue
            last_log = now
            self._logger.log(
                {
                    "metrics.streaming_stall/elapsed_s": float(elapsed),
                    "metrics.streaming_stall/buffer_len": int(len(self._buffer)),
                    "metrics.streaming_stall/tokens_needed": int(count),
                    "metrics.streaming_stall/shuffle_buffer_size": int(
                        self.iterator_factory.shuffle_buffer_size
                    ),
                    "metrics.streaming_stall/epoch": int(self.iterator_factory._epoch),
                    "metrics.streaming_stall/rank": int(self.iterator_factory.rank),
                    "metrics.streaming_stall/world_size": int(self.iterator_factory.world_size),
                    "metrics.streaming_stall/dataset_name": str(self.iterator_factory.dataset_name),
                    "metrics.streaming_stall/dataset_config": str(self.iterator_factory.dataset_config or ""),
                    "metrics.streaming_stall/split": str(self.iterator_factory.split),
                }
            )

    def draw(self, batch_size: int, context_length: int) -> torch.Tensor:
        tokens_needed = batch_size * context_length
        self._ensure_tokens(tokens_needed)
        sequences = []
        for _ in range(batch_size):
            seq = [self._buffer.popleft() for _ in range(context_length)]
            sequences.append(seq)
        clean_targets = torch.tensor(sequences, dtype=torch.long, device=self.device)
        return clean_targets
