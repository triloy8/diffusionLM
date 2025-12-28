from __future__ import annotations

from typing import Iterable, Iterator, Optional
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
        context_length: int,
        eot_token_id: int,
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
        self.context_length = int(context_length)
        self.eot_token_id = int(eot_token_id)
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

    def __call__(self) -> Iterator[list[int]]:
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
            tokens = list(token_ids)
            tokens.append(self.eot_token_id)
            yield tokens


class StreamingBatcher:
    """Packed-stream batcher that emits fixed-length sequences for batching."""

    def __init__(
        self,
        iterator_factory: HFTokenIteratorFactory,
        *,
        device: str | torch.device,
        logger: Optional[Logger] = None,
    ) -> None:
        self.iterator_factory = iterator_factory
        self.device = torch.device(device)
        self._iterator: Iterator[list[int]] = self.iterator_factory()
        self._buffer: list[int] = []
        self._logger = logger

    def _extend_buffer(self) -> None:
        while True:
            try:
                self._buffer.extend(next(self._iterator))
                return
            except StopIteration:
                self._iterator = self.iterator_factory()

    def draw(self, batch_size: int, context_length: int) -> torch.Tensor:
        if context_length <= 0:
            raise ValueError("context_length must be > 0")
        sequences: list[list[int]] = []
        for _ in range(batch_size):
            while len(self._buffer) < context_length:
                self._extend_buffer()
            seq = self._buffer[:context_length]
            self._buffer = self._buffer[context_length:]
            sequences.append(seq)
        clean_targets = torch.tensor(sequences, dtype=torch.long, device=self.device)
        return clean_targets
