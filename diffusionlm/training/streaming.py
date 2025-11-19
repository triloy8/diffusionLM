from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Iterator, Optional

from datasets import load_dataset
import torch

from diffusionlm.tokenizer.tokenizer import Tokenizer


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
        rows = self._apply_shard(dataset)
        self._epoch += 1
        for row in rows:
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
            token_ids = self.tokenizer.encode(normalized)
            for token_id in token_ids:
                yield int(token_id)


class StreamingBatcher:
    """Rolling token buffer that emits fixed-length sequences for batching."""

    def __init__(self, iterator_factory: HFTokenIteratorFactory, *, device: str | torch.device) -> None:
        self.iterator_factory = iterator_factory
        self.device = torch.device(device)
        self._buffer: Deque[int] = deque()
        self._iterator: Iterator[int] = self.iterator_factory()

    def _next_token(self) -> int:
        while True:
            try:
                return next(self._iterator)
            except StopIteration:
                self._iterator = self.iterator_factory()

    def _ensure_tokens(self, count: int) -> None:
        while len(self._buffer) < count:
            self._buffer.append(self._next_token())

    def draw(self, batch_size: int, context_length: int) -> torch.Tensor:
        tokens_needed = batch_size * context_length
        self._ensure_tokens(tokens_needed)
        sequences = []
        for _ in range(batch_size):
            seq = [self._buffer.popleft() for _ in range(context_length)]
            sequences.append(seq)
        clean_targets = torch.tensor(sequences, dtype=torch.long, device=self.device)
        return clean_targets
