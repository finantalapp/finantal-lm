"""
Dataset loaders for tokenized JSONL.

Two formats are supported:
  - Pretrain : {"input_ids": [...]}
  - SFT      : {"input_ids": [...], "labels": [...]}

For pretraining over a ~1GB file we avoid loading everything into RAM:
  * `PackedPretrainDataset` streams the file and packs token streams into
    contiguous `max_seq_len` blocks (the standard, efficient way to pretrain —
    no padding waste). It is an IterableDataset and shards across workers.
  * `JsonlExampleDataset` is a map-style dataset that builds (and caches) a byte
    offset index so each line can be seeked individually — used for SFT, where we
    want one conversation per example.
"""

from __future__ import annotations

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


# --------------------------------------------------------------------------- #
# Offset index (for random-access map-style reading)
# --------------------------------------------------------------------------- #
def build_offset_index(path: str, cache: bool = True) -> np.ndarray:
    """Return an array of byte offsets, one per JSONL line. Cached next to the file."""
    idx_path = path + ".offsets.npy"
    if cache and os.path.exists(idx_path) and os.path.getmtime(idx_path) >= os.path.getmtime(path):
        return np.load(idx_path)

    offsets = []
    with open(path, "rb") as f:
        offset = 0
        for line in f:
            if line.strip():
                offsets.append(offset)
            offset += len(line)
    arr = np.asarray(offsets, dtype=np.int64)
    if cache:
        try:
            np.save(idx_path, arr)
        except OSError:
            pass
    return arr


# --------------------------------------------------------------------------- #
# Map-style dataset (one example per line) — used for SFT
# --------------------------------------------------------------------------- #
class JsonlExampleDataset(Dataset):
    def __init__(self, path: str, max_seq_len: int = 1024, has_labels: bool = True):
        self.path = path
        self.max_seq_len = max_seq_len
        self.has_labels = has_labels
        self.offsets = build_offset_index(path)
        self._fh = None  # lazy per-process file handle

    def __len__(self) -> int:
        return len(self.offsets)

    def _file(self):
        # open lazily so the handle is created inside each DataLoader worker process
        if self._fh is None:
            self._fh = open(self.path, "rb")
        return self._fh

    def __getitem__(self, i: int):
        f = self._file()
        f.seek(int(self.offsets[i]))
        obj = json.loads(f.readline())
        input_ids = obj["input_ids"][: self.max_seq_len]
        if self.has_labels and "labels" in obj:
            labels = obj["labels"][: self.max_seq_len]
        else:
            labels = list(input_ids)
        return {"input_ids": input_ids, "labels": labels}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None  # don't pickle the open file handle across workers
        return state


# --------------------------------------------------------------------------- #
# Iterable packed dataset — used for pretraining
# --------------------------------------------------------------------------- #
class PackedPretrainDataset(IterableDataset):
    """
    Stream a pretrain JSONL and emit fixed-length `block_size` token blocks.

    Documents are concatenated with an EOS separator and chopped into blocks, so
    every block is full (no padding). Sharding: each DataLoader worker reads a
    disjoint subset of lines (round-robin), so multi-worker loading does not
    duplicate data.
    """

    def __init__(self, path: str, block_size: int = 1024, eos_token_id: int = 3,
                 add_eos: bool = True):
        self.path = path
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.add_eos = add_eos

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        buffer: list[int] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if line_no % num_workers != worker_id:
                    continue  # shard by line across workers
                line = line.strip()
                if not line:
                    continue
                ids = json.loads(line)["input_ids"]
                buffer.extend(ids)
                if self.add_eos:
                    buffer.append(self.eos_token_id)

                while len(buffer) >= self.block_size:
                    block = buffer[: self.block_size]
                    buffer = buffer[self.block_size:]
                    yield {"input_ids": block, "labels": list(block)}
        # trailing remainder shorter than block_size is dropped (keeps blocks uniform)


def quick_token_stats(path: str, sample_lines: int | None = None) -> dict:
    """Stream a JSONL once and return token/length statistics (used for dataset_stats.json)."""
    n, total_tokens = 0, 0
    min_len, max_len = None, 0
    has_labels = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ln = len(obj["input_ids"])
            has_labels = has_labels or ("labels" in obj)
            n += 1
            total_tokens += ln
            max_len = max(max_len, ln)
            min_len = ln if min_len is None else min(min_len, ln)
            if sample_lines and n >= sample_lines:
                break
    return {
        "num_examples": n,
        "total_tokens": total_tokens,
        "avg_length": round(total_tokens / max(n, 1), 2),
        "min_length": min_len or 0,
        "max_length": max_len,
        "has_labels": has_labels,
        "sampled": bool(sample_lines),
    }
