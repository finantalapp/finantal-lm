"""
Collators turn a list of {"input_ids", "labels"} examples into padded batch tensors.

The model shifts internally (position i predicts i+1), so the collator just needs
to right-pad every sequence to the batch max length and mark padded label
positions with -100 (ignored by cross-entropy).

Prompt masking (SFT): the dataset ships `labels == input_ids` (loss on the whole
sequence). For correct instruction tuning we want loss ONLY on the assistant's
answer. When `assistant_marker` is given (e.g. the token IDs for "▁Assistant" + ":"),
the collator rebuilds the labels from `input_ids`, setting every token up to and
INCLUDING that marker to -100, and supervising only the response that follows.
The on-disk data format is never modified — masking happens at collation time.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


def find_subsequence(seq: list[int], pattern: list[int]) -> int:
    """Return the start index of the first occurrence of `pattern` in `seq`, else -1."""
    if not pattern:
        return -1
    n, m = len(seq), len(pattern)
    for i in range(n - m + 1):
        if seq[i:i + m] == pattern:
            return i
    return -1


def mask_prompt_labels(input_ids: list[int], assistant_marker: list[int]) -> list[int]:
    """
    Build response-only labels: everything up to and including the FIRST
    `assistant_marker` is set to -100; the response tokens after it keep their IDs.
    Falls back to supervising the whole sequence if the marker is not found.
    """
    start = find_subsequence(input_ids, assistant_marker)
    if start < 0:
        return list(input_ids)  # marker absent -> supervise everything (no example wasted)
    resp_start = start + len(assistant_marker)
    return [IGNORE_INDEX] * resp_start + list(input_ids[resp_start:])


class CausalLMCollator:
    def __init__(self, pad_token_id: int = 0, max_seq_len: int | None = None,
                 assistant_marker: list[int] | None = None):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        # when set, labels are derived from input_ids with the prompt masked out
        self.assistant_marker = assistant_marker

    def __call__(self, batch: list[dict]) -> dict:
        input_ids = [ex["input_ids"] for ex in batch]
        if self.assistant_marker:
            labels = [mask_prompt_labels(x, self.assistant_marker) for x in input_ids]
        else:
            labels = [ex.get("labels", ex["input_ids"]) for ex in batch]

        if self.max_seq_len is not None:
            input_ids = [x[: self.max_seq_len] for x in input_ids]
            labels = [y[: self.max_seq_len] for y in labels]

        max_len = max(len(x) for x in input_ids)

        in_batch, lab_batch, attn_batch = [], [], []
        for x, y in zip(input_ids, labels):
            pad = max_len - len(x)
            attn = [1] * len(x) + [0] * pad
            x = x + [self.pad_token_id] * pad
            y = y + [IGNORE_INDEX] * pad  # padded label positions are ignored by the loss
            in_batch.append(x)
            lab_batch.append(y)
            attn_batch.append(attn)

        return {
            "input_ids": torch.tensor(in_batch, dtype=torch.long),
            "labels": torch.tensor(lab_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attn_batch, dtype=torch.long),
        }


class PackedCollator:
    """For PackedPretrainDataset every block is already exactly block_size — just stack."""

    def __call__(self, batch: list[dict]) -> dict:
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
        labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}
