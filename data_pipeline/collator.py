"""
Collators turn a list of {"input_ids", "labels"} examples into padded batch tensors.

The model shifts internally (position i predicts i+1), so the collator just needs
to right-pad every sequence to the batch max length and mark padded label
positions with -100 (ignored by cross-entropy).

`mask_prompt` (SFT only): if your SFT data has a known prompt/response boundary and
you want loss only on the response, set the prompt label positions to -100. The
provided sft_tokenized.jsonl already carries `labels`, so by default we honour them
as-is and only handle padding.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


class CausalLMCollator:
    def __init__(self, pad_token_id: int = 0, max_seq_len: int | None = None):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, batch: list[dict]) -> dict:
        input_ids = [ex["input_ids"] for ex in batch]
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
            # pad labels with IGNORE_INDEX; also ignore any position the labels
            # themselves marked as pad_token (defensive)
            y = y + [IGNORE_INDEX] * pad
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
