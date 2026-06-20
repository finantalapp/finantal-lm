"""
Generate <DATA_DIR>/dataset_stats.json by streaming each tokenized JSONL once.
Reads paths from config/paths.py (Drive by default). Stdlib only — no torch needed.

    python scripts/build_dataset_stats.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import paths as P


def quick_token_stats(path: str, sample_lines=None) -> dict:
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
    }


def main():
    datasets = {"pretrain": P.PRETRAIN_DATA, "sft": P.SFT_DATA}
    stats = {"vocab_size": 32000, "tokenizer": P.TOKENIZER_MODEL, "splits": {}}
    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"skip {name}: {path} not found")
            continue
        print(f"scanning {name}: {path} ...")
        s = quick_token_stats(path)
        s["path"] = path
        s["file_size_bytes"] = os.path.getsize(path)
        stats["splits"][name] = s
        print(f"  {name}: {s['num_examples']:,} examples, {s['total_tokens']:,} tokens, "
              f"avg_len={s['avg_length']}")

    os.makedirs(P.DATA_DIR, exist_ok=True)
    with open(P.DATASET_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"wrote {P.DATASET_STATS}")


if __name__ == "__main__":
    main()
