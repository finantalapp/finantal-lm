"""
Central path configuration — the SINGLE source of truth for every filesystem
location in the project.

Design: the *code* lives in this Git repo; all *heavy assets* (tokenized data,
SentencePiece tokenizer, checkpoints) live outside it — on Google Drive in Colab,
or a local mirror during development. Nothing here is hard-coded into the training
scripts; everything is read from environment variables with sensible defaults.

Override any path with an environment variable (set them in the Colab bootstrap
notebook, a shell, or a .env). The most important one is FINANTAL_DATA_ROOT:

    Colab (default):  /content/drive/MyDrive/finantal_data
    Local mirror:     export FINANTAL_DATA_ROOT=D:/finantal_data   (Windows)
                      export FINANTAL_DATA_ROOT=/data/finantal     (Linux)
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Repo-internal locations (committed to Git)
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
MODEL_CONFIG = str(CONFIG_DIR / "model_config.json")
TRAIN_CONFIG = str(CONFIG_DIR / "train_config.json")

# --------------------------------------------------------------------------- #
# External data root (Google Drive / local mirror) — NEVER committed to Git
# --------------------------------------------------------------------------- #
DATA_ROOT = os.environ.get("FINANTAL_DATA_ROOT", "/content/drive/MyDrive/finantal_data")

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(DATA_ROOT, "data"))
TOKENIZER_DIR = os.environ.get("TOKENIZER_DIR", os.path.join(DATA_ROOT, "tokenizer"))
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", os.path.join(DATA_ROOT, "checkpoints"))

# Logs are written locally (inside the repo, git-ignored) AND mirrored to Drive
# so they survive a Colab disconnect.
LOG_DIR = os.environ.get("LOG_DIR", str(REPO_ROOT / "logs"))
DRIVE_LOG_DIR = os.environ.get("DRIVE_LOG_DIR", os.path.join(DATA_ROOT, "logs"))

# --------------------------------------------------------------------------- #
# Concrete files / sub-dirs
# --------------------------------------------------------------------------- #
PRETRAIN_DATA = os.environ.get("PRETRAIN_DATA", os.path.join(DATA_DIR, "pretrain_tokenized.jsonl"))
SFT_DATA = os.environ.get("SFT_DATA", os.path.join(DATA_DIR, "sft_tokenized_v2.jsonl"))
DATASET_STATS = os.path.join(DATA_DIR, "dataset_stats.json")

TOKENIZER_MODEL = os.environ.get("TOKENIZER_MODEL", os.path.join(TOKENIZER_DIR, "finantal_tokenizer.model"))
TOKENIZER_VOCAB = os.path.join(TOKENIZER_DIR, "finantal_tokenizer.vocab")

PRETRAIN_CKPT_DIR = os.path.join(CHECKPOINT_DIR, "pretrain")
# New SFT runs (v2 clean data) write to an ISOLATED dir so they:
#   (1) never auto-resume the OLD sft/ checkpoints,
#   (2) never collide with the old step numbering (old step_3900 vs new step_100),
#   (3) leave the old checkpoints/sft/ completely untouched.
# Override with env SFT_CKPT_DIR to point elsewhere. Old SFT kept at <CHECKPOINT_DIR>/sft.
SFT_CKPT_DIR = os.environ.get("SFT_CKPT_DIR", os.path.join(CHECKPOINT_DIR, "sft_v2"))
OLD_SFT_CKPT_DIR = os.path.join(CHECKPOINT_DIR, "sft")  # legacy — NOT a training start point
PRETRAIN_LATEST = os.path.join(PRETRAIN_CKPT_DIR, "latest.pt")
SFT_LATEST = os.path.join(SFT_CKPT_DIR, "latest.pt")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resolve_repo_path(path: str | None) -> str | None:
    """Resolve a possibly-relative path against the repo root (for config files)."""
    if not path:
        return path
    return path if os.path.isabs(path) else str(REPO_ROOT / path)


def ensure_dirs() -> None:
    """Create the external dirs we write to (idempotent)."""
    for d in (DATA_DIR, TOKENIZER_DIR, PRETRAIN_CKPT_DIR, SFT_CKPT_DIR, LOG_DIR, DRIVE_LOG_DIR):
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass


def verify_assets(require_tokenizer: bool = True, require_pretrain: bool = False,
                  require_sft: bool = False) -> list[str]:
    """Return a list of missing required assets (empty list == all good)."""
    missing = []
    if require_tokenizer and not os.path.exists(TOKENIZER_MODEL):
        missing.append(TOKENIZER_MODEL)
    if require_pretrain and not os.path.exists(PRETRAIN_DATA):
        missing.append(PRETRAIN_DATA)
    if require_sft and not os.path.exists(SFT_DATA):
        missing.append(SFT_DATA)
    return missing


def summary() -> str:
    return (
        f"DATA_ROOT      = {DATA_ROOT}\n"
        f"DATA_DIR       = {DATA_DIR}\n"
        f"TOKENIZER_DIR  = {TOKENIZER_DIR}\n"
        f"CHECKPOINT_DIR = {CHECKPOINT_DIR}\n"
        f"LOG_DIR        = {LOG_DIR}\n"
        f"DRIVE_LOG_DIR  = {DRIVE_LOG_DIR}"
    )


if __name__ == "__main__":
    print(summary())
    miss = verify_assets(require_pretrain=True, require_sft=True)
    print("\nMissing assets:" if miss else "\nAll assets present.")
    for m in miss:
        print("  -", m)
