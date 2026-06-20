"""
Shared training engine pieces used by both pretrain.py and sft_train.py:
optimizer construction (with decay/no-decay groups + optional 8-bit Adam),
cosine LR schedule with warmup, and a single optimizer-step routine that handles
fp16 GradScaler, gradient accumulation, and gradient clipping.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


# --------------------------------------------------------------------------- #
# Optimizer
# --------------------------------------------------------------------------- #
def build_optimizer(model, *, lr, weight_decay, betas, eps, use_8bit=False, logger=None):
    """AdamW with weight decay applied only to >=2D params (no decay on norms/biases)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if use_8bit:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(groups, lr=lr, betas=betas, eps=eps)
            if logger:
                logger.info("[optim] using bitsandbytes AdamW8bit (low VRAM)")
            return opt
        except Exception as e:  # pragma: no cover
            if logger:
                logger.info(f"[optim] bitsandbytes unavailable ({e}); falling back to fused AdamW")

    fused_ok = torch.cuda.is_available()
    try:
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused_ok)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


# --------------------------------------------------------------------------- #
# LR schedule: linear warmup -> cosine decay to min_lr
# --------------------------------------------------------------------------- #
def cosine_lr(step, *, warmup_steps, max_steps, base_lr, min_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


# --------------------------------------------------------------------------- #
# Precision helpers
# --------------------------------------------------------------------------- #
def resolve_amp(precision: str):
    """Return (autocast_dtype, use_scaler). bf16 needs no scaler; fp16 does."""
    precision = (precision or "fp16").lower()
    if precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    if precision in ("fp16", "float16", "half"):
        return torch.float16, True
    if precision == "bf16":
        # requested bf16 but unsupported -> fall back to fp16
        return torch.float16, True
    return torch.float32, False


def count_params_human(n: int) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}T"
