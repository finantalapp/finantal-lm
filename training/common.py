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


class CosineScheduler:
    """
    Cosine schedule with linear warmup, exposed as an object with a checkpointable
    state (so optimizer/scheduler/scaler/step are all saved & restored together).

    The LR is a pure function of the step, so resuming is exact: `set_step(g)` writes
    the correct LR into the optimizer for global step `g`. state_dict() persists the
    schedule parameters + last step.
    """

    def __init__(self, optimizer, *, warmup_steps, max_steps, base_lr, min_lr, last_step=-1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.last_step = last_step
        self.set_step(max(0, last_step))

    def get_lr(self, step):
        return cosine_lr(step, warmup_steps=self.warmup_steps, max_steps=self.max_steps,
                         base_lr=self.base_lr, min_lr=self.min_lr)

    def set_step(self, step):
        """Set the optimizer LR for `step` and record it. Returns the LR."""
        self.last_step = step
        lr = self.get_lr(step)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def state_dict(self):
        return {"warmup_steps": self.warmup_steps, "max_steps": self.max_steps,
                "base_lr": self.base_lr, "min_lr": self.min_lr, "last_step": self.last_step}

    def load_state_dict(self, sd):
        self.warmup_steps = sd.get("warmup_steps", self.warmup_steps)
        self.max_steps = sd.get("max_steps", self.max_steps)
        self.base_lr = sd.get("base_lr", self.base_lr)
        self.min_lr = sd.get("min_lr", self.min_lr)
        self.last_step = sd.get("last_step", self.last_step)


@torch.no_grad()
def evaluate(model, val_loader, *, device, amp_dtype, max_batches=50, pad_token_id=0):
    """
    Token-weighted validation loss + perplexity over up to `max_batches` batches.
    Returns (loss, perplexity) or (None, None) if there is no validation data.
    Restores the model to train() mode on exit.
    """
    if val_loader is None:
        return None, None
    import math
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    n = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        labels = labels.masked_fill(input_ids == pad_token_id, -100)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                            dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            _, loss = model(input_ids, labels=labels)
        # weight by the number of supervised target tokens in this batch
        ntok = int((labels[:, 1:] != -100).sum().item())
        if ntok > 0 and loss is not None:
            total_loss += loss.item() * ntok
            total_tokens += ntok
        n += 1
        if n >= max_batches:
            break
    if was_training:
        model.train()
    if total_tokens == 0:
        return None, None
    mean = total_loss / total_tokens
    return mean, math.exp(min(mean, 20.0))


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
