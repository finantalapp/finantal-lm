"""
Minimal training logger: console + JSONL metrics, written locally AND mirrored to
Google Drive (so logs survive a Colab disconnect). No external dependencies
required. TensorBoard is used only if available and enabled. Perplexity is logged
alongside loss.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Optional


class TrainLogger:
    def __init__(self, log_dir: str, run_name: str, use_tensorboard: bool = False,
                 mirror_dir: Optional[str] = None):
        os.makedirs(log_dir, exist_ok=True)
        self.run_name = run_name
        self.metrics_path = os.path.join(log_dir, f"{run_name}_metrics.jsonl")
        self.text_path = os.path.join(log_dir, f"{run_name}.log")
        self._metrics_fh = open(self.metrics_path, "a", encoding="utf-8")
        self._text_fh = open(self.text_path, "a", encoding="utf-8")
        self.start_time = time.time()

        # optional mirror of the metrics file onto Drive
        self._mirror_fh = None
        if mirror_dir and os.path.abspath(mirror_dir) != os.path.abspath(log_dir):
            try:
                os.makedirs(mirror_dir, exist_ok=True)
                self._mirror_fh = open(os.path.join(mirror_dir, f"{run_name}_metrics.jsonl"),
                                       "a", encoding="utf-8")
            except OSError:
                self._mirror_fh = None

        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(os.path.join(log_dir, "tb", run_name))
            except Exception as e:  # pragma: no cover
                self.info(f"[logger] TensorBoard unavailable ({e}); continuing without it.")

    def info(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, file=sys.stdout, flush=True)
        self._text_fh.write(line + "\n")
        self._text_fh.flush()

    def log_step(self, step: int, loss: float, lr: float, *,
                 grad_norm: Optional[float] = None,
                 tokens_per_sec: Optional[float] = None,
                 extra: Optional[dict] = None) -> None:
        ppl = math.exp(min(loss, 20.0))  # clamp to avoid overflow on early noisy steps
        record = {
            "step": step,
            "loss": round(loss, 5),
            "perplexity": round(ppl, 3),
            "lr": lr,
            "elapsed_s": round(time.time() - self.start_time, 1),
        }
        if grad_norm is not None:
            record["grad_norm"] = round(grad_norm, 4)
        if tokens_per_sec is not None:
            record["tokens_per_sec"] = round(tokens_per_sec, 1)
        if extra:
            record.update(extra)

        line = json.dumps(record) + "\n"
        self._metrics_fh.write(line)
        self._metrics_fh.flush()
        if self._mirror_fh is not None:
            try:
                self._mirror_fh.write(line)
                self._mirror_fh.flush()
            except OSError:
                pass

        if self.tb is not None:
            self.tb.add_scalar("train/loss", loss, step)
            self.tb.add_scalar("train/perplexity", ppl, step)
            self.tb.add_scalar("train/lr", lr, step)
            if grad_norm is not None:
                self.tb.add_scalar("train/grad_norm", grad_norm, step)

        tps = f" | tok/s {tokens_per_sec:,.0f}" if tokens_per_sec else ""
        gn = f" | gnorm {grad_norm:.3f}" if grad_norm is not None else ""
        self.info(f"step {step:>7} | loss {loss:.4f} | ppl {ppl:8.2f} | lr {lr:.2e}{gn}{tps}")

    def log_eval(self, step: int, val_loss: float, val_ppl: float) -> None:
        record = {"step": step, "val_loss": round(val_loss, 5),
                  "val_perplexity": round(val_ppl, 3),
                  "elapsed_s": round(time.time() - self.start_time, 1)}
        line = json.dumps(record) + "\n"
        self._metrics_fh.write(line)
        self._metrics_fh.flush()
        if self._mirror_fh is not None:
            try:
                self._mirror_fh.write(line)
                self._mirror_fh.flush()
            except OSError:
                pass
        if self.tb is not None:
            self.tb.add_scalar("val/loss", val_loss, step)
            self.tb.add_scalar("val/perplexity", val_ppl, step)
        self.info(f"  [eval] step {step:>7} | val_loss {val_loss:.4f} | val_ppl {val_ppl:8.2f}")

    def close(self) -> None:
        for fh in (self._metrics_fh, self._text_fh, self._mirror_fh):
            try:
                if fh is not None:
                    fh.close()
            except Exception:
                pass
        if self.tb is not None:
            try:
                self.tb.close()
            except Exception:
                pass
