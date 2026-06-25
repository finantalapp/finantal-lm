"""
Checkpoint save/load with rotation.

A checkpoint bundles: model weights, optimizer state, LR-scheduler state, GradScaler
state, global step, and the model config. `latest.pt` always points to the newest
checkpoint (copied, not symlinked — Windows/Colab friendly). Old step checkpoints
beyond `keep_last_n` are pruned.
"""

from __future__ import annotations

import glob
import os
import re
import shutil

import torch

# --------------------------------------------------------------------------- #
# Permanent deletion (Google Drive trash fix)
# --------------------------------------------------------------------------- #
# On Colab, `os.remove` on a file under /content/drive moves it to Drive Trash,
# which KEEPS consuming quota -> Drive fills up -> checkpoint saving breaks.
# When FINANTAL_DRIVE_PURGE != "0", we additionally delete the trashed copy
# permanently via the Drive API (best-effort; silently falls back off-Colab).
_PURGE = os.environ.get("FINANTAL_DRIVE_PURGE", "1") not in ("0", "false", "False")
_DRIVE_SVC = None  # cached Drive API client (None=untried, False=unavailable)


def _drive_service():
    global _DRIVE_SVC
    if _DRIVE_SVC is not None:
        return _DRIVE_SVC
    try:
        from google.colab import auth          # only present on Colab
        auth.authenticate_user()
        from googleapiclient.discovery import build
        _DRIVE_SVC = build("drive", "v3", cache_discovery=False)
    except Exception:
        _DRIVE_SVC = False
    return _DRIVE_SVC


def _permanent_delete(path: str) -> None:
    """Remove a checkpoint file and, on Colab, purge its trashed copy permanently."""
    name = os.path.basename(path)
    # Truncate to 0 bytes BEFORE deleting: even if the Drive API purge below is
    # unavailable (no auth / off-Colab), the copy that lands in Drive Trash is
    # empty and consumes ~no quota.
    try:
        with open(path, "wb"):
            pass
    except OSError:
        pass
    try:
        os.remove(path)                          # removes from the working tree (-> Drive trash)
    except OSError:
        pass
    if not _PURGE:
        return
    svc = _drive_service()
    if not svc:
        return
    try:                                          # find the just-trashed file by name and hard-delete
        q = f"name = '{name}' and trashed = true"
        res = svc.files().list(q=q, spaces="drive",
                               fields="files(id,name)", pageSize=20).execute()
        for f in res.get("files", []):
            svc.files().delete(fileId=f["id"]).execute()   # permanent (skips trash)
    except Exception:
        pass


def save_checkpoint(output_dir: str, step: int, *, model, optimizer=None, scheduler=None,
                    scaler=None, model_config: dict | None = None, extra: dict | None = None,
                    keep_last_n: int = 3) -> str:
    os.makedirs(output_dir, exist_ok=True)
    # unwrap torch.compile / DataParallel if present
    raw_model = getattr(model, "_orig_mod", model)
    raw_model = getattr(raw_model, "module", raw_model)

    payload = {
        "step": step,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "model_config": model_config,
    }
    if extra:
        payload["extra"] = extra

    ckpt_path = os.path.join(output_dir, f"step_{step}.pt")
    tmp_path = ckpt_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, ckpt_path)  # atomic

    latest_path = os.path.join(output_dir, "latest.pt")
    shutil.copyfile(ckpt_path, latest_path)

    _prune(output_dir, keep_last_n)
    return ckpt_path


def _prune(output_dir: str, keep_last_n: int) -> None:
    """Keep only the newest `keep_last_n` step_*.pt files (latest.pt is never touched).
    Older ones are permanently deleted (Drive-trash-safe)."""
    if keep_last_n is None or keep_last_n <= 0:
        return
    ckpts = glob.glob(os.path.join(output_dir, "step_*.pt"))

    def step_of(p):
        m = re.search(r"step_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    ckpts.sort(key=step_of)
    for p in ckpts[:-keep_last_n]:        # everything except the newest keep_last_n
        _permanent_delete(p)


def load_checkpoint(path: str, *, model=None, optimizer=None, scheduler=None,
                    scaler=None, map_location="cpu", strict: bool = True) -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None and ckpt.get("model") is not None:
        raw_model = getattr(model, "_orig_mod", model)
        raw_model = getattr(raw_model, "module", raw_model)
        raw_model.load_state_dict(ckpt["model"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def load_weights_only(path: str, model, map_location="cpu", strict: bool = False) -> None:
    """Used by SFT to initialise from a pretrained checkpoint (weights only, fresh optimizer)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    raw_model = getattr(model, "_orig_mod", model)
    raw_model = getattr(raw_model, "module", raw_model)
    missing, unexpected = raw_model.load_state_dict(state, strict=strict)
    return {"missing": missing, "unexpected": unexpected}
