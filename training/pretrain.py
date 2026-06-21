"""
Pretraining entry point — from-scratch PyTorch training loop (no HF Trainer).

Paths are NOT hard-coded: data, tokenizer and checkpoints come from config/paths.py
(env-driven, default to Google Drive). Checkpoints are written to Drive; logs are
written locally and mirrored to Drive; training auto-resumes from the latest
checkpoint if one exists.

Usage:
    python -m training.pretrain --config config/train_config.json
    python -m training.pretrain --override micro_batch_size=4 gradient_accumulation_steps=32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import paths as P
from models.model import FinantalForCausalLM, ModelConfig
from data_pipeline.dataset_loader import (PackedPretrainDataset, JsonlExampleDataset,
                                          train_val_split)
from data_pipeline.collator import PackedCollator, CausalLMCollator
from training.common import (build_optimizer, resolve_amp, count_params_human,
                             CosineScheduler, evaluate)
from utils.seed import set_seed, seed_worker
from utils.logging import TrainLogger
from utils.checkpoint import save_checkpoint, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=P.TRAIN_CONFIG)
    p.add_argument("--section", default="pretrain")
    p.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    return p.parse_args()


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        cfg[k] = v
    return cfg


def resolve_resume(cfg: dict, output_dir: str, logger) -> str | None:
    """Explicit resume_from wins; otherwise auto-resume from <output_dir>/latest.pt."""
    if cfg.get("resume_from"):
        return cfg["resume_from"]
    if cfg.get("auto_resume", True):
        cand = os.path.join(output_dir, "latest.pt")
        if os.path.exists(cand):
            logger.info(f"[auto-resume] found {cand}")
            return cand
    return None


def main():
    args = parse_args()
    P.ensure_dirs()

    cfg_path = P.resolve_repo_path(args.config) if not os.path.exists(args.config) else args.config
    with open(cfg_path, "r", encoding="utf-8") as f:
        full_cfg = json.load(f)
    cfg = apply_overrides(dict(full_cfg[args.section]), args.override)
    set_seed(full_cfg.get("seed", 1234))

    # resolve external paths from the central config (with optional per-run override)
    data_path = cfg.get("data_path") or P.PRETRAIN_DATA
    output_dir = cfg.get("output_dir") or P.PRETRAIN_CKPT_DIR
    model_config_path = P.resolve_repo_path(full_cfg.get("model_config")) or P.MODEL_CONFIG

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = TrainLogger(P.LOG_DIR, run_name="pretrain", mirror_dir=P.DRIVE_LOG_DIR)
    logger.info(f"device={device} | section='{args.section}'")
    logger.info(f"data={data_path}\noutput={output_dir}")
    logger.info(f"effective config: {json.dumps(cfg, ensure_ascii=False)}")

    missing = [p for p in [data_path, model_config_path] if not os.path.exists(p)]
    if missing:
        logger.info(f"FATAL: missing required files: {missing}")
        logger.info("Check that Google Drive is mounted and FINANTAL_DATA_ROOT is correct.")
        sys.exit(1)

    # ----- model -----
    model_cfg = ModelConfig.from_json(model_config_path)
    model_cfg.max_position_embeddings = max(model_cfg.max_position_embeddings, cfg["max_seq_len"])
    model = FinantalForCausalLM(model_cfg).to(device)
    logger.info(f"model parameters: {count_params_human(model.num_parameters())} "
                f"({model.num_parameters():,})")

    # ----- data (95/5 train/val split) -----
    val_ratio = cfg.get("val_ratio", 0.0) or 0.0
    if cfg.get("pack_sequences", True):
        train_ds = PackedPretrainDataset(data_path, block_size=cfg["max_seq_len"],
                                         eos_token_id=model_cfg.eos_token_id,
                                         split=("train" if val_ratio > 0 else "all"),
                                         val_ratio=val_ratio)
        val_ds = (PackedPretrainDataset(data_path, block_size=cfg["max_seq_len"],
                                        eos_token_id=model_cfg.eos_token_id,
                                        split="val", val_ratio=val_ratio)
                  if val_ratio > 0 else None)
        collate = PackedCollator()
        shuffle = False
    else:
        full = JsonlExampleDataset(data_path, max_seq_len=cfg["max_seq_len"], has_labels=False)
        train_ds, val_ds = train_val_split(full, val_ratio, seed=full_cfg.get("seed", 1234))
        collate = CausalLMCollator(pad_token_id=model_cfg.pad_token_id, max_seq_len=cfg["max_seq_len"])
        shuffle = True

    loader = DataLoader(
        train_ds, batch_size=cfg["micro_batch_size"], shuffle=shuffle,
        num_workers=cfg.get("num_workers", 2), collate_fn=collate,
        pin_memory=(device == "cuda"), drop_last=True,
        worker_init_fn=seed_worker, persistent_workers=cfg.get("num_workers", 2) > 0,
    )
    val_loader = (DataLoader(val_ds, batch_size=cfg["micro_batch_size"], shuffle=False,
                             num_workers=0, collate_fn=collate,
                             pin_memory=(device == "cuda"), drop_last=True)
                  if val_ds is not None else None)
    val_status = ("enabled (~%.0f%% holdout)" % (val_ratio * 100)) if val_loader is not None else "disabled"
    logger.info(f"validation: {val_status}")

    # ----- optimizer / amp -----
    accum = cfg["gradient_accumulation_steps"]
    max_steps = cfg["max_steps"]
    optimizer = build_optimizer(
        model, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
        betas=(cfg["beta1"], cfg["beta2"]), eps=cfg["eps"],
        use_8bit=cfg.get("use_8bit_optimizer", False), logger=logger,
    )
    amp_dtype, use_scaler = resolve_amp(cfg["precision"])
    # GradScaler is ONLY needed for fp16 (narrow range -> gradient underflow/overflow).
    # bf16 has fp32-like dynamic range, so we drop the scaler entirely (scaler = None).
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None
    scheduler = CosineScheduler(optimizer, warmup_steps=cfg["warmup_steps"], max_steps=max_steps,
                                base_lr=cfg["learning_rate"], min_lr=cfg["min_lr"])
    logger.info(f"precision={cfg['precision']} (amp_dtype={amp_dtype}, grad_scaler={'on' if use_scaler else 'OFF (bf16)'})")

    # ----- auto / explicit resume (restores model + optimizer + scheduler + scaler + step) -----
    start_step = 0
    resume_from = resolve_resume(cfg, output_dir, logger)
    if resume_from:
        ckpt = load_checkpoint(resume_from, model=model, optimizer=optimizer,
                               scheduler=scheduler, scaler=scaler, map_location=device)
        start_step = ckpt.get("step", 0)
        logger.info(f"resumed from {resume_from} at step {start_step}")

    # ----- train loop -----
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = start_step
    micro = 0
    running_loss = 0.0
    t0 = time.time()
    tokens_since = 0
    tokens_per_batch = cfg["micro_batch_size"] * cfg["max_seq_len"]

    logger.info(f"starting pretraining: max_steps={max_steps}, "
                f"effective_batch={cfg['micro_batch_size'] * accum} seqs, "
                f"tokens/step={tokens_per_batch * accum:,}")

    done = False
    while not done:
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                _, loss = model(input_ids, labels=labels)
                loss = loss / accum

            # fp16: scale the loss before backward; bf16/fp32: plain backward (no scaler)
            (scaler.scale(loss) if use_scaler else loss).backward()
            running_loss += loss.item() * accum
            tokens_since += tokens_per_batch
            micro += 1

            if micro % accum == 0:
                grad_norm = None
                # fp16 only: unscale grads back to true magnitude before clipping
                if use_scaler:
                    scaler.unscale_(optimizer)
                if cfg["grad_clip"] and cfg["grad_clip"] > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]).item()

                lr = scheduler.set_step(global_step)

                # fp16: scaler.step + update; bf16/fp32: a normal optimizer step
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg["log_every"] == 0:
                    dt = time.time() - t0
                    tps = tokens_since / dt if dt > 0 else 0.0
                    logger.log_step(global_step, running_loss / (cfg["log_every"] * accum), lr,
                                    grad_norm=grad_norm, tokens_per_sec=tps)
                    running_loss, tokens_since, t0 = 0.0, 0, time.time()

                if val_loader is not None and global_step % cfg.get("eval_every", 100) == 0:
                    vloss, vppl = evaluate(model, val_loader, device=device, amp_dtype=amp_dtype,
                                           max_batches=cfg.get("eval_max_batches", 50),
                                           pad_token_id=model_cfg.pad_token_id)
                    if vloss is not None:
                        logger.log_eval(global_step, vloss, vppl)
                    t0 = time.time()  # don't count eval time against tok/s

                if global_step % cfg["save_every"] == 0:
                    path = save_checkpoint(output_dir, global_step, model=model,
                                           optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                                           model_config=model_cfg.to_dict(),
                                           keep_last_n=cfg["keep_last_n"])
                    logger.info(f"saved checkpoint -> {path}")

                if global_step >= max_steps:
                    done = True
                    break

    final = save_checkpoint(output_dir, global_step, model=model, optimizer=optimizer,
                            scheduler=scheduler, scaler=scaler, model_config=model_cfg.to_dict(),
                            keep_last_n=cfg["keep_last_n"])
    logger.info(f"pretraining complete at step {global_step}. final checkpoint -> {final}")
    logger.close()


if __name__ == "__main__":
    main()
