"""
Supervised fine-tuning (SFT) entry point — from-scratch PyTorch training loop.

Loads pretrained weights via `init_from` (defaults to the pretrain latest.pt on
Drive), then fine-tunes on sft_tokenized.jsonl.

Key behaviours:
  * PROMPT MASKING: with mask_prompt=true, loss is computed ONLY on the assistant's
    answer. Everything up to and including the "▁Assistant :" marker
    (assistant_marker_ids) is set to -100. The on-disk data is never modified.
  * 95/5 train/val split with periodic validation loss + perplexity.
  * Checkpoints every `save_every` steps as step_<N>.pt + latest.pt, saving
    model + optimizer + scheduler + scaler + step. Auto-resume on restart.

Usage:
    python -m training.sft_train --config config/train_config.json
    python -m training.sft_train --override num_epochs=3 micro_batch_size=2
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
from data_pipeline.dataset_loader import JsonlExampleDataset, train_val_split
from data_pipeline.collator import CausalLMCollator, IGNORE_INDEX
from training.common import (build_optimizer, resolve_amp, count_params_human,
                             CosineScheduler, evaluate)
from utils.seed import set_seed, seed_worker
from utils.logging import TrainLogger
from utils.checkpoint import save_checkpoint, load_checkpoint, load_weights_only


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=P.TRAIN_CONFIG)
    p.add_argument("--section", default="sft")
    p.add_argument("--override", nargs="*", default=[])
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

    data_path = cfg.get("data_path") or P.SFT_DATA
    output_dir = cfg.get("output_dir") or P.SFT_CKPT_DIR
    init_from = cfg.get("init_from") or P.PRETRAIN_LATEST
    model_config_path = P.resolve_repo_path(full_cfg.get("model_config")) or P.MODEL_CONFIG

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = TrainLogger(P.LOG_DIR, run_name="sft", mirror_dir=P.DRIVE_LOG_DIR)
    logger.info(f"device={device} | section='{args.section}'")
    logger.info(f"data={data_path}\noutput={output_dir}\ninit_from={init_from}")
    logger.info(f"effective config: {json.dumps(cfg, ensure_ascii=False)}")

    if not os.path.exists(data_path) or not os.path.exists(model_config_path):
        logger.info(f"FATAL: missing required files: "
                    f"{[p for p in [data_path, model_config_path] if not os.path.exists(p)]}")
        sys.exit(1)

    # ----- model -----
    model_cfg = ModelConfig.from_json(model_config_path)
    model_cfg.max_position_embeddings = max(model_cfg.max_position_embeddings, cfg["max_seq_len"])
    model = FinantalForCausalLM(model_cfg).to(device)
    logger.info(f"model parameters: {count_params_human(model.num_parameters())}")

    # ----- data (95/5 split) + prompt masking -----
    assistant_marker = cfg.get("assistant_marker_ids") if cfg.get("mask_prompt", False) else None
    if assistant_marker:
        logger.info(f"prompt masking ON: loss only after marker {assistant_marker} (▁Assistant :)")
    else:
        logger.info("prompt masking OFF: loss on full sequence (labels as provided)")

    full_ds = JsonlExampleDataset(data_path, max_seq_len=cfg["max_seq_len"], has_labels=True)
    val_ratio = cfg.get("val_ratio", 0.0) or 0.0
    train_ds, val_ds = train_val_split(full_ds, val_ratio, seed=full_cfg.get("seed", 1234))
    collate = CausalLMCollator(pad_token_id=model_cfg.pad_token_id, max_seq_len=cfg["max_seq_len"],
                               assistant_marker=assistant_marker)
    loader = DataLoader(
        train_ds, batch_size=cfg["micro_batch_size"], shuffle=True,
        num_workers=cfg.get("num_workers", 2), collate_fn=collate,
        pin_memory=(device == "cuda"), drop_last=True,
        worker_init_fn=seed_worker, persistent_workers=cfg.get("num_workers", 2) > 0,
    )
    val_loader = (DataLoader(val_ds, batch_size=cfg["micro_batch_size"], shuffle=False,
                             num_workers=0, collate_fn=collate,
                             pin_memory=(device == "cuda"), drop_last=True)
                  if val_ds is not None else None)

    # ----- horizon / schedule -----
    accum = cfg["gradient_accumulation_steps"]
    steps_per_epoch = max(1, len(train_ds) // (cfg["micro_batch_size"] * accum))
    if cfg.get("max_steps", -1) and cfg["max_steps"] > 0:
        max_steps = cfg["max_steps"]
    else:
        max_steps = steps_per_epoch * cfg["num_epochs"]
    warmup_steps = int(cfg.get("warmup_ratio", 0.03) * max_steps)
    logger.info(f"train={len(train_ds):,} | val={len(val_ds) if val_ds else 0:,} | "
                f"steps/epoch={steps_per_epoch} | max_steps={max_steps} | warmup={warmup_steps}")

    # ----- optimizer / amp / scheduler -----
    optimizer = build_optimizer(
        model, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
        betas=(cfg["beta1"], cfg["beta2"]), eps=cfg["eps"],
        use_8bit=cfg.get("use_8bit_optimizer", False), logger=logger,
    )
    amp_dtype, use_scaler = resolve_amp(cfg["precision"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    scheduler = CosineScheduler(optimizer, warmup_steps=warmup_steps, max_steps=max_steps,
                                base_lr=cfg["learning_rate"], min_lr=cfg["min_lr"])
    logger.info(f"precision={cfg['precision']} (amp_dtype={amp_dtype}, grad_scaler={use_scaler})")

    # ----- resume (priority) else init_from pretrained weights -----
    start_step = 0
    resume_from = resolve_resume(cfg, output_dir, logger)
    if resume_from:
        ckpt = load_checkpoint(resume_from, model=model, optimizer=optimizer,
                               scheduler=scheduler, scaler=scaler, map_location=device)
        start_step = ckpt.get("step", 0)
        logger.info(f"resumed SFT from {resume_from} at step {start_step}")
    elif init_from and os.path.exists(init_from):
        info = load_weights_only(init_from, model, map_location=device, strict=False)
        logger.info(f"initialised from pretrained: {init_from} "
                    f"(missing={len(info['missing'])}, unexpected={len(info['unexpected'])})")
    else:
        logger.info(f"WARNING: init_from='{init_from}' not found — SFT from random init.")

    # ----- train loop -----
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = start_step
    micro = 0
    running_loss = 0.0
    t0 = time.time()
    tokens_since = 0
    tokens_per_batch = cfg["micro_batch_size"] * cfg["max_seq_len"]

    done = False
    epoch = 0
    while not done and global_step < max_steps:
        epoch += 1
        logger.info(f"=== epoch {epoch}/{cfg['num_epochs']} ===")
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            # defensive: never supervise pad positions (prompt is already -100 from collator)
            labels = labels.masked_fill(input_ids == model_cfg.pad_token_id, IGNORE_INDEX)

            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                _, loss = model(input_ids, labels=labels)
                loss = loss / accum

            scaler.scale(loss).backward()
            running_loss += loss.item() * accum
            tokens_since += tokens_per_batch
            micro += 1

            if micro % accum == 0:
                grad_norm = None
                if cfg["grad_clip"] and cfg["grad_clip"] > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]).item()

                lr = scheduler.set_step(global_step)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg["log_every"] == 0:
                    dt = time.time() - t0
                    tps = tokens_since / dt if dt > 0 else 0.0
                    logger.log_step(global_step, running_loss / (cfg["log_every"] * accum), lr,
                                    grad_norm=grad_norm, tokens_per_sec=tps, extra={"epoch": epoch})
                    running_loss, tokens_since, t0 = 0.0, 0, time.time()

                if val_loader is not None and global_step % cfg.get("eval_every", 100) == 0:
                    vloss, vppl = evaluate(model, val_loader, device=device, amp_dtype=amp_dtype,
                                           max_batches=cfg.get("eval_max_batches", 50),
                                           pad_token_id=model_cfg.pad_token_id)
                    if vloss is not None:
                        logger.log_eval(global_step, vloss, vppl)
                    t0 = time.time()

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
    logger.info(f"SFT complete at step {global_step}. final checkpoint -> {final}")
    logger.close()


if __name__ == "__main__":
    main()
