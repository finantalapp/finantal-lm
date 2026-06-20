# Finantal-LM

Train a **decoder-only language model from scratch** (PyTorch, no HuggingFace
Trainer, no pretrained weights) — built to run on a free **Google Colab T4**, with
**code on GitHub** and **data + tokenizer + checkpoints on Google Drive**.

```
finantal-lm/                      ← GitHub (code only)
├── training/        pretrain.py · sft_train.py · common.py · sample.py
├── models/          model.py  (RMSNorm + RoPE + GQA + SwiGLU, from scratch)
├── data_pipeline/   dataset_loader.py · collator.py
├── utils/           logging.py · seed.py · checkpoint.py · tokenizer.py
├── config/          paths.py · model_config.json · train_config.json
├── scripts/         run_pretrain.sh · run_sft.sh · build_dataset_stats.py · build_notebooks.py
├── colab/           setup_colab.ipynb · run_pretrain_colab.ipynb · run_sft_colab.ipynb
├── requirements.txt · README.md · .gitignore
```

Heavy assets are **never** committed (see `.gitignore`). They live on Drive:

```
MyDrive/finantal_data/            ← Google Drive (data + weights)
├── data/         pretrain_tokenized.jsonl · sft_tokenized.jsonl · dataset_stats.json
├── tokenizer/    finantal_tokenizer.model · finantal_tokenizer.vocab
└── checkpoints/  pretrain/ · sft/      (written here during training)
```

---

## How paths work (no hard-coding)

Every path is resolved through [`config/paths.py`](config/paths.py) from environment
variables, defaulting to the Drive layout above. The only one you usually set is:

```bash
FINANTAL_DATA_ROOT=/content/drive/MyDrive/finantal_data   # Colab default
# Local dev mirror:
FINANTAL_DATA_ROOT=D:/finantal_data                       # Windows
FINANTAL_DATA_ROOT=/data/finantal                         # Linux
```

Other overridable vars: `DATA_DIR`, `TOKENIZER_DIR`, `CHECKPOINT_DIR`, `LOG_DIR`,
`DRIVE_LOG_DIR`, `PRETRAIN_DATA`, `SFT_DATA`, `TOKENIZER_MODEL`.
Inspect the resolved paths anytime: `python -m config.paths`.

---

## Quick start on Colab (one-click)

1. **Push this repo to GitHub** (code only — `.gitignore` keeps data out).
2. **Upload your data to Drive** as `MyDrive/finantal_data/` (the layout above).
3. Open a notebook from `colab/` in Google Colab (Runtime → **GPU / T4**):
   - **`setup_colab.ipynb`** — mounts Drive, clones the repo, installs deps, verifies data.
   - **`run_pretrain_colab.ipynb`** — runs pretraining end-to-end (self-contained).
   - **`run_sft_colab.ipynb`** — runs SFT from the pretrained checkpoint.
4. In the first cell of each, set `REPO_URL` to your GitHub URL. Run cells top to bottom.

Checkpoints land in `MyDrive/finantal_data/checkpoints/`, so they survive disconnects.

---

## Training

```bash
# Pretraining
python -m training.pretrain --config config/train_config.json

# SFT (auto-loads checkpoints/pretrain/latest.pt from Drive)
python -m training.sft_train --config config/train_config.json

# Override any hyper-parameter without editing JSON:
python -m training.pretrain --override micro_batch_size=4 gradient_accumulation_steps=32 max_steps=5000
```

Or via the shell launchers (set `FINANTAL_DATA_ROOT` first):

```bash
FINANTAL_DATA_ROOT=/data/finantal bash scripts/run_pretrain.sh max_steps=8000
FINANTAL_DATA_ROOT=/data/finantal bash scripts/run_sft.sh num_epochs=3
```

### Resume (automatic)
`auto_resume` is `true` by default: if `checkpoints/<stage>/latest.pt` exists, training
continues from it (step, optimizer, scaler all restored). Force a specific checkpoint
with `--override resume_from=/path/to/checkpoint_step1234.pt`, or start fresh with
`--override auto_resume=false`.

### Logging
Loss / perplexity / LR / grad-norm are written to `logs/<stage>_metrics.jsonl` **locally**
and mirrored to `MyDrive/finantal_data/logs/` on **Drive**. The notebooks plot the curve.

---

## Model & config

[`config/model_config.json`](config/model_config.json) defines the architecture
(default ≈ **300M params**, T4-friendly with fp16). It documents 1B / 3B presets — the
larger ones need `use_8bit_optimizer: true` in `train_config.json`. The model itself is
in [`models/model.py`](models/model.py): RMSNorm pre-norm, RoPE, Grouped-Query Attention
via `scaled_dot_product_attention`, SwiGLU, weight tying, optional gradient checkpointing.

Training features (all hand-written): fp16/bf16 mixed precision, gradient accumulation,
gradient clipping, cosine LR + warmup, packed-sequence pretraining, checkpoint rotation,
8-bit Adam option.

---

## Out-of-memory tips (T4 = 16 GB)
- Lower `micro_batch_size`, raise `gradient_accumulation_steps` (effective batch unchanged).
- Keep `use_gradient_checkpointing: true` in `model_config.json`.
- For ≥1B models set `use_8bit_optimizer: true` (needs `bitsandbytes`) + `micro_batch_size: 1`.
- Reduce `max_seq_len` (e.g. 512) to halve activation memory.

## Regenerate dataset stats
```bash
python scripts/build_dataset_stats.py     # writes <DATA_DIR>/dataset_stats.json
```

## Local development (no Colab)
```bash
git clone https://github.com/<YOUR_USERNAME>/finantal-lm.git && cd finantal-lm
pip install -r requirements.txt
export FINANTAL_DATA_ROOT=/path/to/finantal_data   # your local data mirror
python -m training.pretrain
```
