"""Generate the Colab notebooks under colab/ as valid .ipynb JSON (stdlib only).

    python scripts/build_notebooks.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "colab"
OUT.mkdir(exist_ok=True)


def nb(cells):
    return {"cells": cells,
            "metadata": {"accelerator": "GPU",
                         "colab": {"provenance": [], "gpuType": "T4"},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": t.splitlines(keepends=True)}


# Shared snippets -----------------------------------------------------------
GPU = """\
# Check the GPU (expect a Tesla T4)
!nvidia-smi -L
import torch
print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"""

MOUNT = """\
# Mount Google Drive (data + tokenizer + checkpoints live here)
from google.colab import drive
drive.mount('/content/drive')"""

CLONE = """\
# Clone the code repo from GitHub
import os
REPO_URL = 'https://github.com/<YOUR_USERNAME>/finantal-lm.git'   # <-- EDIT
REPO_DIR = '/content/finantal-lm'
if not os.path.isdir(REPO_DIR):
    !git clone $REPO_URL $REPO_DIR
else:
    !cd $REPO_DIR && git pull --ff-only
%cd $REPO_DIR"""

DEPS = """\
# Install dependencies (torch already on Colab — not reinstalled)
!pip -q install -r requirements.txt"""

ENV = """\
# Point the code at the data on Drive. The default already matches this path,
# but we set it explicitly so it's obvious and overridable.
import os
os.environ['FINANTAL_DATA_ROOT'] = '/content/drive/MyDrive/finantal_data'
# Make the repo importable from notebook cells too
import sys; sys.path.insert(0, '/content/finantal-lm')
from config import paths as P
print(P.summary())"""

VERIFY = """\
# Verify the required assets exist on Drive before training
from config import paths as P
missing = P.verify_assets(require_tokenizer=True, require_pretrain=True, require_sft=True)
if missing:
    print('MISSING — upload these to Drive:')
    for m in missing: print('  -', m)
else:
    print('All assets present on Drive ✓')
    import json
    if os.path.exists(P.DATASET_STATS):
        print(json.dumps(json.load(open(P.DATASET_STATS)), ensure_ascii=False, indent=2))"""


def setup_nb():
    return nb([
        md("# Finantal-LM — Colab bootstrap (one-time setup)\n\n"
           "Run this once per Colab session. It mounts Drive, clones the code repo, "
           "installs dependencies, sets the data paths, and verifies the data is present.\n\n"
           "**Expected Drive layout** (`MyDrive/finantal_data/`):\n"
           "```\n"
           "finantal_data/\n"
           "├── data/        pretrain_tokenized.jsonl, sft_tokenized.jsonl\n"
           "├── tokenizer/   finantal_tokenizer.model, .vocab\n"
           "└── checkpoints/ pretrain/, sft/\n"
           "```\n"),
        code(GPU), code(MOUNT), code(CLONE), code(DEPS), code(ENV), code(VERIFY),
        md("Setup done. Now open **run_pretrain_colab.ipynb** or **run_sft_colab.ipynb**."),
    ])


def pretrain_nb():
    return nb([
        md("# Finantal-LM — Pretraining (PyTorch from scratch, T4)\n\n"
           "Self-contained: mounts Drive, clones the repo, installs deps, then trains. "
           "Checkpoints are written to Drive (`finantal_data/checkpoints/pretrain/`) and "
           "training **auto-resumes** from `latest.pt` if you reconnect.\n"),
        code(GPU), code(MOUNT), code(CLONE), code(DEPS), code(ENV), code(VERIFY),
        md("## Train\n"
           "Reads `config/train_config.json` → `pretrain`. Override any field with "
           "`--override key=value`. On CUDA OOM: lower `micro_batch_size`, raise "
           "`gradient_accumulation_steps` to keep the effective batch constant.\n"),
        code("!python -m training.pretrain --config config/train_config.json \\\n"
             "    --override micro_batch_size=8 gradient_accumulation_steps=16 max_steps=8000"),
        md("### Safer low-memory setting\n"),
        code("# !python -m training.pretrain --override micro_batch_size=4 "
             "gradient_accumulation_steps=32 max_seq_len=1024"),
        md("### Resume manually (auto-resume is on by default)\n"),
        code("# !python -m training.pretrain --override "
             "resume_from=/content/drive/MyDrive/finantal_data/checkpoints/pretrain/latest.pt"),
        md("## Loss / perplexity curve\n"),
        code("import json, matplotlib.pyplot as plt\n"
             "rows = [json.loads(l) for l in open('logs/pretrain_metrics.jsonl')]\n"
             "plt.figure(figsize=(8,4)); plt.plot([r['step'] for r in rows], [r['loss'] for r in rows])\n"
             "plt.xlabel('step'); plt.ylabel('loss'); plt.grid(True); plt.title('Pretrain loss'); plt.show()"),
        md("## Generation sanity check\n"),
        code("!python -m training.sample --ckpt pretrain --prompt \"التمويل هو\" --max_new_tokens 80"),
    ])


def sft_nb():
    return nb([
        md("# Finantal-LM — Supervised Fine-Tuning (SFT)\n\n"
           "Loads the pretrained checkpoint from Drive (`init_from` defaults to "
           "`checkpoints/pretrain/latest.pt`) and fine-tunes on the SFT data. "
           "Auto-resumes from `checkpoints/sft/latest.pt` if present.\n"),
        code(GPU), code(MOUNT), code(CLONE), code(DEPS), code(ENV),
        code("# Confirm the pretrained checkpoint exists on Drive\n"
             "from config import paths as P; import os\n"
             "assert os.path.exists(P.PRETRAIN_LATEST), f'Run pretraining first: {P.PRETRAIN_LATEST} missing'\n"
             "print('pretrained checkpoint OK:', P.PRETRAIN_LATEST)"),
        md("## Train (SFT)\n"),
        code("!python -m training.sft_train --config config/train_config.json \\\n"
             "    --override num_epochs=3"),
        md("### Low-memory variant\n"),
        code("# !python -m training.sft_train --override micro_batch_size=2 gradient_accumulation_steps=16"),
        md("## Loss curve\n"),
        code("import json, matplotlib.pyplot as plt\n"
             "rows = [json.loads(l) for l in open('logs/sft_metrics.jsonl')]\n"
             "plt.figure(figsize=(8,4)); plt.plot([r['step'] for r in rows], [r['loss'] for r in rows])\n"
             "plt.xlabel('step'); plt.ylabel('loss'); plt.grid(True); plt.title('SFT loss'); plt.show()"),
        md("## Chat sanity check\n"),
        code("!python -m training.sample --ckpt sft --prompt \"ما هي الميزانية؟\" "
             "--max_new_tokens 120 --temperature 0.7"),
    ])


for fname, builder in [("setup_colab.ipynb", setup_nb),
                       ("run_pretrain_colab.ipynb", pretrain_nb),
                       ("run_sft_colab.ipynb", sft_nb)]:
    with open(OUT / fname, "w", encoding="utf-8") as f:
        json.dump(builder(), f, ensure_ascii=False, indent=1)
    print("wrote", OUT / fname)
