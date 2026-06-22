"""Generate colab/chat_ui_colab.ipynb (valid .ipynb JSON). Run: python colab/build_chat_ui_notebook.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_SRC = (ROOT / "models" / "model.py").read_text(encoding="utf-8")
OUT = ROOT / "colab" / "chat_ui_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": t.splitlines(keepends=True)}


CELLS = []

CELLS.append(md(
"# FinantalLM — Chat UI (Gradio) for Google Colab\n"
"\n"
"A ChatGPT-style interface to test your model **while it trains**, loading checkpoints\n"
"straight from Google Drive. Run the cells top to bottom.\n"
"\n"
"- Cell 4 **inspects** your `.pt` file and prints exactly what it contains (no assumptions).\n"
"- The model architecture is rebuilt automatically from the `model_config` stored inside the checkpoint.\n"
"- Switch between `latest.pt` / `step_2700.pt` live from the UI.\n"
))

# ── 1. install ──
CELLS.append(code(
"# 1) Dependencies (torch is already on Colab)\n"
"!pip -q install gradio sentencepiece\n"
"import torch, gradio as gr\n"
"print('torch', torch.__version__, '| gradio', gr.__version__, '| CUDA', torch.cuda.is_available())"
))

# ── 2. mount ──
CELLS.append(code(
"# 2) Mount Google Drive\n"
"from google.colab import drive\n"
"drive.mount('/content/drive')"
))

# ── 3. config / paths ──
CELLS.append(code(
"# 3) Paths — EDIT these if your layout differs\n"
"import os, glob\n"
"\n"
"CKPT_DIR = '/content/drive/MyDrive/pretrain/checkpoints'   # folder with latest.pt / step_*.pt\n"
"\n"
"# Tokenizer (SentencePiece .model) is NOT inside the checkpoint — locate it on Drive.\n"
"TOKENIZER_PATH = ''  # leave '' to auto-search, or set the full path explicitly\n"
"TOKENIZER_CANDIDATES = [\n"
"    '/content/drive/MyDrive/finantal_data/tokenizer/finantal_tokenizer.model',\n"
"    '/content/drive/MyDrive/pretrain/tokenizer/finantal_tokenizer.model',\n"
"    '/content/drive/MyDrive/tokenizer/finantal_tokenizer.model',\n"
"    '/content/drive/MyDrive/pretrain/finantal_tokenizer.model',\n"
"]\n"
"\n"
"# --- checkpoints found ---\n"
"ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, '*.pt')))\n"
"assert ckpt_files, f'No .pt files found in {CKPT_DIR}. Fix CKPT_DIR.'\n"
"print('Checkpoints found:')\n"
"for p in ckpt_files: print('  -', os.path.basename(p), f'({os.path.getsize(p)/1e6:.1f} MB)')\n"
"\n"
"# --- tokenizer auto-search ---\n"
"if not TOKENIZER_PATH:\n"
"    for c in TOKENIZER_CANDIDATES:\n"
"        if os.path.exists(c):\n"
"            TOKENIZER_PATH = c; break\n"
"if not TOKENIZER_PATH:\n"
"    hits = glob.glob('/content/drive/MyDrive/**/*.model', recursive=True)\n"
"    TOKENIZER_PATH = hits[0] if hits else ''\n"
"print('\\nTokenizer:', TOKENIZER_PATH or 'NOT FOUND -- set TOKENIZER_PATH manually')\n"
"assert TOKENIZER_PATH and os.path.exists(TOKENIZER_PATH), 'Set TOKENIZER_PATH to your finantal_tokenizer.model'"
))

# ── 4. INSPECT checkpoint ──
CELLS.append(code(
"# 4) INSPECT the checkpoint — see exactly what is stored (no assumptions)\n"
"import torch\n"
"_sample = os.path.join(CKPT_DIR, 'latest.pt')\n"
"if not os.path.exists(_sample): _sample = ckpt_files[0]\n"
"print('Inspecting:', _sample, '\\n')\n"
"\n"
"_ck = torch.load(_sample, map_location='cpu', weights_only=False)\n"
"print('type:', type(_ck).__name__)\n"
"if isinstance(_ck, dict):\n"
"    for k, v in _ck.items():\n"
"        if isinstance(v, dict):\n"
"            n_t = sum(1 for x in v.values() if torch.is_tensor(x))\n"
"            print(f'  {k:14s}: dict ({len(v)} entries, {n_t} tensors)')\n"
"        elif torch.is_tensor(v):\n"
"            print(f'  {k:14s}: tensor {tuple(v.shape)}')\n"
"        else:\n"
"            print(f'  {k:14s}: {type(v).__name__} = {str(v)[:80]}')\n"
"    if isinstance(_ck.get('model_config'), dict):\n"
"        print('\\nmodel_config:')\n"
"        for k, v in _ck['model_config'].items(): print(f'    {k} = {v}')\n"
"else:\n"
"    print('Checkpoint is a bare state_dict (no wrapper).')"
))

# ── 5. model architecture (verbatim from models/model.py) ──
CELLS.append(code(
"# 5) Model architecture (identical to the training code so weights load exactly)\n"
+ MODEL_SRC
))

# ── 6. loaders ──
CELLS.append(code(
"# 6) Build model from a checkpoint + load tokenizer (robust key detection)\n"
"import sentencepiece as spm\n"
"\n"
"DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
"\n"
"def _find_state_dict(ck):\n"
"    if isinstance(ck, dict):\n"
"        for k in ('model', 'model_state_dict', 'state_dict'):\n"
"            if isinstance(ck.get(k), dict):\n"
"                return ck[k]\n"
"        if all(torch.is_tensor(v) for v in ck.values()):\n"
"            return ck\n"
"    raise ValueError('Could not find a model state_dict in this checkpoint.')\n"
"\n"
"def _find_config(ck, state):\n"
"    if isinstance(ck, dict):\n"
"        for k in ('model_config', 'config'):\n"
"            if isinstance(ck.get(k), dict):\n"
"                return ck[k]\n"
"    # fallback: infer the essentials from tensor shapes\n"
"    emb = state.get('embed_tokens.weight')\n"
"    n_layers = 1 + max(int(k.split('.')[1]) for k in state if k.startswith('layers.'))\n"
"    print('[warn] no model_config in checkpoint -> inferring vocab/hidden/layers from weights')\n"
"    return {'vocab_size': emb.shape[0], 'hidden_size': emb.shape[1], 'num_hidden_layers': n_layers}\n"
"\n"
"_MODEL_CACHE = {}\n"
"\n"
"def load_model(ckpt_path):\n"
"    if ckpt_path in _MODEL_CACHE:\n"
"        return _MODEL_CACHE[ckpt_path]\n"
"    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)\n"
"    state = _find_state_dict(ck)\n"
"    cfg_d = _find_config(ck, state)\n"
"    cfg = ModelConfig(**{k: v for k, v in cfg_d.items() if k in ModelConfig.__dataclass_fields__})\n"
"    cfg.use_gradient_checkpointing = False           # inference only\n"
"    model = FinantalForCausalLM(cfg)\n"
"    missing, unexpected = model.load_state_dict(state, strict=False)\n"
"    if missing:    print('[load] missing keys:', missing[:6], '...' if len(missing) > 6 else '')\n"
"    if unexpected: print('[load] unexpected keys:', unexpected[:6], '...' if len(unexpected) > 6 else '')\n"
"    model.to(DEVICE).eval()\n"
"    step = ck.get('step') if isinstance(ck, dict) else None\n"
"    print(f'[load] {os.path.basename(ckpt_path)} | step={step} | params={model.num_parameters():,} | {DEVICE}')\n"
"    _MODEL_CACHE[ckpt_path] = (model, cfg)\n"
"    return model, cfg\n"
"\n"
"# tokenizer\n"
"SP = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)\n"
"print('tokenizer vocab:', SP.get_piece_size(), '| bos', SP.bos_id(), '| eos', SP.eos_id())\n"
"\n"
"# warm-load the default checkpoint\n"
"_default = os.path.join(CKPT_DIR, 'latest.pt')\n"
"if not os.path.exists(_default): _default = ckpt_files[0]\n"
"load_model(_default)"
))

# ── 7. generation ──
CELLS.append(code(
"# 7) Sampling generation (top_p / top_k / repetition_penalty / do_sample),\n"
"#    returns ONLY the newly generated text.\n"
"import torch.nn.functional as F\n"
"\n"
"@torch.no_grad()\n"
"def generate(model, cfg, prompt, max_new_tokens=200, temperature=0.8, top_p=0.9,\n"
"             top_k=50, repetition_penalty=1.15, do_sample=True, stop_str='\\nUser:'):\n"
"    ids = SP.encode(prompt, out_type=int)\n"
"    if SP.bos_id() >= 0:\n"
"        ids = [SP.bos_id()] + ids\n"
"    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)\n"
"    max_ctx = cfg.max_position_embeddings\n"
"    eos_id = cfg.eos_token_id\n"
"    new_tokens = []\n"
"    for _ in range(int(max_new_tokens)):\n"
"        logits, _ = model(x[:, -max_ctx:])\n"
"        logits = logits[:, -1, :].float()\n"
"        # repetition penalty over tokens already in the sequence\n"
"        if repetition_penalty and repetition_penalty != 1.0:\n"
"            for t in set(x[0].tolist()):\n"
"                logits[0, t] = logits[0, t] * repetition_penalty if logits[0, t] < 0 else logits[0, t] / repetition_penalty\n"
"        if do_sample:\n"
"            logits = logits / max(float(temperature), 1e-5)\n"
"            if top_k and top_k > 0:\n"
"                v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))\n"
"                logits[logits < v[:, [-1]]] = float('-inf')\n"
"            if top_p and top_p < 1.0:\n"
"                s_logits, s_idx = torch.sort(logits, descending=True)\n"
"                cum = torch.cumsum(F.softmax(s_logits, dim=-1), dim=-1)\n"
"                remove = cum > top_p\n"
"                remove[..., 1:] = remove[..., :-1].clone(); remove[..., 0] = False\n"
"                logits = logits.masked_fill(remove.scatter(1, s_idx, remove), float('-inf'))\n"
"            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)\n"
"        else:\n"
"            nxt = logits.argmax(dim=-1, keepdim=True)\n"
"        tok = int(nxt.item())\n"
"        x = torch.cat([x, nxt], dim=1)\n"
"        if eos_id is not None and tok == eos_id:\n"
"            break\n"
"        new_tokens.append(tok)\n"
"        if stop_str and stop_str in SP.decode(new_tokens):\n"
"            return SP.decode(new_tokens).split(stop_str)[0].strip()\n"
"    return SP.decode(new_tokens).strip()"
))

# ── 8. gradio chat ──
CELLS.append(code(
"# 8) Chat plumbing + Gradio UI\n"
"def _history_to_prompt(history, message):\n"
"    # works with both gradio 'messages' (list of dicts) and 'tuples' formats\n"
"    p = ''\n"
"    if history and isinstance(history[0], dict):\n"
"        for m in history:\n"
"            role = m.get('role'); c = m.get('content', '')\n"
"            if role == 'user':      p += f'User: {c}\\n'\n"
"            elif role == 'assistant': p += f'Assistant: {c}\\n'\n"
"    else:\n"
"        for u, a in (history or []):\n"
"            if u:            p += f'User: {u}\\n'\n"
"            if a is not None: p += f'Assistant: {a}\\n'\n"
"    p += f'User: {message}\\nAssistant:'\n"
"    return p\n"
"\n"
"def respond(message, history, checkpoint, max_new_tokens, temperature, top_p,\n"
"            top_k, repetition_penalty, do_sample):\n"
"    model, cfg = load_model(os.path.join(CKPT_DIR, checkpoint))\n"
"    prompt = _history_to_prompt(history, message)\n"
"    return generate(model, cfg, prompt, max_new_tokens=max_new_tokens, temperature=temperature,\n"
"                    top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, do_sample=do_sample)\n"
"\n"
"_choices = [os.path.basename(p) for p in ckpt_files]\n"
"_default_choice = 'latest.pt' if 'latest.pt' in _choices else _choices[0]\n"
"\n"
"demo = gr.ChatInterface(\n"
"    fn=respond,\n"
"    type='messages',\n"
"    title='FinantalLM',\n"
"    description='Chat with your financial LM checkpoints (loaded live from Google Drive).',\n"
"    additional_inputs=[\n"
"        gr.Dropdown(choices=_choices, value=_default_choice, label='Checkpoint'),\n"
"        gr.Slider(16, 1024, value=200, step=8, label='max_new_tokens'),\n"
"        gr.Slider(0.1, 2.0, value=0.8, step=0.05, label='temperature'),\n"
"        gr.Slider(0.1, 1.0, value=0.9, step=0.05, label='top_p'),\n"
"        gr.Slider(0, 200, value=50, step=1, label='top_k (0 = off)'),\n"
"        gr.Slider(1.0, 2.0, value=1.15, step=0.05, label='repetition_penalty'),\n"
"        gr.Checkbox(value=True, label='do_sample'),\n"
"    ],\n"
"    additional_inputs_accordion=gr.Accordion('Generation settings', open=True),\n"
")\n"
"\n"
"demo.launch(share=True, debug=False)"
))

nb = {"cells": CELLS,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT, "| cells:", len(CELLS))
