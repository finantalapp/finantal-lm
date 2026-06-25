"""Generate colab/chat_simple_colab.ipynb (lean, self-contained). Run: python colab/build_chat_simple_notebook.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_SRC = (ROOT / "models" / "model.py").read_text(encoding="utf-8")
OUT = ROOT / "colab" / "chat_simple_colab.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": t.splitlines(keepends=True)}


CELLS = [
    md("# Finantal LM — Chat (Colab)\n\nMinimal chat UI. Loads `latest.pt` from Drive and runs inference. "
       "Run the cells in order.\n"),

    code("# 1) deps + mount\n"
         "!pip -q install gradio sentencepiece\n"
         "from google.colab import drive\n"
         "drive.mount('/content/drive')"),

    code("# 2) paths  (EDIT to your locations)\n"
         "import os, glob\n"
         "LATEST    = '/content/drive/MyDrive/pretrain/checkpoints/latest.pt'\n"
         "TOKENIZER = '/content/drive/MyDrive/finantal_data/tokenizer/finantal_tokenizer.model'\n"
         "if not os.path.exists(TOKENIZER):                      # quick fallback search\n"
         "    h = glob.glob('/content/drive/MyDrive/**/*.model', recursive=True)\n"
         "    TOKENIZER = h[0] if h else TOKENIZER\n"
         "assert os.path.exists(LATEST), f'not found: {LATEST}'\n"
         "assert os.path.exists(TOKENIZER), f'not found: {TOKENIZER}'\n"
         "print('checkpoint:', LATEST)\nprint('tokenizer :', TOKENIZER)"),

    code("# 3) model definition (required: the checkpoint stores weights only)\n" + MODEL_SRC),

    code("# 4) load model + tokenizer, then launch the chat UI\n"
         "import torch, torch.nn.functional as F, sentencepiece as spm, gradio as gr\n"
         "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
         "\n"
         "ck = torch.load(LATEST, map_location=DEVICE, weights_only=False)\n"
         "cfg = ModelConfig(**{k: v for k, v in ck['model_config'].items() if k in ModelConfig.__dataclass_fields__})\n"
         "cfg.use_gradient_checkpointing = False\n"
         "model = FinantalForCausalLM(cfg); model.load_state_dict(ck['model']); model.to(DEVICE).eval()\n"
         "SP = spm.SentencePieceProcessor(model_file=TOKENIZER)\n"
         "print(f'loaded | step={ck.get(\"step\")} | params={model.num_parameters():,} | {DEVICE}')\n"
         "\n"
         "@torch.no_grad()\n"
         "def generate(prompt, max_new_tokens=160, temperature=0.2, top_p=0.9, repetition_penalty=1.3, do_sample=True):\n"
         "    # do NOT prepend BOS: SFT data starts with '▁User'(26054), never BOS -> match training\n"
         "    ids = SP.encode(prompt, out_type=int)\n"
         "    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)\n"
         "    new = []\n"
         "    for _ in range(int(max_new_tokens)):\n"
         "        logits, _ = model(x[:, -cfg.max_position_embeddings:])\n"
         "        logits = logits[:, -1, :].float()\n"
         "        # repetition penalty — penalise tokens already in context\n"
         "        if repetition_penalty and repetition_penalty != 1.0:\n"
         "            for tid in set(x[0].tolist()):\n"
         "                if logits[0, tid] < 0: logits[0, tid] *= repetition_penalty\n"
         "                else: logits[0, tid] /= repetition_penalty\n"
         "        if not do_sample:\n"
         "            nxt = logits.argmax(dim=-1, keepdim=True)   # greedy (temperature 0)\n"
         "        else:\n"
         "            logits = logits / max(float(temperature), 1e-5)\n"
         "            if top_p and top_p < 1.0:\n"
         "                sl, si = torch.sort(logits, descending=True)\n"
         "                cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)\n"
         "                rm = cum > top_p; rm[..., 1:] = rm[..., :-1].clone(); rm[..., 0] = False\n"
         "                logits = logits.masked_fill(rm.scatter(1, si, rm), float('-inf'))\n"
         "            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)\n"
         "        tok = int(nxt.item()); x = torch.cat([x, nxt], dim=1)\n"
         "        if tok == cfg.eos_token_id: break\n"
         "        new.append(tok)\n"
         "        # stop if the model starts a new User turn\n"
         "        decoded_so_far = SP.decode(new)\n"
         "        if 'User:' in decoded_so_far and len(new) > 5:\n"
         "            return decoded_so_far.split('User:')[0].strip()\n"
         "    return SP.decode(new).strip()\n"
         "\n"
         "# STATELESS: history is intentionally ignored.\n"
         "# The model is single-turn trained — feeding history causes context bleed.\n"
         "# Each message is wrapped in the exact SFT template: 'User: <Q> Assistant:'\n"
         "def respond(message, history, max_new_tokens, temperature, top_p, repetition_penalty, do_sample):\n"
         "    prompt = f'User: {message.strip()} Assistant:'\n"
         "    return generate(prompt, max_new_tokens, temperature, top_p, repetition_penalty, do_sample)\n"
         "\n"
         "gr.ChatInterface(\n"
         "    fn=respond, type='messages', title='Finantal LM',\n"
         "    description='كل سؤال يُعالَج بشكل مستقل (Stateless). النموذج لا يرى المحادثة السابقة.',\n"
         "    additional_inputs=[\n"
         "        gr.Slider(16, 512, value=160, step=8, label='max_new_tokens'),\n"
         "        gr.Slider(0.1, 1.5, value=0.2, step=0.05, label='temperature (low=focused)'),\n"
         "        gr.Slider(0.1, 1.0, value=0.9, step=0.05, label='top_p'),\n"
         "        gr.Slider(1.0, 2.0, value=1.3, step=0.05, label='repetition_penalty'),\n"
         "        gr.Checkbox(value=True, label='do_sample (أزِل الصح + 🔄 Retry = greedy/حرارة 0)'),\n"
         "    ],\n"
         "    additional_inputs_accordion=gr.Accordion('⚙️ إعدادات التوليد', open=False),\n"
         ").launch(share=True)"),
]

nb = {"cells": CELLS,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT, "| cells:", len(CELLS))
