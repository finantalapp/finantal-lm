"""
Finantal LM — minimal chat UI (inference only).

Loads weights directly from a single checkpoint (latest.pt) and serves a Gradio
chat. Run locally:

    python chat_app.py
    # override paths if needed:
    LATEST=/path/to/latest.pt TOKENIZER=/path/to/finantal_tokenizer.model python chat_app.py
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import sentencepiece as spm
import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models.model import FinantalForCausalLM, ModelConfig          # model definition (weights-only ckpt)
from config import paths as P                                       # default paths

LATEST = os.environ.get("LATEST", P.PRETRAIN_LATEST)                # or set to your sft latest.pt
TOKENIZER = os.environ.get("TOKENIZER", P.TOKENIZER_MODEL)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── load model (direct, known keys) ───────────────────────────────────────────
ck = torch.load(LATEST, map_location=DEVICE, weights_only=False)
cfg = ModelConfig(**{k: v for k, v in ck["model_config"].items()
                     if k in ModelConfig.__dataclass_fields__})
cfg.use_gradient_checkpointing = False
model = FinantalForCausalLM(cfg)
model.load_state_dict(ck["model"])
model.to(DEVICE).eval()
SP = spm.SentencePieceProcessor(model_file=TOKENIZER)
print(f"loaded {LATEST} | step={ck.get('step')} | params={model.num_parameters():,} | {DEVICE}")


# ── generation: tokenize -> generate -> decode (new text only) ────────────────
@torch.no_grad()
def generate(prompt, max_new_tokens=200, temperature=0.8, top_p=0.9):
    ids = SP.encode(prompt, out_type=int)
    if SP.bos_id() >= 0:
        ids = [SP.bos_id()] + ids
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    new = []
    for _ in range(int(max_new_tokens)):
        logits, _ = model(x[:, -cfg.max_position_embeddings:])
        logits = logits[:, -1, :].float() / max(float(temperature), 1e-5)
        if top_p and top_p < 1.0:
            s_logits, s_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(s_logits, dim=-1), dim=-1)
            rm = cum > top_p
            rm[..., 1:] = rm[..., :-1].clone(); rm[..., 0] = False
            logits = logits.masked_fill(rm.scatter(1, s_idx, rm), float("-inf"))
        nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        tok = int(nxt.item())
        x = torch.cat([x, nxt], dim=1)
        if tok == cfg.eos_token_id:
            break
        new.append(tok)
        if "\nUser:" in SP.decode(new):
            return SP.decode(new).split("\nUser:")[0].strip()
    return SP.decode(new).strip()


# ── chat ──────────────────────────────────────────────────────────────────────
def respond(message, history, max_new_tokens, temperature, top_p):
    prompt = ""
    for m in history:
        prompt += ("User: " if m["role"] == "user" else "Assistant: ") + m["content"] + "\n"
    prompt += f"User: {message}\nAssistant:"
    return generate(prompt, max_new_tokens, temperature, top_p)


demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="Finantal LM",
    additional_inputs=[
        gr.Slider(16, 1024, value=200, step=8, label="max_new_tokens"),
        gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="temperature"),
        gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p"),
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)
