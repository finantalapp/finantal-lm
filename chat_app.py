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
def generate(prompt: str, max_new_tokens: int = 200,
             temperature: float = 0.8, top_p: float = 0.9,
             repetition_penalty: float = 1.3) -> str:
    ids = SP.encode(prompt, out_type=int)
    if SP.bos_id() >= 0:
        ids = [SP.bos_id()] + ids
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    new: list[int] = []
    for _ in range(int(max_new_tokens)):
        logits, _ = model(x[:, -cfg.max_position_embeddings:])
        logits = logits[:, -1, :].float()

        # repetition penalty: down-weight tokens already in the full context
        if repetition_penalty and repetition_penalty != 1.0:
            for tok_id in set(x[0].tolist()):
                if logits[0, tok_id] < 0:
                    logits[0, tok_id] *= repetition_penalty
                else:
                    logits[0, tok_id] /= repetition_penalty

        logits = logits / max(float(temperature), 1e-5)

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
        # stop if the model tries to start a new "User:" turn
        decoded_so_far = SP.decode(new)
        if "User:" in decoded_so_far and len(new) > 5:
            return decoded_so_far.split("User:")[0].strip()

    return SP.decode(new).strip()


# ── chat (STATELESS — history is intentionally ignored) ───────────────────────
# The model is small and single-turn trained. Feeding accumulated history
# saturates the attention window and causes hallucination / context bleed.
# Each message is sent as a fully self-contained SFT-template prompt.
def respond(message: str, history,          # history arg kept for Gradio API compat
            max_new_tokens: int, temperature: float,
            top_p: float, repetition_penalty: float) -> str:
    # Exact SFT training template: "User: <question> Assistant:"
    # (space before Assistant — matches what the tokenizer saw during training)
    prompt = f"User: {message.strip()} Assistant:"
    return generate(prompt, max_new_tokens, temperature, top_p, repetition_penalty)


demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="Finantal LM",
    description=(
        "كل سؤال يُعالَج بشكل مستقل (Stateless). "
        "النموذج لا يرى المحادثة السابقة — فقط سؤالك الحالي."
    ),
    additional_inputs=[
        gr.Slider(16, 512, value=200, step=8,
                  label="max_new_tokens"),
        gr.Slider(0.1, 2.0, value=0.8, step=0.05,
                  label="temperature"),
        gr.Slider(0.1, 1.0, value=0.9, step=0.05,
                  label="top_p"),
        gr.Slider(1.0, 2.0, value=1.3, step=0.05,
                  label="repetition_penalty"),
    ],
    additional_inputs_accordion=gr.Accordion("⚙️ إعدادات التوليد", open=False),
)

if __name__ == "__main__":
    demo.launch(share=True)
