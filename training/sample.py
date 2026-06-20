"""
Quick generation sanity check for a trained checkpoint.

    python -m training.sample --ckpt <ckpt or 'sft'/'pretrain'> --prompt "ما هو التمويل؟"

--ckpt accepts an explicit path, or the shortcuts 'sft' / 'pretrain' which resolve
to the latest checkpoint on Drive via config/paths.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import paths as P
from models.model import FinantalForCausalLM, ModelConfig
from utils.tokenizer import SPTokenizer


def resolve_ckpt(arg: str) -> str:
    return {"sft": P.SFT_LATEST, "pretrain": P.PRETRAIN_LATEST}.get(arg, arg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path, or 'sft' / 'pretrain'")
    p.add_argument("--tokenizer", default=P.TOKENIZER_MODEL)
    p.add_argument("--prompt", default="")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(resolve_ckpt(args.ckpt), map_location=device, weights_only=False)
    cfg = ModelConfig(**{k: v for k, v in ckpt["model_config"].items()
                         if k in ModelConfig.__dataclass_fields__})
    model = FinantalForCausalLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tok = SPTokenizer(args.tokenizer)
    ids = tok.encode(args.prompt, add_bos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
