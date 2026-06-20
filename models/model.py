"""
Finantal decoder-only language model — implemented from scratch in PyTorch.

Architecture: Llama-style causal transformer
  - Token embedding (optionally tied to the output projection)
  - Pre-norm RMSNorm
  - Rotary Position Embeddings (RoPE)
  - Grouped-Query Attention (GQA) using F.scaled_dot_product_attention
  - SwiGLU feed-forward
  - Optional gradient checkpointing (essential for fitting larger models on a T4)

No HuggingFace `transformers` model classes, no pretrained weights. Everything
here is plain torch.nn so the training loop has full control.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: Optional[int] = None
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    use_gradient_checkpointing: bool = True
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3

    def __post_init__(self):
        if self.head_dim is None:
            assert self.hidden_size % self.num_attention_heads == 0, \
                "hidden_size must be divisible by num_attention_heads"
            self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            "num_attention_heads must be divisible by num_key_value_heads"

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # keep only fields the dataclass knows about (config JSON has extra metadata keys)
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in raw.items() if k in fields}
        return cls(**clean)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute the norm in fp32 for stability, then cast back
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    """Precompute cos/sin tables for rotary embeddings: shape [seq_len, head_dim]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)               # [seq_len, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)        # [seq_len, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: [B, n_heads, T, head_dim]; cos/sin: [T, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, n_kv, T, hd] -> [B, n_kv*n_rep, T, hd] for grouped-query attention."""
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, t, hd)
        .reshape(b, n_kv * n_rep, t, hd)
    )


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.attention_dropout

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Flash / memory-efficient SDPA with a causal mask — no [T,T] mask materialized.
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(attn)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.post_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)
        self.res_dropout = nn.Dropout(cfg.residual_dropout)

    def forward(self, x, cos, sin):
        x = x + self.res_dropout(self.attn(self.input_norm(x), cos, sin))
        x = x + self.res_dropout(self.mlp(self.post_norm(x)))
        return x


class FinantalForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self._rope_cache = {}  # (seq_len, device, dtype) -> (cos, sin)
        self.apply(self._init_weights)

        # scaled init for residual projections (GPT-2 style) — stabilizes deep nets
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.initializer_range / math.sqrt(2 * cfg.num_hidden_layers))

    def _init_weights(self, module):
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def _get_rope(self, seq_len, device, dtype):
        key = (seq_len, device, dtype)
        if key not in self._rope_cache:
            self._rope_cache[key] = build_rope_cache(
                seq_len, self.cfg.head_dim, self.cfg.rope_theta, device, dtype
            )
        return self._rope_cache[key]

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        b, t = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = self._get_rope(t, x.device, x.dtype)

        use_ckpt = self.cfg.use_gradient_checkpointing and self.training
        for layer in self.layers:
            if use_ckpt:
                x = checkpoint(layer, x, cos, sin, use_reentrant=False)
            else:
                x = layer(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # shift so position i predicts token i+1
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, temperature=1.0, top_k=None, eos_token_id=None):
        """Lightweight sampler for sanity-checking a checkpoint (no KV cache — keep it short)."""
        self.eval()
        eos_token_id = self.cfg.eos_token_id if eos_token_id is None else eos_token_id
        max_ctx = self.cfg.max_position_embeddings
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -max_ctx:]
            logits, _ = self.forward(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, nxt], dim=1)
            if eos_token_id is not None and (nxt == eos_token_id).all():
                break
        return input_ids

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
