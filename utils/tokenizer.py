"""
Thin wrapper around the existing SentencePiece tokenizer. The tokenizer file lives
on Drive (config.paths.TOKENIZER_MODEL) — never in the Git repo. Tokenizer logic is
unchanged; this only wires in the central path.
"""

from __future__ import annotations

import sentencepiece as spm

try:
    from config.paths import TOKENIZER_MODEL as _DEFAULT_TOKENIZER
except Exception:  # pragma: no cover — allows import even if config path isn't set up
    _DEFAULT_TOKENIZER = "tokenizer/finantal_tokenizer.model"


class SPTokenizer:
    def __init__(self, model_path: str | None = None):
        model_path = model_path or _DEFAULT_TOKENIZER
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.pad_id = self.sp.piece_to_id("<pad>")
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos and self.bos_id >= 0:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id >= 0:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode([int(i) for i in ids if int(i) >= 0])
