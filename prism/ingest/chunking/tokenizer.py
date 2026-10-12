"""Token 计数与分句。

- 有 tiktoken 时用 cl100k_base（与主流 LLM 计数接近），否则用启发式：
  中文一个字 ≈ 1 token，英文一个词 ≈ 1.3 token。切片只需要"量级正确"。
- 分句同时处理中英文标点，超长句再按逗号 / 分号二次切分，避免单句就超过 child 长度。
"""

from __future__ import annotations

import re
from functools import lru_cache

_CJK = re.compile(r"[一-鿿㐀-䶿぀-ヿ가-힯]")
_WORD = re.compile(r"[A-Za-z0-9_]+(?:[-'’.][A-Za-z0-9_]+)*|[^\sA-Za-z0-9_一-鿿㐀-䶿぀-ヿ가-힯]")
_SENT_END = re.compile(r"(?<=[。！？!?；;])\s*|(?<=[\.!?])\s+(?=[A-Z一-鿿])|\n+")
_SOFT_SPLIT = re.compile(r"(?<=[，,、：:])")


@lru_cache(maxsize=1)
def _tiktoken_encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _tiktoken_encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    cjk = len(_CJK.findall(text))
    rest = _CJK.sub(" ", text)
    words = _WORD.findall(rest)
    return cjk + int(sum(max(1.0, len(w) / 4) for w in words) * 1.0)


def split_sentences(text: str, max_tokens: int | None = None) -> list[str]:
    parts = [p.strip() for p in _SENT_END.split(text) if p and p.strip()]
    if max_tokens is None:
        return parts
    out: list[str] = []
    for p in parts:
        if count_tokens(p) <= max_tokens:
            out.append(p)
            continue
        buf = ""
        for piece in _SOFT_SPLIT.split(p):
            if buf and count_tokens(buf + piece) > max_tokens:
                out.append(buf.strip())
                buf = piece
            else:
                buf += piece
        if buf.strip():
            out.append(buf.strip())
    return out
