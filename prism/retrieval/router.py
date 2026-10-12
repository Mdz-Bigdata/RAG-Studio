"""查询路由：按类别 / 语言把 query 分流到最合适的知识库（多重索引 + 路由）。

产品手册、售后政策、法律合同混在一个索引里会互相干扰；按业务域拆库后，需要一个路由器决定查哪个库。
- embedding 模式：把知识库描述（名称 + 描述 + 标签 + 代表性标题）和 query 一起 embed，余弦相似度选 top-N；
- llm 模式：让 LLM 直接选；
- 语言检测：中 / 英 / 混合，可用于选择 embedding instruction 或分词器。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

_CJK = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")


@dataclass
class KBDescriptor:
    kb_id: str
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    sample_titles: list[str] = field(default_factory=list)

    def text(self) -> str:
        parts = [self.name, self.description, " ".join(self.tags), " / ".join(self.sample_titles[:12])]
        return "\n".join(p for p in parts if p)


def detect_language(text: str) -> str:
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN.findall(text))
    if cjk == 0 and latin == 0:
        return "unknown"
    if cjk >= latin * 2:
        return "zh"
    if latin >= cjk * 4:
        return "en"
    return "mixed"


class KBRouter:
    def __init__(self, embedder=None, llm=None) -> None:
        self.embedder = embedder
        self.llm = llm

    def route(self, query: str, kbs: list[KBDescriptor], *, top_n: int = 1, mode: str = "embedding") -> list[dict]:
        if not kbs:
            return []
        if len(kbs) == 1:
            return [{"kb_id": kbs[0].kb_id, "name": kbs[0].name, "score": 1.0, "reason": "唯一知识库"}]
        if mode == "llm" and self.llm is not None:
            picked = self._route_llm(query, kbs, top_n)
            if picked:
                return picked
        return self._route_embedding(query, kbs, top_n)

    def _route_embedding(self, query: str, kbs: list[KBDescriptor], top_n: int) -> list[dict]:
        if self.embedder is None:
            return [{"kb_id": kb.kb_id, "name": kb.name, "score": 0.0, "reason": "无 embedder，按顺序"} for kb in kbs[:top_n]]
        q = self.embedder.embed_queries([query])[0]
        m = self.embedder.embed_documents([kb.text() for kb in kbs])
        q = q / (np.linalg.norm(q) + 1e-9)
        m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
        sims = m @ q
        order = np.argsort(-sims)[:top_n]
        return [{"kb_id": kbs[i].kb_id, "name": kbs[i].name, "score": float(sims[i]), "reason": "描述向量相似度"} for i in order]

    def _route_llm(self, query: str, kbs: list[KBDescriptor], top_n: int) -> list[dict]:
        listing = "\n".join(f"{i + 1}. {kb.name}：{kb.description or '（无描述）'}；标签：{', '.join(kb.tags) or '-'}" for i, kb in enumerate(kbs))
        prompt = f"""下面是可用的知识库列表：
{listing}

用户问题：{query}

请选出最可能包含答案的 {top_n} 个知识库，只输出 JSON 数组，如 [{{"id": 1, "reason": "..."}}]。"""
        try:
            raw = self.llm.complete(prompt, max_tokens=200)
            m = re.search(r"\[.*\]", raw, re.S)
            data = json.loads(m.group(0)) if m else []
        except Exception:
            return []
        out = []
        for item in data:
            try:
                i = int(item.get("id", 0)) - 1
            except (TypeError, ValueError, AttributeError):
                continue
            if 0 <= i < len(kbs):
                out.append({"kb_id": kbs[i].kb_id, "name": kbs[i].name, "score": 1.0 - len(out) * 0.1, "reason": str(item.get("reason", "LLM 选择"))})
        return out[:top_n]
