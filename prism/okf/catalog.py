"""OKF 目录：知识库里哪些文档是权威条目、各叫什么名字；问题点名了哪个条目。

匹配是确定性的，不走向量：问题里出现了条目的标题或别名，就是问它——"月流失率怎么算"命中《Monthly Churn Rate》（别名"月流失率"）。
中文名按子串匹配；拉丁字母的名字按整词匹配（"churn" 不能被 "churned" 里的碎片命中）。多个条目都命中时，名字长的优先
（"客户流失率公式"比"流失率"更具体），最多取 max_pinned 个。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..schemas import DocMetadata

_LATIN = re.compile(r"^[A-Za-z0-9 _.\-/]+$")
_SEP = re.compile(r"[\s_\-]+")


@dataclass
class OKFMatch:
    doc_id: str
    title: str
    matched: str                        # 命中的名字（标题或别名）
    type: str = ""
    owner: str = ""


class OKFCatalog:
    def __init__(self, docs: Mapping[str, DocMetadata]) -> None:
        self.entries: list[tuple[str, str, dict[str, Any], list[str]]] = []   # (doc_id, title, okf 元数据, 名字)
        for doc_id, md in docs.items():
            okf = md.extra.get("okf") if md is not None else None
            if not isinstance(okf, dict):
                continue
            names = [md.title or ""] + [str(a) for a in (okf.get("aliases") or [])]
            names = [" ".join(n.split()) for n in names if len(" ".join(str(n).split())) >= 2]
            if names:
                self.entries.append((doc_id, md.title or doc_id, okf, names))

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _hit(name: str, query: str) -> bool:
        n = name.casefold()
        if _LATIN.match(name):
            # 整词匹配，空格 / 下划线 / 连字符互通："monthly churn rate" 也能命中 "Monthly_Churn_Rate"
            pat = r"(?<![a-z0-9])" + r"[\s_\-]*".join(re.escape(t) for t in _SEP.split(n) if t) + r"(?![a-z0-9])"
            return re.search(pat, query) is not None
        return n in query

    def match(self, query: str, *, limit: int = 2, allowed_doc: Callable[[str], bool] | None = None) -> list[OKFMatch]:
        q = " ".join((query or "").casefold().split())
        if not q or not self.entries:
            return []
        hits: list[tuple[int, int, OKFMatch]] = []
        for order, (doc_id, title, okf, names) in enumerate(self.entries):
            if allowed_doc is not None and not allowed_doc(doc_id):
                continue
            best = max((n for n in names if self._hit(n, q)), key=len, default=None)
            if best is not None:
                hits.append((-len(best), order, OKFMatch(doc_id=doc_id, title=title, matched=best, type=str(okf.get("type") or ""), owner=str(okf.get("owner") or ""))))
        hits.sort(key=lambda x: (x[0], x[1]))
        return [h[2] for h in hits[:limit]]
