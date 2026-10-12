"""别名 / 同义词词典：缩写 ↔ 全称、产品代号 ↔ 正式名、口语 ↔ 书面语。

用户问"K8s 怎么扩容"，文档里只写"Kubernetes 集群扩容"——BM25 一个词都对不上，向量也未必够近。
词典在两端生效：

- 入库（文档扩展）：切片里出现组内任一词，就把同组其余词写进"别名"行，一起进向量与 BM25；
- 查询（查询扩展）：BM25 查询里出现组内任一词，补上同组其余词。

两端都做是因为各自兜不住另一端：只扩展查询，向量路仍然看不到别名；只扩展文档，改词典后必须重建索引才生效。
（不把词典注册进 jieba：jieba 是进程全局的，会让其他知识库的查询与其已建索引的分词不一致。）
"""

from __future__ import annotations

import re
from collections import Counter

_ASCII = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _\-\.+/]*$")


def _norm(term: str) -> str:
    return re.sub(r"\s+", " ", term).strip()


class AliasDict:
    def __init__(self, groups: list[list[str]] | None = None) -> None:
        self.groups: list[list[str]] = []
        for g in groups or []:
            terms = list(dict.fromkeys(t for t in (_norm(x) for x in g) if t))
            if len(terms) >= 2:
                self.groups.append(terms)

    def __len__(self) -> int:
        return len(self.groups)

    def __bool__(self) -> bool:
        return bool(self.groups)

    def words(self) -> list[str]:
        return [t for g in self.groups for t in g]

    @staticmethod
    def _contains(text_low: str, term: str) -> bool:
        low = term.lower()
        if _ASCII.match(term):
            # 英文缩写按词边界匹配，避免 "SLA" 命中 "translate"
            return re.search(rf"(?<![A-Za-z0-9]){re.escape(low)}(?![A-Za-z0-9])", text_low) is not None
        return low in text_low

    def expansions(self, text: str) -> list[str]:
        """文本里出现了组内某个词、但没出现的同组词。"""
        low = text.lower()
        out: list[str] = []
        for g in self.groups:
            present = [t for t in g if self._contains(low, t)]
            if present:
                out.extend(t for t in g if t not in present and t not in out)
        return out

    def expand_query(self, query: str) -> str:
        extra = self.expansions(query)
        return f"{query} {' '.join(extra)}" if extra else query


# ----------------------------------------------------------------------------------------
# 从语料里发现别名候选：文档自己写了"全称（简称）"的地方
_CJK_TERM = r"[一-鿿][一-鿿A-Za-z0-9·\-]{1,19}"
_ABBR = r"[A-Za-z][A-Za-z0-9\-\.&]{1,11}"
_QUOTE = r"[\"“”‘’'「」『』]?"
_PATTERNS = [
    # 服务等级协议（SLA）/ Kubernetes（K8s）/ 服务等级协议（Service Level Agreement，SLA）
    re.compile(rf"({_CJK_TERM}|[A-Z][A-Za-z0-9\- ]{{3,40}}?)\s*[（(](?:[A-Za-z ,，\-]+[,，]\s*)?({_ABBR})[）)]"),
    # SLA（服务等级协议）
    re.compile(rf"(?<![A-Za-z0-9])({_ABBR})\s*[（(]({_CJK_TERM})[）)]"),
    # 员工带薪年休假（以下简称"年假"）/ 全称……简称……
    re.compile(rf"({_CJK_TERM})\s*[（(，,]\s*(?:以下)?简称\s*{_QUOTE}({_CJK_TERM}|{_ABBR}){_QUOTE}\s*[）)]?"),
    # 年休假，又称/也称/俗称/亦称"带薪假"
    re.compile(rf"({_CJK_TERM})\s*[，,]?\s*(?:又称|也称|也叫|俗称|亦称|又名|别名)\s*{_QUOTE}({_CJK_TERM}|{_ABBR}){_QUOTE}"),
]
_BAD_TAIL = re.compile(r"(的|和|与|及|或|等|在|是|为|对|将|把|被|由|从|向|以|于)$")
_STOP_ABBR = {"PDF", "Word", "Excel", "PPT", "HTTP", "HTTPS", "URL", "ID", "API", "OK", "No", "NO", "the", "and", "or"}


def discover_aliases(texts: list[str], existing: AliasDict | None = None, limit: int = 100) -> list[dict]:
    """返回 [{terms: [全称, 简称], count, evidence}]，按出现次数降序；已在词典里的组合不再推荐。"""
    known = {frozenset(t.lower() for t in g) for g in (existing.groups if existing else [])}
    known_pairs = {frozenset((a, b)) for g in known for a in g for b in g if a != b}
    counts: Counter = Counter()
    evidence: dict[frozenset, tuple[list[str], str]] = {}
    for text in texts:
        for pat in _PATTERNS:
            for m in pat.finditer(text):
                a, b = _norm(m.group(1)), _norm(m.group(2))
                a = _trim_prefix(a)
                if not a or not b or a.lower() == b.lower() or a in _STOP_ABBR or b in _STOP_ABBR or len(a) < 2 or len(b) < 2:
                    continue
                if _BAD_TAIL.search(a) or a.isdigit() or b.isdigit():
                    continue
                key = frozenset((a.lower(), b.lower()))
                if key in known_pairs:
                    continue
                counts[key] += 1
                if key not in evidence:
                    s, e = max(0, m.start() - 20), min(len(text), m.end() + 20)
                    evidence[key] = ([a, b], text[s:e].replace("\n", " "))
    out = []
    for key, n in counts.most_common(limit):
        terms, ev = evidence[key]
        out.append({"terms": terms, "count": n, "evidence": ev})
    return out


_LEAD = re.compile(r"^(?:.*?[，。；：:,;、]|根据|按照|依据|关于|对于|其中|本|该|此|所有|各)")
_STOP_POS = ("u", "c", "d", "p", "r", "x", "y", "e", "o", "m", "q", "f")   # 助词 / 连词 / 副词 / 介词 / 代词 / 标点 / 数量词 / 方位词


def _trim_prefix(term: str) -> str:
    """"要求先产出清晰的规格说明书"→"规格说明书"：从右往左取最长的实词串，遇到助词 / 连词 / 副词 / 介词就停。"""
    t = _LEAD.sub("", term).strip()
    if not t or not ("一" <= t[0] <= "鿿"):
        return t if len(t) >= 2 else term
    try:
        import jieba.posseg as pseg

        toks = list(pseg.cut(t))
    except Exception:
        return t if len(t) >= 2 else term
    keep: list[str] = []
    for w in reversed(toks):
        if w.flag.startswith(_STOP_POS) or w.word in ("要求", "进行", "使用", "采用", "通过", "包括", "称为", "简称"):
            break
        keep.insert(0, w.word)
    out = "".join(keep).strip()
    return out if len(out) >= 2 else (t if len(t) >= 2 else term)
