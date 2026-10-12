"""切片关键词（文档扩展，doc2query 的零成本版本）。

每个 parent 抽出 N 个关键词：TF-IDF（jieba 内置 IDF）挑主题词 + 正则保住编号 / 错误码 / 型号 / 版本号。
关键词写进 Chunk.keywords，经 embed_text() 同时进入向量与 BM25：
- BM25：主题词多出现一次，等于给这段的"核心概念"加权；编号整词进入词表，查询能整体命中；
- 向量：短块（一两句话）语义稀薄，补上主题词能把它拉回正确的主题簇。
不调用 LLM，入库成本可以忽略。
"""

from __future__ import annotations

import re

from ..index.bm25_store import PRECISE_TOKEN

_STOP = {
    "可以", "进行", "需要", "应当", "以及", "或者", "如果", "其中", "相关", "有关", "具体", "情况", "规定", "要求", "按照", "根据",
    "包括", "通过", "对于", "以上", "以下", "本制度", "本办法", "公司", "员工", "我们", "他们", "这些", "那些", "一个", "没有",
    "密级", "版本", "附件", "说明", "内容", "方式", "时间", "部门", "人员", "工作", "管理", "问题",
}
_POS = ("n", "nr", "ns", "nt", "nz", "nw", "vn", "an", "eng", "l", "j", "i")  # 名词性 + 英文 + 习用语 / 简称
_STRUCT = re.compile(r"^第[一二三四五六七八九十百零\d]+[条章节款项页]$")


def extract_keywords(text: str, n: int = 8, title: str = "", precise: bool = True) -> list[str]:
    """precise token 优先（最容易被向量稀释），其余按 TF-IDF。标题里已有的词不重复加。
    precise=False：只要主题词（用作文档标签时，编号 / 版本号不是"主题"）。"""
    if not text.strip() or n <= 0:
        return []
    seen_title = title.lower()
    out: list[str] = []
    for m in (PRECISE_TOKEN.finditer(text) if precise else ()):
        tok = m.group(0)
        if len(tok) >= 3 and not tok.isdigit() and tok not in out and tok.lower() not in seen_title:
            out.append(tok)
        if len(out) >= max(2, n // 2):
            break
    try:
        import jieba.analyse

        cands = jieba.analyse.extract_tags(text, topK=n * 3, allowPOS=_POS)
    except Exception:
        cands = []
    for w in cands:
        w = w.strip()
        if len(out) >= n:
            break
        if len(w) < 2 or w in _STOP or not re.search(r"[A-Za-z一-鿿]", w) or re.fullmatch(r"[\d.%\-:/]+", w) or _STRUCT.match(w) or w in out or w.lower() in seen_title:
            continue
        if any(w.lower() in o.lower() for o in out) or (len(w) <= 2 and w.isascii()):
            continue
        out.append(w)
    return out[:n]
