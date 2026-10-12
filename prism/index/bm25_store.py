"""BM25 关键词索引（bm25s + jieba）。

BM25 补的是向量检索的三个盲区：专有名词 / 错误码、新词与产品名、长尾精确匹配。
中文必须先分词，且索引与查询必须用同一套分词器。

额外的"精确 token 保护"：`SO-8817`、`CVE-2024-3094`、`MySQL 1062` 这类编号，jieba 会切碎；
这里用正则把它们整体再补进 token 列表，保证 query 与文档能整词命中。
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path

import numpy as np

from ..config import SparseConfig

PRECISE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"[A-Za-z]{1,10}[-_]?\d[A-Za-z0-9_\-\.]*"   # SO-8817, CVE-2024-3094, A2024-001, K8s, GPT-4o（只吃 ASCII：\w 会把"Q3各区域…"整句粘成一个 token）
    r"|\d+\.\d+(?:\.\d+)+"                  # 版本号 3.2.1
    r"|[A-Z]{2,}[A-Z0-9\-]*"                # ORD、HTTP、OSPF
    r"|\d{3,}"                              # 502、1062、20260831001
    r")(?![A-Za-z0-9])"
)
_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-\.]*")
_STOP = {"的", "了", "是", "在", "和", "与", "或", "及", "等", "对", "把", "被", "着", "过", "吗", "呢", "吧", "啊", "呀", "怎么", "什么", "如何", "哪些", "请问", "一下", "这个", "那个", "我", "你", "我们", "咱们", "公司", " ", ""}


_BUILD_LOCK = threading.RLock()


class BM25Store:
    def __init__(self, cfg: SparseConfig) -> None:
        self.cfg = cfg
        self.ids: list[str] = []
        self.corpus_tokens: list[list[str]] = []
        self._index = None
        self._dirty = False
        self._jieba = None
        if cfg.use_jieba:
            import jieba

            jieba.setLogLevel(60)
            if cfg.user_dict and Path(cfg.user_dict).exists():
                jieba.load_userdict(str(cfg.user_dict))
            self._jieba = jieba

    # ----------------------------------------------------------------------------------
    def tokenize(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        precise = [m.group(0).lower() for m in PRECISE_TOKEN.finditer(text)]
        if self._jieba is not None:
            base = [t.strip().lower() for t in self._jieba.lcut(text, HMM=True)]
        else:
            base = [t.lower() for t in _WORD.findall(text)] + [ch for ch in text if "一" <= ch <= "鿿"]
        toks = [t for t in base if t and t not in _STOP and not _is_punct(t)]
        return toks + precise

    def add(self, ids: list[str], texts: list[str]) -> None:
        for cid, text in zip(ids, texts):
            self.ids.append(cid)
            self.corpus_tokens.append(self.tokenize(text))
        self._dirty = True

    def _ensure_index(self) -> None:
        if self._index is not None and not self._dirty:
            return
        import bm25s

        # 多路召回并发执行：先在局部变量里建好再一次性替换，并加锁防止重复构建。
        # 否则其他线程会拿到"已创建、还没 index() 完"的对象（实测报 'BM25' object has no attribute 'vocab_dict'）。
        with _BUILD_LOCK:
            if self._index is not None and not self._dirty:
                return
            logging.getLogger("bm25s").setLevel(logging.WARNING)  # bm25s 自带 DEBUG 输出，静音
            index = bm25s.BM25(k1=self.cfg.k1, b=self.cfg.b)
            # 空文档需占位 token，否则 bm25s 无法建索引
            corpus = [toks if toks else ["<empty>"] for toks in self.corpus_tokens]
            index.index(corpus, show_progress=False)
            self._index = index
            self._dirty = False

    def search(self, query: str, top_n: int, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        self._ensure_index()
        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []
        scores = self._scores(q_tokens)
        if self.cfg.exact_token_boost > 0:
            precise = {m.group(0).lower() for m in PRECISE_TOKEN.finditer(query)}
            if precise:
                for i, toks in enumerate(self.corpus_tokens):
                    if precise & set(toks):
                        scores[i] += self.cfg.exact_token_boost
        if allowed_ids is not None:
            mask = np.fromiter((cid in allowed_ids for cid in self.ids), dtype=bool, count=len(self.ids))
            scores = np.where(mask, scores, -np.inf)
        k = min(top_n, len(self.ids))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.ids[i], float(scores[i])) for i in idx if np.isfinite(scores[i]) and scores[i] > 0]

    def _scores(self, q_tokens: list[str]) -> np.ndarray:
        # 查询里不在词表中的 token 会被 bm25s 忽略；全部未知时返回全零
        vocab = self._index.vocab_dict if hasattr(self._index, "vocab_dict") else None
        if vocab is not None:
            q_tokens = [t for t in q_tokens if t in vocab]
            if not q_tokens:
                return np.zeros(len(self.ids), dtype=np.float32)
        return np.asarray(self._index.get_scores(q_tokens), dtype=np.float32)

    # ----------------------------------------------------------------------------------
    def remove(self, ids: set[str]) -> None:
        """删除若干文档后重建（bm25s 不支持原地删除）。"""
        keep = [(cid, toks) for cid, toks in zip(self.ids, self.corpus_tokens) if cid not in ids]
        if len(keep) != len(self.ids):
            self.ids = [cid for cid, _ in keep]
            self.corpus_tokens = [toks for _, toks in keep]
            self._dirty = True

    def save(self, directory: Path, prefix: str = "") -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{prefix}bm25_corpus.jsonl").open("w", encoding="utf-8") as f:
            for cid, toks in zip(self.ids, self.corpus_tokens):
                f.write(json.dumps({"id": cid, "tokens": toks}, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: Path, cfg: SparseConfig, prefix: str = "") -> "BM25Store":
        store = cls(cfg)
        p = directory / f"{prefix}bm25_corpus.jsonl"
        if p.exists():
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        store.ids.append(rec["id"])
                        store.corpus_tokens.append(rec["tokens"])
            store._dirty = True
        return store


def _is_punct(t: str) -> bool:
    return all(not ch.isalnum() and not ("一" <= ch <= "鿿") for ch in t)
