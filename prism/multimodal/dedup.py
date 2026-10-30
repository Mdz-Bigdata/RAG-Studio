"""内容查重与相似推荐：图片之间、视频之间（以及图片与视频片段之间）的向量相似度。

    图片      图片向量（图文空间）两两余弦 ≥ 阈值 → 重复（翻拍 / 转码 / 裁边 / 加水印）
    视频      片段级：A 的每个片段在 B 里找最像的片段，余弦 ≥ 阈值算"这一段在 B 里出现过"，
              重合度 = 出现过的片段占比（取 A→B、B→A 的较大者：剪辑版 / 片段截取也能认出来）；
              整体：各片段向量平均后的余弦，看"讲的是不是同一类内容"，用于相似推荐
    片段向量  能直接编码视频的模型（WeMM）用整段的"纯视频"向量，否则用关键帧的图片向量

阈值不给时按模型自动（BaseImageEmbedder.dedup_threshold）。
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from .image_index import ImageIndex


class SimilarVideo(BaseModel):
    doc_id: str
    score: float                          # 整体相似度：两个视频片段向量均值的余弦
    overlap: float                        # 片段重合度（0~1）
    duplicate: bool = False
    pairs: list[tuple[str, str, float]] = Field(default_factory=list)   # (本视频片段, 对方片段, 余弦)，按对方片段时间排序


class DuplicateGroup(BaseModel):
    kind: str                             # image | video
    ids: list[str]                        # image_id（图片）或 doc_id（视频）
    score: float                          # 组内最高的相似度（视频是重合度）


def _segment_vectors(index: ImageIndex, doc_id: str) -> tuple[list[str], np.ndarray]:
    """某个视频的片段 id 与片段向量（按时间顺序）。优先整段的视频向量，没有就用关键帧的图片向量。"""
    recs = sorted((r for r in index.records.values() if r.doc_id == doc_id and r.is_video), key=lambda r: r.time_start or r.time or 0.0)
    ids = [r.image_id for r in recs]
    vecs = index.video.get_vectors(ids) if len(index.video) else {}
    if len(vecs) < len(ids):
        vecs = {**index.visual.get_vectors([i for i in ids if i not in vecs]), **vecs}
    ids = [i for i in ids if i in vecs]
    return ids, (np.vstack([vecs[i] for i in ids]).astype(np.float32) if ids else np.zeros((0, 0), dtype=np.float32))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class ContentDeduper:
    def __init__(self, index: ImageIndex, *, threshold: float | None = None, overlap: float = 0.6) -> None:
        self.index = index
        self.threshold = float(threshold if threshold is not None else index.embedder.dedup_threshold)
        self.overlap = overlap

    def videos(self) -> list[str]:
        return sorted({r.doc_id for r in self.index.records.values() if r.is_video})

    # --- 相似 / 重复：单个条目 ----------------------------------------------------------------
    def similar_images(self, image_id: str, top_k: int = 10) -> list[tuple[str, float]]:
        """与这张图（或这个视频片段的关键帧）最像的图片 / 片段。同一个视频的其他片段不算。"""
        rec = self.index.get(image_id)
        vec = self.index.visual.get_vectors([image_id]).get(image_id)
        if rec is None or vec is None:
            return []
        skip = {r.image_id for r in self.index.records.values() if r.doc_id == rec.doc_id} if rec.is_video else {image_id}
        allowed = {i for i in self.index.records if i not in skip}
        return self.index.visual.search(vec, top_k, allowed) if allowed else []

    def compare_videos(self, a: str, b: str, ids_a: list[str] | None = None, va: np.ndarray | None = None) -> SimilarVideo | None:
        if ids_a is None or va is None:
            ids_a, va = _segment_vectors(self.index, a)
        ids_b, vb = _segment_vectors(self.index, b)
        if not ids_a or not ids_b or va.shape[1] != vb.shape[1]:
            return None
        sim = va @ vb.T
        cover_a = float((sim.max(axis=1) >= self.threshold).mean())
        cover_b = float((sim.max(axis=0) >= self.threshold).mean())
        best = sim.argmax(axis=1)
        pairs = sorted(((ids_a[i], ids_b[int(best[i])], float(sim[i, best[i]])) for i in range(len(ids_a)) if sim[i, best[i]] >= self.threshold),
                       key=lambda p: ids_b.index(p[1]))
        overlap = max(cover_a, cover_b)
        score = float(_unit(va.mean(axis=0)) @ _unit(vb.mean(axis=0)))
        return SimilarVideo(doc_id=b, score=score, overlap=overlap, duplicate=overlap >= self.overlap, pairs=pairs)

    def similar_videos(self, doc_id: str, top_k: int = 10) -> list[SimilarVideo]:
        """与这个视频相似的视频：重复（片段重合）的排前面，其余按整体相似度。"""
        ids, va = _segment_vectors(self.index, doc_id)
        if not ids:
            return []
        out = [s for other in self.videos() if other != doc_id and (s := self.compare_videos(doc_id, other, ids, va)) is not None]
        out.sort(key=lambda s: (s.duplicate, s.overlap if s.duplicate else s.score), reverse=True)
        return out[:top_k]

    # --- 全库查重 ---------------------------------------------------------------------------
    def duplicate_groups(self, kinds: tuple[str, ...] = ("image", "video")) -> list[DuplicateGroup]:
        groups: list[DuplicateGroup] = []
        if "image" in kinds:
            groups += self._image_groups()
        if "video" in kinds:
            groups += self._video_groups()
        groups.sort(key=lambda g: (len(g.ids), g.score), reverse=True)
        return groups

    def _image_groups(self) -> list[DuplicateGroup]:
        ids = [i for i, r in self.index.records.items() if not r.is_video]
        vecs = self.index.visual.get_vectors(ids)
        ids = [i for i in ids if i in vecs]
        if len(ids) < 2:
            return []
        m = np.vstack([vecs[i] for i in ids]).astype(np.float32)
        pairs: list[tuple[int, int, float]] = []
        for start in range(0, len(ids), 1024):           # 分块算，几千张图也不会一次占满内存
            block = m[start:start + 1024] @ m.T
            for r, c in zip(*np.nonzero(block >= self.threshold)):
                i = start + int(r)
                if int(c) > i:
                    pairs.append((i, int(c), float(block[r, c])))
        return [DuplicateGroup(kind="image", ids=[ids[k] for k in g], score=s) for g, s in _union(len(ids), pairs)]

    def _video_groups(self) -> list[DuplicateGroup]:
        docs = self.videos()
        segs = {d: _segment_vectors(self.index, d) for d in docs}
        pairs: list[tuple[int, int, float]] = []
        for i, a in enumerate(docs):
            for j in range(i + 1, len(docs)):
                s = self.compare_videos(a, docs[j], *segs[a])
                if s is not None and s.duplicate:
                    pairs.append((i, j, s.overlap))
        return [DuplicateGroup(kind="video", ids=[docs[k] for k in g], score=s) for g, s in _union(len(docs), pairs)]


def _union(n: int, pairs: list[tuple[int, int, float]]) -> list[tuple[list[int], float]]:
    """并查集：两两相似的边 → 连通分量（只返回 ≥2 个成员的组）与组内最高分。"""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    best: dict[int, float] = {}
    for a, b, s in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            best[ra] = max(best.get(ra, 0.0), best.pop(rb, 0.0), s)
        else:
            best[ra] = max(best.get(ra, 0.0), s)
    comps: dict[int, list[int]] = {}
    for k in range(n):
        comps.setdefault(find(k), []).append(k)
    return [(members, best.get(root, 0.0)) for root, members in comps.items() if len(members) > 1]

