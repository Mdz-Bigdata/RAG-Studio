"""图片索引：一张图片三份索引 + 原图文件，和文本索引放在同一个知识库目录下。

    <index_dir>/images/<doc_id>/<NNN>.png|jpg      原图（已缩放到 max_side）
    <index_dir>/image_index/images.jsonl           ImageRecord
    <index_dir>/image_index/visual/                图片向量（图文统一空间：CLIP 图像塔 / 离线感知特征）
    <index_dir>/image_index/desc/                  "图片文本摘要"的文本向量（与正文同一个文本向量模型）
    <index_dir>/image_index/bm25_corpus.jsonl      "图片文本摘要"的 BM25
    <index_dir>/image_index/chunk_clip/            正文 child 块在图文空间里的文本向量（图搜文直达路，仅跨模态模型）

向量模型换了怎么办：原图就在磁盘上，加载时发现签名不一致会自动重算向量，不需要重新解析文档。

向量后端是 LanceDB（vector_store.backend: lancedb）时，visual / desc / chunk_clip 三份向量改存在知识库的 LanceDB 库里
（表 img_visual / img_desc / img_chunk_clip），记录（images.jsonl）、原图文件与图片文本 BM25 不变；
切换后、知识库重建之前，已有的 numpy 向量文件照常加载使用。
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..config import LanceDBConfig, SparseConfig
from ..index.bm25_store import BM25Store
from ..index.embeddings import BaseEmbedder
from ..index.vector_store import NumpyVectorStore
from ..schemas import Chunk
from .embedders import BaseImageEmbedder
from .extract import ExtractedImage
from .schemas import ImageRecord

logger = logging.getLogger(__name__)

_BIGRAM_SKIP = set(" \n\t，。、；：,.;:()（）[]【】")


class ImageIndex:
    SUBDIR = "image_index"
    FILES = "images"

    def __init__(self, directory: Path, *, embedder_factory: Callable[[], BaseImageEmbedder], text_embedder: BaseEmbedder, sparse_cfg: SparseConfig, lance_cfg: LanceDBConfig | None = None) -> None:
        self.directory = Path(directory)
        self._embedder_factory = embedder_factory
        self._embedder: BaseImageEmbedder | None = None
        self.text_embedder = text_embedder
        self.sparse_cfg = sparse_cfg
        self.lance_cfg = lance_cfg
        self.records: dict[str, ImageRecord] = {}
        self.visual = self._new_store("visual")
        self.desc = self._new_store("desc")
        self.chunk_clip = self._new_store("chunk_clip")
        self.bm25 = BM25Store(sparse_cfg)
        self._meta: dict[str, Any] = {}
        self._checked = False

    def __len__(self) -> int:
        return len(self.records)

    def _new_store(self, name: str, *, fresh: bool = False):
        """numpy 或 LanceDB（表 img_<name>）向量存储。fresh：换了模型要整份重算时清空旧表（维度可能变了）。"""
        if self.lance_cfg is None:
            return NumpyVectorStore(dim=0)
        from ..lance import LanceDBVectorStore   # 延迟导入：没装 lancedb 时 numpy 版照常可用

        store = LanceDBVectorStore(self.directory / "lancedb", 0, table=f"img_{name}", cfg=self.lance_cfg)
        if fresh:
            store.drop()
        return store

    @property
    def store_backend(self) -> str:
        return getattr(self.visual, "name", "numpy")

    # --- 向量模型（懒加载：纯文本知识库不会为此加载 CLIP）-------------------------------------
    @property
    def embedder(self) -> BaseImageEmbedder:
        if self._embedder is None:
            self._embedder = self._embedder_factory()
        return self._embedder

    @property
    def embedder_loaded(self) -> bool:
        return self._embedder is not None

    def _text_signature(self) -> dict[str, Any]:
        return {"embedder": self.text_embedder.name, "model": getattr(self.text_embedder, "model_name", None)}

    def ensure_fresh(self) -> None:
        """索引是用别的向量模型建的 → 用磁盘上的原图 / 摘要文本重算，保证查询向量与索引向量同空间。"""
        if self._checked:
            return
        self._checked = True
        if not self.records:
            return
        built_text = self._meta.get("text")
        if built_text is not None and built_text != self._text_signature():
            logger.info("文本向量模型已更换，重算 %d 张图片的描述向量", len(self.records))
            self._reembed_desc(list(self.records.values()))
        built = self._meta.get("visual") or {}
        sig = self.embedder.signature()
        if (built.get("embedder"), built.get("model")) != (sig["embedder"], sig["model"]) or len(self.visual) != len(self.records):
            logger.info("图文向量模型已更换（%s → %s），重算 %d 张图片的向量", built.get("model"), sig["model"], len(self.records))
            self._reembed_visual()
            self.chunk_clip = self._new_store("chunk_clip", fresh=True)

    def _reembed_visual(self) -> None:
        from PIL import Image

        self.visual = self._new_store("visual", fresh=True)
        ids, images = [], []
        for rec in self.records.values():
            p = self.path_of(rec)
            if p.is_file():
                ids.append(rec.image_id)
                images.append(Image.open(p).convert("RGB"))
        if ids:
            self.visual.add(ids, self.embedder.embed_images(images))

    def _reembed_desc(self, records: list[ImageRecord]) -> None:
        if not records:
            return
        self.desc.remove([r.image_id for r in records])
        self.desc.add([r.image_id for r in records], self.text_embedder.embed_documents([r.search_text() or r.doc_title or "图片" for r in records]))
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        self.bm25 = BM25Store(self.sparse_cfg)
        recs = list(self.records.values())
        self.bm25.add([r.image_id for r in recs], [r.search_text() for r in recs])

    # --- 写入 -----------------------------------------------------------------------------
    def replace_document(self, doc_id: str, doc_title: str, items: list[ExtractedImage], chunks: list[Chunk]) -> list[ImageRecord]:
        """用新抽取的图片替换该文档的全部图片。"""
        self.remove_document(doc_id)
        if not items:
            return []
        self.ensure_fresh()
        out_dir = self.directory / self.FILES / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        records: list[ImageRecord] = []
        for n, it in enumerate(items):
            name = _save_image(it.image, out_dir / f"{n:03d}")
            w, h = it.image.size
            records.append(ImageRecord(
                image_id=f"{doc_id}_img{n:03d}", doc_id=doc_id, file=f"{self.FILES}/{doc_id}/{name}", sha1=it.sha1, width=w, height=h, order=n,
                page=it.page, bbox=it.bbox, origin=it.origin, doc_title=doc_title, caption=it.caption, ocr_text=it.ocr_text, description=it.description, context_text=it.context,
            ))
        _link(records, items, chunks)
        for r in records:
            self.records[r.image_id] = r
        self.visual.add([r.image_id for r in records], self.embedder.embed_images([it.image for it in items]))
        self._reembed_desc(records)
        return records

    def relink(self, doc_id: str, chunks: list[Chunk]) -> None:
        """文档重新切片后（如 Markdown 校对回灌）：图片保留，只重建图文关联与描述索引。"""
        records = self.of_document(doc_id)
        if not records:
            return
        _link(records, [None] * len(records), chunks)
        self._reembed_desc(records)

    def remove_document(self, doc_id: str, *, delete_files: bool = True) -> list[str]:
        ids = [i for i, r in self.records.items() if r.doc_id == doc_id]
        if ids:
            for i in ids:
                del self.records[i]
            self.visual.remove(ids)
            self.desc.remove(ids)
            self._rebuild_bm25()
        if delete_files:
            shutil.rmtree(self.directory / self.FILES / doc_id, ignore_errors=True)
        return ids

    def sync_chunk_vectors(self, children: list[Chunk], *, limit: int | None = None) -> tuple[int, int]:
        """让 chunk_clip 与当前 child 集合一致（增量）。只有跨模态模型才有意义。
        limit：本次最多补建多少条——大知识库首次图搜文时一次补齐会在一个请求里卡上几分钟（CPU 上每条约 20ms），
        分摊到后续查询里逐步补齐。返回 (本次新增, 仍缺)。"""
        if not self.embedder.cross_modal:
            return 0, 0
        want = {c.chunk_id: c for c in children}
        self.chunk_clip.remove([i for i in self.chunk_clip.ids if i not in want])
        have = set(self.chunk_clip.ids)
        missing = [c for cid, c in want.items() if cid not in have]
        batch = missing if limit is None else missing[:limit]
        if batch:
            self.chunk_clip.add([c.chunk_id for c in batch], self.embedder.embed_texts([c.text for c in batch]))
        return len(batch), len(missing) - len(batch)

    def add_chunk_vectors(self, chunks: list[Chunk]) -> int:
        """入库时只为本文档的 child 增量建图文空间向量（不碰其他文档）。"""
        if not self.embedder.cross_modal:
            return 0
        have = set(self.chunk_clip.ids)
        new = [c for c in chunks if c.kind == "child" and c.chunk_id not in have]
        if new:
            self.chunk_clip.add([c.chunk_id for c in new], self.embedder.embed_texts([c.text for c in new]))
        return len(new)

    # --- 读取 -----------------------------------------------------------------------------
    def get(self, image_id: str) -> ImageRecord | None:
        return self.records.get(image_id)

    def of_document(self, doc_id: str) -> list[ImageRecord]:
        return sorted((r for r in self.records.values() if r.doc_id == doc_id), key=lambda r: r.order)

    def by_anchor(self, chunk_ids: list[str]) -> list[ImageRecord]:
        """按给定的 parent 块顺序返回挂在这些块上的图片。"""
        pos = {cid: i for i, cid in enumerate(chunk_ids)}
        hits = [r for r in self.records.values() if r.anchor_chunk_id in pos]
        return sorted(hits, key=lambda r: (pos[r.anchor_chunk_id], r.order))

    def path_of(self, record: ImageRecord) -> Path:
        return self.directory / record.file

    def load_image(self, record: ImageRecord) -> Any:
        from PIL import Image

        return Image.open(self.path_of(record)).convert("RGB")

    # --- 持久化 ---------------------------------------------------------------------------
    def save(self, directory: Path | None = None) -> None:
        d = Path(directory or self.directory) / self.SUBDIR
        if not self.records and not d.exists():
            return
        d.mkdir(parents=True, exist_ok=True)
        with (d / "images.jsonl").open("w", encoding="utf-8") as f:
            for r in self.records.values():
                f.write(r.model_dump_json() + "\n")
        self.visual.save(d / "visual")
        self.desc.save(d / "desc")
        self.chunk_clip.save(d / "chunk_clip")
        self.bm25.save(d)
        meta = {"images": len(self.records), "text": self._text_signature(), "visual": self._embedder.signature() if self._embedder is not None else self._meta.get("visual")}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self._meta = meta

    @classmethod
    def load(cls, directory: Path, *, embedder_factory: Callable[[], BaseImageEmbedder], text_embedder: BaseEmbedder, sparse_cfg: SparseConfig, lance_cfg: LanceDBConfig | None = None) -> "ImageIndex":
        idx = cls(directory, embedder_factory=embedder_factory, text_embedder=text_embedder, sparse_cfg=sparse_cfg, lance_cfg=lance_cfg)
        d = Path(directory) / cls.SUBDIR
        if not (d / "images.jsonl").exists():
            return idx
        with (d / "images.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = ImageRecord.model_validate_json(line)
                    idx.records[r.image_id] = r
        for name in ("visual", "desc", "chunk_clip"):
            store = getattr(idx, name)
            # numpy 模式照旧读文件；LanceDB 模式下表还空着（切换后还没重建）就先沿用旧的 numpy 文件
            if (d / name / "vectors.npy").exists() and (isinstance(store, NumpyVectorStore) or not len(store)):
                setattr(idx, name, NumpyVectorStore.load(d / name))
        idx.bm25 = BM25Store.load(d, sparse_cfg)
        if (d / "meta.json").exists():
            idx._meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return idx

    def stats(self) -> dict[str, Any]:
        visual = self._embedder.signature() if self._embedder is not None else (self._meta.get("visual") or {})
        return {"images": len(self.records), "image_embedder": visual.get("embedder"), "image_model": visual.get("model"), "cross_modal": visual.get("cross_modal"), "chunk_clip": len(self.chunk_clip), "image_store": self.store_backend}


# --------------------------------------------------------------------------------------
def _save_image(image: Any, stem: Path) -> str:
    """照片类（颜色很多）存 JPEG，图表 / 截图存 PNG。返回文件名。"""
    photo = image.resize((64, 64)).getcolors(1024) is None
    if photo:
        path = stem.with_suffix(".jpg")
        image.save(path, format="JPEG", quality=88)
    else:
        path = stem.with_suffix(".png")
        image.save(path, format="PNG", optimize=True)
    return path.name


def _bigrams(text: str) -> set[str]:
    chars = [c for c in text.lower() if c not in _BIGRAM_SKIP]
    return {a + b for a, b in zip(chars, chars[1:])}


def _link(records: list[ImageRecord], items: list[ExtractedImage | None], chunks: list[Chunk]) -> None:
    """图文关联：把每张图挂到它所在的 parent 块上（同页优先，再按图注 / 上下文与块文本的重合度）。"""
    parents = [c for c in chunks if c.kind == "parent"]
    if not parents:
        return
    grams = {p.chunk_id: _bigrams(p.text) for p in parents}
    for rec, item in zip(records, items):
        markup = item.markup if item is not None else ""
        best = next((p for p in parents if markup and markup in p.text), None)
        if best is None:
            cands = parents
            if rec.page is not None:
                same = [p for p in parents if p.page_start is not None and p.page_start <= rec.page <= (p.page_end if p.page_end is not None else p.page_start)]
                cands = same or parents
            probe = _bigrams(f"{rec.caption} {rec.context_text}")
            scored = [(len(probe & grams[p.chunk_id]) / (len(probe) or 1), p) for p in cands]
            top = max(scored, key=lambda x: x[0], default=(0.0, None))
            # 没有任何文本线索时：同页有块就挂第一块；否则不硬挂，避免错误的图文关联
            best = top[1] if top[0] > 0 else (cands[0] if cands is not parents or len(parents) == 1 else None)
        rec.anchor_chunk_id = best.chunk_id if best is not None else None
        rec.breadcrumbs = list(best.breadcrumbs) if best is not None else ([rec.doc_title] if rec.doc_title else [])
