"""索引门面：把向量索引、BM25、块存储、实体图组合成一个可持久化的 Index。

索引建在 child 上（检索粒度），parent 只进块存储（阅读粒度）；实体定义从 parent 抽取。
文档里的图片进 ImageIndex（图文向量 + 描述文本向量 + 描述 BM25），并挂到所在的 parent 块上。
一个 Index = 一个目录（Web 应用中每个知识库一个目录）。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import Settings
from ..graph.unweaver import EntityStore, LLMEntityExtractor, extract_definitions
from ..schemas import Chunk, DocMetadata
from .aux_index import AuxIndex
from .bm25_store import BM25Store
from .doc_store import ChunkStore
from .embeddings import BaseEmbedder, build_embedder
from .table_store import TableData, TableStore, link_chunks
from .vector_store import ChromaVectorStore, NumpyVectorStore, QdrantVectorStore

if TYPE_CHECKING:
    from ..multimodal.embedders import BaseImageEmbedder
    from ..multimodal.extract import ExtractedImage

logger = logging.getLogger(__name__)

__all__ = ["Index", "BaseEmbedder", "build_embedder", "BM25Store", "ChunkStore", "NumpyVectorStore", "QdrantVectorStore", "ChromaVectorStore"]


AUX_NAMES = ("summary", "section", "sentence")


class Index:
    def __init__(self, settings: Settings, embedder: BaseEmbedder | None = None, directory: Path | None = None, llm=None, image_embedder: "BaseImageEmbedder | None" = None, image_embedder_factory: "Callable[[], BaseImageEmbedder] | None" = None) -> None:
        from ..multimodal.image_index import ImageIndex

        self.settings = settings
        self.directory = Path(directory or settings.paths.index_dir)
        self.embedder = embedder or build_embedder(settings.embedding)
        self.image_embedder = image_embedder
        self._image_embedder_factory = image_embedder_factory
        self.images = ImageIndex(self.directory, embedder_factory=self._image_embedder, text_embedder=self.embedder, sparse_cfg=settings.sparse, lance_cfg=self._aux_lance().get("lance_cfg"))
        self.vectors = self._make_vector_store()
        self._tune_vectors()
        self.bm25 = BM25Store(settings.sparse)
        # 辅助索引：摘要 / 章节级 / 句子级（入库时开了对应开关才会有内容）
        dim = self.embedder.dim or settings.embedding.dim
        self.aux: dict[str, AuxIndex] = {name: AuxIndex(name, dim, settings.sparse, **self._aux_lance()) for name in AUX_NAMES}
        self.store = ChunkStore()
        self.entities = EntityStore()
        self.tables = TableStore()
        self.entity_extractor = LLMEntityExtractor(llm) if (llm is not None and settings.retrieval.graph.llm_extraction) else None
        self.aliases = None              # corpus.AliasDict：BM25 查询扩展（Web 应用按知识库注入）

    def _aux_lance(self) -> dict:
        """向量后端是 LanceDB 时，摘要 / 章节 / 句子辅助索引也各建一张 LanceDB 表（同一个库）。"""
        cfg = self.settings.vector_store
        return {"directory": self.directory, "lance_cfg": cfg.lancedb} if cfg.backend == "lancedb" else {}

    def _image_embedder(self) -> "BaseImageEmbedder":
        """图文向量模型懒加载：纯文本知识库不会为此加载 CLIP。"""
        if self.image_embedder is None:
            from ..multimodal.embedders import build_image_embedder

            self.image_embedder = self._image_embedder_factory() if self._image_embedder_factory else build_image_embedder(self.settings.multimodal.embedding)
        return self.image_embedder

    def _make_vector_store(self):
        cfg = self.settings.vector_store
        dim = self.embedder.dim or self.settings.embedding.dim
        if cfg.backend == "qdrant":
            return QdrantVectorStore(f"{cfg.collection_prefix}_{self.directory.name}", dim, url=cfg.qdrant_url, path=cfg.qdrant_path, hnsw_m=cfg.hnsw_m, hnsw_ef_construct=cfg.hnsw_ef_construct, quantization=cfg.quantization, ef_search=cfg.search.ef_search, oversample=cfg.search.oversample)
        if cfg.backend == "chroma":
            return ChromaVectorStore(self.directory / "chroma", collection=cfg.collection_prefix)
        if cfg.backend == "lancedb":
            from ..lance import LanceDBVectorStore   # 延迟导入：没装 lancedb 时其他后端照常可用

            return LanceDBVectorStore(self.directory / "lancedb", dim, cfg=cfg.lancedb)
        return NumpyVectorStore(dim=dim)

    def _tune_vectors(self) -> None:
        s = self.settings.vector_store.search
        configure = getattr(self.vectors, "configure", None)
        if callable(configure):
            configure(pca_dim=s.pca_dim, quantize=s.quantize, rescore=s.rescore, oversample=s.oversample, ef_search=s.ef_search, nprobes=s.nprobes, refine_factor=s.refine_factor)

    # ----------------------------------------------------------------------------------
    def add_document(self, doc_id: str, metadata: DocMetadata, chunks: list[Chunk], images: "list[ExtractedImage] | None" = None, tables: list[TableData] | None = None) -> None:
        """images=None：保留该文档已有的图片，只按新切片重建图文关联（Markdown 校对回灌）；
        传入列表（可为空）：用它替换该文档的全部图片。tables：Excel / CSV 的结构化行，进 SQL 检索路。"""
        if doc_id in self.store.docs:
            self.remove_document(doc_id, keep_images=True)
        self.store.add_document(doc_id, metadata)
        levels: list[Chunk] = []
        if self.settings.chunking.multi_granularity:
            from ..ingest.chunking.hierarchy import build_hierarchy   # 延迟导入：ingest 包依赖 index 包

            levels = build_hierarchy(doc_id, chunks, self.settings.chunking)
        self.store.add_chunks(chunks + levels)
        self._add_aux(chunks, levels)
        children = [c for c in chunks if c.kind == "child"]
        if children:
            texts = [c.embed_text() for c in children]
            ids = [c.chunk_id for c in children]
            vecs = self.embedder.embed_documents(texts)
            if getattr(self.vectors, "stores_text", False):   # LanceDB：正文与向量同表，原生全文 / 混合检索直接可用
                self.vectors.add(ids, vecs, texts=texts, doc_ids=[doc_id] * len(ids))
            else:
                self.vectors.add(ids, vecs)
            self.bm25.add(ids, texts)
        if self.settings.retrieval.graph.unweaver_enabled:
            relations = []
            parents = [c for c in chunks if c.kind == "parent"]
            for p in parents:
                relations.extend(extract_definitions(p.text, p.chunk_id, doc_id))
            if self.entity_extractor is not None and parents:
                # 每个 parent 一次 LLM 调用，并发跑；结果按 parent 顺序合并，保证可复现
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=min(6, len(parents))) as pool:
                    for found in pool.map(lambda p: self.entity_extractor.extract(p.text, p.chunk_id, doc_id), parents):
                        relations.extend(found)
            self.entities.add(relations)
        if tables:
            link_chunks(tables, chunks)
            self.tables.add(tables)
        if images is not None:
            self.images.replace_document(doc_id, metadata.title or doc_id, images, chunks)
        else:
            self.images.relink(doc_id, chunks)
        if self.images.embedder_loaded and self.images.embedder.cross_modal:
            # 图搜文直达路：正文块在图文空间里的文本向量。只补本文档的（模型已加载时顺手做）；
            # 其余文档留给查询时分批补建，避免一次上传就把整库重新编码一遍
            self.images.add_chunk_vectors(chunks)

    def _add_aux(self, chunks: list[Chunk], levels: list[Chunk]) -> None:
        summaries = [c for c in chunks if c.kind == "parent" and c.summary]
        if summaries:
            texts = [(f"[来源：{c.breadcrumb_text}]\n" if c.breadcrumbs else "") + c.summary for c in summaries]
            self.aux["summary"].add([c.chunk_id for c in summaries], texts, self.embedder.embed_documents(texts))
        for kind in ("section", "sentence"):
            units = [c for c in levels if c.kind == kind]
            if units:
                # 章节级文本可能很长：只拿开头进向量（语义主旨），BM25 用全文
                texts = [c.embed_text() for c in units]
                vec_texts = [t[:1500] for t in texts] if kind == "section" else texts
                self.aux[kind].add([c.chunk_id for c in units], texts, self.embedder.embed_documents(vec_texts))

    def remove_document(self, doc_id: str, *, keep_images: bool = False) -> None:
        if not keep_images:
            self.images.remove_document(doc_id)
        self.tables.remove_document(doc_id)
        removed = self.store.remove_document(doc_id)
        if not removed:
            self.entities.remove_document(doc_id)
            return
        removed_set = set(removed)
        for aux in self.aux.values():
            aux.remove(removed_set)
        self.vectors.remove(removed)
        new_bm25 = BM25Store(self.settings.sparse)
        for cid, toks in zip(self.bm25.ids, self.bm25.corpus_tokens):
            if cid not in removed_set:
                new_bm25.ids.append(cid)
                new_bm25.corpus_tokens.append(toks)
        new_bm25._dirty = True
        self.bm25 = new_bm25
        self.entities.remove_document(doc_id)
        self.images.chunk_clip.remove(removed)

    def remove_chunks(self, parent_ids: set[str]) -> list[str]:
        """语料治理：把若干 parent 连同其 child / 句子级单元移出检索（向量、BM25、辅助索引、块存储）。"""
        ids = set()
        for pid in parent_ids:
            p = self.store.get(pid)
            if p is None:
                continue
            ids.add(pid)
            ids.update(p.child_ids)
        ids.update(c.chunk_id for c in self.store.chunks.values() if c.parent_id in ids)
        # 章节级单元（多粒度索引）通过 child_ids 引用 parent，正文里含被剔除段落：一并移出，重新解析时按剔除后的段落重建
        ids.update(c.chunk_id for c in self.store.chunks.values() if c.kind == "section" and ids.intersection(c.child_ids))
        removed = self.store.remove_chunks(ids)
        if not removed:
            return []
        rm = set(removed)
        self.vectors.remove(removed)
        self.bm25.remove(rm)
        for aux in self.aux.values():
            aux.remove(rm)
        self.images.chunk_clip.remove(removed)
        return removed

    def sparse_query(self, query: str) -> str:
        """BM25 查询扩展：查询里出现别名词典中的词，补上同组的其余词。"""
        return self.aliases.expand_query(query) if self.aliases else query

    # ----------------------------------------------------------------------------------
    def signature(self) -> dict:
        return {"embedder": self.embedder.name, "model": getattr(self.embedder, "model_name", None), "dim": getattr(self.vectors, "dim", None), "vector_store": getattr(self.vectors, "name", "numpy")}

    def save(self, directory: Path | None = None) -> Path:
        directory = Path(directory or self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.vectors.save(directory)
        self.bm25.save(directory)
        self.store.save(directory)
        self.entities.save(directory)
        self.images.save(directory)
        self.tables.save(directory)
        for aux in self.aux.values():
            aux.save(directory)
        children = sum(1 for c in self.store.chunks.values() if c.kind == "child")
        parents = sum(1 for c in self.store.chunks.values() if c.kind == "parent")
        meta = {**self.signature(), "chunks": len(self.store), "children": children, "parents": parents, **{f"aux_{k}": len(v) for k, v in self.aux.items()}, "aux_store": {k: v.backend for k, v in self.aux.items()}, "docs": len(self.store.docs), "entities": len(self.entities), "images": len(self.images)}
        (directory / "index_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.settings.multimodal.lance_table:
            self._sync_lance_mm(directory)
        return directory

    def _sync_lance_mm(self, directory: Path) -> None:
        """多模态检索接入了 LanceDB 多模态表（multimodal.lance_table）：保存索引时顺手增量同步它（只写变了的行）。
        它是派生数据：同步失败不影响保存，查询前发现没同步上还会再补。"""
        try:
            from ..lance.multimodal import LanceMultimodalTable

            table = LanceMultimodalTable(directory / "lancedb", keep_versions_days=self.settings.vector_store.lancedb.keep_versions_days)
            table.sync_from_index(self, embed="new")
        except Exception as exc:
            logger.warning("LanceDB 多模态表同步失败（不影响索引保存）：%s", exc)

    @classmethod
    def read_meta(cls, directory: Path) -> dict | None:
        p = Path(directory) / "index_meta.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    @classmethod
    def load(cls, settings: Settings, directory: Path | None = None, embedder: BaseEmbedder | None = None, llm=None, strict: bool = True, image_embedder: "BaseImageEmbedder | None" = None, image_embedder_factory: "Callable[[], BaseImageEmbedder] | None" = None) -> "Index":
        from ..multimodal.image_index import ImageIndex

        directory = Path(directory or settings.paths.index_dir)
        idx = cls(settings, embedder=embedder, directory=directory, llm=llm, image_embedder=image_embedder, image_embedder_factory=image_embedder_factory)
        meta = cls.read_meta(directory)
        if meta:
            built_with = (meta.get("embedder"), meta.get("model"))
            current = (idx.embedder.name, getattr(idx.embedder, "model_name", None))
            if built_with != current:
                msg = (
                    f"索引由 embedder={built_with[0]} model={built_with[1]} 构建，当前配置为 embedder={current[0]} model={current[1]}；"
                    "向量空间不一致会得到无意义的检索结果。请用相同配置查询，或重建索引。"
                )
                if strict:
                    raise RuntimeError(msg)
                logger.warning(msg)
        if (directory / "vectors.npy").exists() and isinstance(idx.vectors, NumpyVectorStore):
            idx.vectors = NumpyVectorStore.load(directory)
            idx._tune_vectors()
        dim = idx.embedder.dim or settings.embedding.dim
        lance_cfg = settings.vector_store.lancedb if settings.vector_store.backend == "lancedb" else None
        idx.aux = {name: AuxIndex.load(name, directory, dim, settings.sparse, lance_cfg=lance_cfg) for name in AUX_NAMES}
        idx.bm25 = BM25Store.load(directory, settings.sparse)
        idx.store = ChunkStore.load(directory)
        idx.entities = EntityStore.load(directory)
        idx.tables = TableStore.load(directory)
        idx.images = ImageIndex.load(directory, embedder_factory=idx._image_embedder, text_embedder=idx.embedder, sparse_cfg=settings.sparse, lance_cfg=lance_cfg)
        return idx

    @classmethod
    def exists(cls, directory: Path) -> bool:
        return (Path(directory) / "chunks.jsonl").exists()

    def stats(self) -> dict:
        parents = sum(1 for c in self.store.chunks.values() if c.kind == "parent")
        children = sum(1 for c in self.store.chunks.values() if c.kind == "child")
        return {"docs": len(self.store.docs), "parents": parents, "children": children, **{f"aux_{k}": len(v) for k, v in self.aux.items()}, "vectors": len(self.vectors), "bm25_docs": len(self.bm25.ids), "entities": len(self.entities), "tables": len(self.tables), **self.images.stats(), **self.signature()}
