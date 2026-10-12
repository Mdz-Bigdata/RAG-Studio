"""文档服务：上传 → 后台解析 / 切片 / 索引；Markdown 校对回灌；删除；切片预览；临时解析；整库重建。"""

from __future__ import annotations

import functools
import json
import logging
import re
import shutil
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from prism.corpus.metadata import FIELDS, apply_metadata, extract_metadata
from prism.ingest.parsers.mineru_parser import mineru_available
from prism.ingest.quick_text import extract_text
from prism.schemas import DocMetadata, ParsedDocument

from ..config import app_config
from ..db import Document, Repository, SessionLocal
from .index_service import index_service
from .settings_service import settings_service

logger = logging.getLogger(__name__)


def _doc_metadata(doc: Document) -> DocMetadata:
    md = DocMetadata(source=doc.file_path or doc.file_name, file_format=(doc.file_format or ""), title=doc.title or Path(doc.file_name).stem, version=doc.version)
    md.valid_from = _date(doc.valid_from)
    md.valid_to = _date(doc.valid_to)
    md.tags = [t for t in (doc.tags or "").split(",") if t]
    md.department = doc.department or None
    md.access_level = int(doc.access_level or 0)
    md.allowed_departments = [t for t in (doc.allowed_departments or "").split(",") if t]
    return md


def _keep_manual(parsed: DocMetadata, saved: DocMetadata, stored_auto: dict) -> None:
    """语料治理里人工改过的字段（auto_meta 里 source=manual）优先于文件自带的元数据（如 Markdown front matter）——
    否则重新解析 / 整库重建会把人工修改悄悄改回文件里的值。"""
    for key, v in stored_auto.items():
        if key in FIELDS and isinstance(v, dict) and v.get("source") == "manual":
            setattr(parsed, key, getattr(saved, key))


def _loads(raw: str | None) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except ValueError:
        return {}


def _date(v: str | None) -> date | None:
    try:
        return date.fromisoformat(v) if v else None
    except ValueError:
        return None


def _lake_event(doc_id: str, reason: str, before: str | None, after: str | None = None, **kw) -> None:
    """文档变更记进 Agent 湖（doc_event）。后台线程、失败只记日志：不能因为湖写不进去而影响文档处理。"""
    try:
        from .lake_service import lake_service

        lake_service.record_doc_event_async(doc_id, reason, before, after, **kw)
    except Exception as exc:
        logger.debug("Agent 湖文档变更记录跳过：%s", exc)


def _writes_index(method):
    """整个调用登记为"本进程正在写这个知识库的索引"（见 DocumentService.index_writes）。"""

    @functools.wraps(method)
    def wrapper(self: DocumentService, doc_id: str, *args: Any, **kwargs: Any):
        with SessionLocal() as db:
            doc = Repository(db).get_document(doc_id)
            kb_id = doc.kb_id if doc is not None else None
        with self._writing(kb_id):
            return method(self, doc_id, *args, **kwargs)

    return wrapper


class DocumentService:
    def __init__(self) -> None:
        # 本进程在途的入库 / 删除：kb_id -> [开始次数, 结束次数]。上个进程（热重载 / 崩溃）留下的 PROCESSING 不在其中
        self._writes: dict[str, list[int]] = {}
        self._idle_hooks: dict[str, list[Callable[[], None]]] = {}
        self._writes_lock = threading.Lock()

    @contextmanager
    def _writing(self, kb_id: str | None) -> Iterator[None]:
        if kb_id is None:
            yield
            return
        with self._writes_lock:
            self._writes.setdefault(kb_id, [0, 0])[0] += 1
        try:
            yield
        finally:
            with self._writes_lock:
                counts = self._writes[kb_id]
                counts[1] += 1
                hooks = self._idle_hooks.pop(kb_id, []) if counts[0] == counts[1] else []
            for hook in hooks:
                try:
                    hook()
                except Exception:
                    logger.exception("知识库 %s 写入结束后的回调失败", kb_id)

    def index_writes(self, kb_id: str) -> tuple[int, int]:
        """本进程对这个知识库的入库 / 删除（开始次数, 结束次数）。从开始入库到状态落库（COMPLETED / FAILED）都算在途：
        这期间数据库里的文档状态与索引文件可能对不上。两次读数相同且两数相等 = 两次读之间没有写入，也没有进行中的写入。"""
        with self._writes_lock:
            started, finished = self._writes.get(kb_id, (0, 0))
        return started, finished

    def when_idle(self, kb_id: str, hook: Callable[[], None]) -> None:
        """这个知识库在本进程里没有在途写入时调用 hook：现在就空闲则立即调用，否则在最后一个写入结束时（在那个线程里）调用。"""
        with self._writes_lock:
            started, finished = self._writes.get(kb_id, (0, 0))
            if started != finished:
                self._idle_hooks.setdefault(kb_id, []).append(hook)
                return
        hook()

    # --- 上传 / 处理 ----------------------------------------------------------------
    def save_upload(self, kb_id: str, filename: str, fileobj) -> Document:
        doc_id = str(uuid.uuid4())
        ext = Path(filename).suffix.lower()
        target_dir = app_config.upload_dir / kb_id
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{doc_id}{ext}"
        with path.open("wb") as f:
            shutil.copyfileobj(fileobj, f)
        with SessionLocal() as db:
            return Repository(db).create_document(id=doc_id, kb_id=kb_id, file_name=filename, file_path=str(path), file_size=path.stat().st_size, file_format=ext.lstrip("."), status="PENDING", title=Path(filename).stem)

    @_writes_index
    def process(self, doc_id: str, *, reason: str = "parse") -> None:
        """后台任务：解析 → 清洗 → 切片 → 建索引 → 更新状态。reason 记进 Agent 湖的文档变更（upload / reprocess / reprocess_source / rebuild …）。"""
        with SessionLocal() as db:
            repo = Repository(db)
            doc = repo.get_document(doc_id)
            if doc is None:
                return
            before_md = doc.markdown_content
            repo.update_document(doc, status="PROCESSING", error_message=None)
            kb_id, path = doc.kb_id, Path(doc.file_path)
            metadata = _doc_metadata(doc)
            stored_auto = _loads(doc.auto_meta)
            stem = Path(doc.file_name).stem
        auto: dict = {}
        try:
            with index_service.lock(kb_id):
                pipeline = index_service.pipeline(kb_id)
                parsed = pipeline.parse(path, metadata)
                parsed.doc_id = doc_id
                _keep_manual(parsed.metadata, metadata, stored_auto)
                auto = self._auto_metadata(parsed, stored_auto, stem)
                chunks = pipeline.ingest_document(parsed, source_path=path)
                index_service.save(kb_id, pipeline.index)
            entities = sum(1 for r in pipeline.index.entities.relations if r.doc_id == doc_id)
            with SessionLocal() as db:
                repo = Repository(db)
                doc = repo.get_document(doc_id)
                if doc:
                    repo.update_document(
                        doc,
                        status="COMPLETED",
                        parser=parsed.metadata.parser,
                        parser_tier=parsed.metadata.parser_tier,
                        markdown_content=parsed.to_markdown(),
                        chunk_count=sum(1 for c in chunks if c.kind == "child"),
                        parent_count=sum(1 for c in chunks if c.kind == "parent"),
                        entity_count=entities,
                        title=parsed.metadata.title or doc.title,
                        version=parsed.metadata.version or doc.version,
                        valid_from=parsed.metadata.valid_from.isoformat() if parsed.metadata.valid_from else doc.valid_from,
                        valid_to=parsed.metadata.valid_to.isoformat() if parsed.metadata.valid_to else doc.valid_to,
                        tags=",".join(parsed.metadata.tags) if parsed.metadata.tags else doc.tags,
                        department=parsed.metadata.department or doc.department,
                        access_level=max(int(doc.access_level or 0), int(parsed.metadata.access_level or 0)),
                        auto_meta=json.dumps(auto, ensure_ascii=False) if auto else doc.auto_meta,
                    )
        except Exception as exc:
            logger.exception("文档处理失败 %s", doc_id)
            with SessionLocal() as db:
                repo = Repository(db)
                doc = repo.get_document(doc_id)
                if doc:
                    repo.update_document(doc, status="FAILED", error_message=f"{type(exc).__name__}: {exc}"[:900])
            _lake_event(doc_id, reason, before_md, before_md, status="failed", error=f"{type(exc).__name__}: {exc}"[:300])
            return
        _lake_event(doc_id, reason, before_md)

    @staticmethod
    def _auto_metadata(parsed: ParsedDocument, stored: dict, stem: str) -> dict:
        """自动元数据：只填空字段，人工值不覆盖。已经抽取过的文档（重新解析 / 整库重建）复用上次结果，不重复调用模型。"""
        cfg = settings_service.get().corpus
        if not cfg.auto_metadata:
            return {}
        try:
            if stored:
                found = {k: v for k, v in stored.items() if isinstance(v, dict) and "value" in v}
            else:
                from .model_service import model_service

                found = extract_metadata(parsed, model_service.llm(aux=True) if cfg.auto_metadata_llm else None, file_stem=stem)
            applied = apply_metadata(parsed.metadata, found, file_stem=stem)
        except Exception as exc:
            logger.warning("自动元数据抽取失败 %s：%s", parsed.doc_id, exc)
            return {}
        return {k: {**v, "applied": k in applied or bool(v.get("applied"))} for k, v in found.items()}

    # --- 校对回灌 ---------------------------------------------------------------------
    # --- 校对版备份 -----------------------------------------------------------------
    @staticmethod
    def _backup_folder(doc: Document) -> Path:
        return Path(doc.file_path).parent if doc.file_path else app_config.upload_dir / doc.kb_id

    def backup_edited_markdown(self, doc_id: str) -> Path | None:
        """放弃人工校对（回到源文件重新解析）或恢复另一份备份之前，把当前校对版 Markdown 存到上传目录（<doc_id>.edited-<时间>.md）。"""
        with SessionLocal() as db:
            doc = Repository(db).get_document(doc_id)
            if doc is None or not doc.markdown_content or "edited" not in (doc.parser or ""):
                return None
            folder, md = self._backup_folder(doc), doc.markdown_content
        folder.mkdir(parents=True, exist_ok=True)
        stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
        target = folder / f"{doc_id}.edited-{stamp}.md"
        n = 2
        while target.exists():                    # 同一秒内多次备份
            target = folder / f"{doc_id}.edited-{stamp}-{n}.md"
            n += 1
        target.write_text(md, encoding="utf-8")
        logger.info("已备份校对版 Markdown：%s", target)
        return target

    def _backup_path(self, doc: Document, name: str) -> Path:
        """只接受本文档自己的备份文件名（防路径穿越）。"""
        if not re.fullmatch(re.escape(doc.id) + r"\.edited-\d{8}-\d{6}(-\d+)?\.md", name or ""):
            raise ValueError("无效的备份文件名")
        path = self._backup_folder(doc) / name
        if not path.is_file():
            raise KeyError(name)
        return path

    def edit_backups(self, doc_id: str) -> list[dict[str, Any]]:
        """本文档的校对版备份，新的在前。identical：与当前正文完全相同（恢复它没有意义）。"""
        with SessionLocal() as db:
            doc = Repository(db).get_document(doc_id)
            if doc is None:
                raise KeyError(doc_id)
            folder, current = self._backup_folder(doc), doc.markdown_content or ""
        out = []
        for path in folder.glob(f"{doc_id}.edited-*.md"):
            m = re.search(r"\.edited-(\d{8}-\d{6})", path.name)
            if not m:
                continue
            text = path.read_text(encoding="utf-8")
            out.append({"name": path.name, "createdAt": datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").astimezone(timezone.utc).isoformat(),   # 文件名是本地时间；转成带时区的 UTC，前端按本地显示 "chars": len(text),
                        "preview": text[:600], "identical": text == current})
        return sorted(out, key=lambda b: b["name"], reverse=True)

    def count_edit_backups(self, doc: Document) -> int:
        try:
            return sum(1 for _ in self._backup_folder(doc).glob(f"{doc.id}.edited-*.md"))
        except OSError:
            return 0

    def read_edit_backup(self, doc_id: str, name: str) -> str:
        """恢复前的校验 + 读取；当前若也是校对版，先备份它，恢复本身也可以撤销。"""
        with SessionLocal() as db:
            doc = Repository(db).get_document(doc_id)
            if doc is None:
                raise KeyError(doc_id)
            path = self._backup_path(doc, name)
            current = doc.markdown_content or ""
        text = path.read_text(encoding="utf-8")
        if text != current:
            self.backup_edited_markdown(doc_id)
        return text

    def delete_edit_backup(self, doc_id: str, name: str) -> None:
        with SessionLocal() as db:
            doc = Repository(db).get_document(doc_id)
            if doc is None:
                raise KeyError(doc_id)
            path = self._backup_path(doc, name)
        path.unlink()

    @_writes_index
    def update_markdown(self, doc_id: str, markdown: str, *, reason: str = "markdown") -> Document:
        """用 Markdown（人工校对稿 / 保存的校对版）重建这份文档。reason 记进 Agent 湖的文档变更（edit / restore_backup / rebuild …）。"""
        with SessionLocal() as db:
            repo = Repository(db)
            doc = repo.get_document(doc_id)
            if doc is None:
                raise KeyError(doc_id)
            before_md = doc.markdown_content
            kb_id = doc.kb_id
            metadata = _doc_metadata(doc)
            stored_auto = _loads(doc.auto_meta)
            source = doc.file_path or doc.file_name
            repo.update_document(doc, status="PROCESSING", markdown_content=markdown)
        try:
            with index_service.lock(kb_id):
                pipeline = index_service.pipeline(kb_id)
                parsed = pipeline.parse_markdown_text(markdown, doc_id=doc_id, source=source, metadata=metadata)
                _keep_manual(parsed.metadata, metadata, stored_auto)
                # 正文用校对后的 Markdown；图片平时沿用索引里已有的。索引里没有这份文档的图片时（整库重建会先把旧索引挪走），
                # 从原始文件重新抽取——否则人工校对过的文档在重建后会悄悄丢掉全部图片。
                src = Path(source) if source and Path(source).is_file() else None
                reextract = src is not None and not pipeline.index.images.of_document(doc_id)
                image_doc = None
                if reextract and src.suffix.lower() not in (".md", ".markdown"):
                    # 正文用校对稿，图片的页码 / 图注 / 上下文要从原始文件的结构里取（Markdown 没有页码）
                    try:
                        image_doc = pipeline.parse(src, metadata)
                        image_doc.doc_id = doc_id
                    except Exception as exc:
                        logger.warning("重新抽图时解析原始文件失败（%s），图片将缺少图注与页内上下文", exc)
                chunks = pipeline.ingest_document(parsed, source_path=src if reextract else None, image_doc=image_doc)
                index_service.save(kb_id, pipeline.index)
            entities = sum(1 for r in pipeline.index.entities.relations if r.doc_id == doc_id)
            with SessionLocal() as db:
                repo = Repository(db)
                doc = repo.get_document(doc_id)
                updated = repo.update_document(doc, status="COMPLETED", parser=(doc.parser or "markdown") + "+edited" if "edited" not in (doc.parser or "") else doc.parser, chunk_count=sum(1 for c in chunks if c.kind == "child"), parent_count=sum(1 for c in chunks if c.kind == "parent"), entity_count=entities, error_message=None)
            _lake_event(doc_id, reason, before_md, markdown)
            return updated
        except Exception as exc:
            with SessionLocal() as db:
                repo = Repository(db)
                doc = repo.get_document(doc_id)
                repo.update_document(doc, status="FAILED", error_message=f"{type(exc).__name__}: {exc}"[:900])
            _lake_event(doc_id, reason, before_md, markdown, status="failed", error=f"{type(exc).__name__}: {exc}"[:300])
            raise

    # --- 删除 -----------------------------------------------------------------------
    @_writes_index
    def delete(self, doc_id: str) -> None:
        with SessionLocal() as db:
            repo = Repository(db)
            doc = repo.get_document(doc_id)
            if doc is None:
                raise KeyError(doc_id)
            kb_id, path = doc.kb_id, doc.file_path
            info = {"fileName": doc.file_name, "kbId": doc.kb_id, "parser": doc.parser, "chunks": doc.chunk_count}
            before_md = doc.markdown_content
            repo.delete_document(doc)
        with index_service.lock(kb_id):
            idx = index_service.get(kb_id)
            idx.remove_document(doc_id)
            index_service.save(kb_id, idx)
        _lake_event(doc_id, "delete", before_md, "", info=info)
        if path and Path(path).exists():
            try:
                Path(path).unlink()
            except OSError:
                pass

    # --- 切片预览 ---------------------------------------------------------------------
    def chunks(self, doc_id: str) -> list[dict]:
        with SessionLocal() as db:
            doc = Repository(db).get_document(doc_id)
            if doc is None:
                raise KeyError(doc_id)
            kb_id = doc.kb_id
        idx = index_service.get(kb_id)
        out = []
        for c in sorted((c for c in idx.store.chunks.values() if c.doc_id == doc_id), key=lambda c: c.order):
            out.append(c)
        return [c.model_dump() for c in out]

    # --- 重建整库 ---------------------------------------------------------------------
    def reindex_kb(self, kb_id: str) -> list[str]:
        """整库重建。人工校对过的文档（parser 带 +edited）用保存的 Markdown 重建，不回到原始文件重新解析——
        否则换个向量模型、改个切片参数，校对成果就被原文件覆盖了。

        新索引建在临时目录里（见 index_service.begin_rebuild），重建期间检索照常使用完整的旧索引；
        建好后锁住知识库，把重建期间别处的改动（新上传 / 校对 / 改元数据 / 删除）补进新索引，再原子替换。

        单份文档失败（已标 FAILED）不影响其余文档，返回失败的文件名；全部失败多半是系统性问题（模型加载不了等），
        此时放弃新索引、保留重建前的索引并抛错，不能拿一个空索引替换掉最后一份完整的。"""
        with SessionLocal() as db:
            docs = [(d.id, d.file_path, d.markdown_content, "edited" in (d.parser or "")) for d in Repository(db).list_documents(kb_id)]
        index_service.begin_rebuild(kb_id)
        attempted: set[str] = set()
        seen: dict[str, Any] = {}                 # doc_id -> 本线程入库后的 updated_at；之后再变说明别处改过

        def ingest(doc_id: str, file_path: str | None, markdown: str | None, edited: bool) -> None:
            with SessionLocal() as db:
                if Repository(db).get_document(doc_id) is None:
                    return                        # 重建期间被删除：收尾时会从新索引里移除，不算失败
            try:
                if edited and markdown:
                    attempted.add(doc_id)
                    self.update_markdown(doc_id, markdown)
                elif file_path and Path(file_path).exists():
                    attempted.add(doc_id)
                    self.process(doc_id)
                elif markdown:
                    attempted.add(doc_id)
                    self.update_markdown(doc_id, markdown)
            except Exception:
                logger.exception("重建索引：文档 %s 失败，继续其余文档", doc_id)   # update_markdown 已把它标为 FAILED
            with SessionLocal() as db:
                d = Repository(db).get_document(doc_id)
                seen[doc_id] = getattr(d, "updated_at", None)

        try:
            for doc in docs:
                ingest(*doc)
            # 收尾分两步，加锁顺序固定为 swap_guard → lock（反过来会和检索互相等死，见 index_service._ReadWriteLock）：
            # 1) 只挡写入者，补齐重建期间的改动——可能要重新解析文档，期间检索照常进行；
            with index_service.lock(kb_id):
                self._reconcile_rebuild(kb_id, seen, ingest)
            # 2) 挡住检索并等在途检索结束，再补一次（通常为空）后替换，中间不能再有改动落到旧索引
            with index_service.swap_guard(kb_id), index_service.lock(kb_id):
                self._reconcile_rebuild(kb_id, seen, ingest)
                with SessionLocal() as db:
                    failed = [d.file_name for d in Repository(db).list_documents(kb_id) if d.id in attempted and d.status == "FAILED"]
                if attempted and len(failed) == len(attempted):
                    kept = index_service.abort_rebuild(kb_id)
                    raise RuntimeError(f"全部 {len(failed)} 份文档重建失败" + ("，已保留重建前的索引" if kept else ""))
                index_service.finish_rebuild(kb_id)
        except BaseException:
            index_service.abort_rebuild(kb_id)    # 幂等：已放弃时无事发生
            raise
        return failed

    def _reconcile_rebuild(self, kb_id: str, seen: dict[str, Any], ingest) -> None:
        """重建期间别的线程写的是旧索引：新上传 / 被改过的文档补进新索引，已删除的从新索引移除。"""
        live_docs: set[str] | None = None
        staging = index_service.get(kb_id)        # 当前线程在重建：拿到的是新索引
        with SessionLocal() as db:
            current = {d.id: (d.file_path, d.markdown_content, "edited" in (d.parser or ""), d.status, d.updated_at) for d in Repository(db).list_documents(kb_id)}
        removed = [doc_id for doc_id in list(staging.store.docs) if doc_id not in current]
        for doc_id in removed:
            staging.remove_document(doc_id)
        if removed:
            index_service.save(kb_id, staging)
        for doc_id, (file_path, markdown, edited, status, updated_at) in current.items():
            if doc_id in seen:
                changed = updated_at != seen[doc_id]
            else:                                 # 重建开始后才有的文档：已入库（进了旧索引）的才补
                if live_docs is None:
                    live_docs = set(index_service.live(kb_id).store.docs)
                changed = status == "COMPLETED" or doc_id in live_docs
            if changed:
                logger.info("重建收尾：补入重建期间有改动的文档 %s", doc_id)
                ingest(doc_id, file_path, markdown, edited)

    # --- MinerU-Popo：页面级结果 → 文档树 ------------------------------------------------
    def _popo_paths(self, file_path: str) -> dict[str, Path]:
        """与 IngestPipeline 一致：Popo 侧的文档 id = 源文件绝对路径的哈希。"""
        popo_dir = settings_service.get().paths.popo_dir
        pid = ParsedDocument.make_doc_id(str(Path(file_path).resolve()))
        return {
            "popoDir": popo_dir,
            "normalized": popo_dir / "outputs" / "label_normalization" / "mineru" / f"{pid}.json",
            "inference": popo_dir / "outputs" / "inference" / "mineru" / f"{pid}.json",
            "tree": popo_dir / "outputs" / "build_tree" / "mineru" / f"{pid}.json",
        }

    def popo_status(self, doc_id: str) -> dict:
        with SessionLocal() as db:
            doc = Repository(db).get_document(doc_id)
            if doc is None:
                raise KeyError(doc_id)
            file_path, fmt, parser = doc.file_path, (doc.file_format or ""), (doc.parser or "")
        s = settings_service.get()
        paths = self._popo_paths(file_path) if file_path else {}
        return {
            "applicable": fmt == "pdf",
            "mineruAvailable": mineru_available(),
            "popoEnabled": s.parsing.popo.enabled,
            "exported": bool(paths) and paths["normalized"].exists(),
            "inferred": bool(paths) and paths["inference"].exists(),
            "treeReady": bool(paths) and paths["tree"].exists(),
            "usingTree": "popo" in parser,
            "paths": {k: str(v) for k, v in paths.items()},
            "commands": [
                "POPO_REPO=/path/to/MinerU-Popo POPO_DIR=" + str(paths.get("popoDir", s.paths.popo_dir)) + " bash scripts/run_popo.sh",
                "策略配置 → 解析 → 打开 parsing.popo.enabled，然后对该文档点「重新解析」",
            ],
        }

    def popo_export(self, doc_id: str) -> dict:
        """MinerU 解析 → 写出 Popo 的三套输入（原始 Content List / PDF / 归一化块）。需要已安装 MinerU。"""
        with SessionLocal() as db:
            doc = Repository(db).get_document(doc_id)
            if doc is None:
                raise KeyError(doc_id)
            kb_id, file_path, fmt = doc.kb_id, doc.file_path, (doc.file_format or "")
        if fmt != "pdf" or not file_path or not Path(file_path).exists():
            raise ValueError("MinerU-Popo 需要源 PDF（推理时要渲染页面拼接图）")
        if not mineru_available():
            raise RuntimeError('未安装 MinerU：uv pip install -U "mineru[torch]>=4.0,<5" && mineru-kit models download --tier standard')
        out = index_service.pipeline(kb_id).export_popo_inputs(Path(file_path))
        return {**self.popo_status(doc_id), "written": {k: str(v) for k, v in out.items()}}

    # --- 临时解析（聊天附件） -------------------------------------------------------------
    def parse_temp(self, filename: str, data: bytes) -> str:
        ext = Path(filename).suffix.lower() or ".txt"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            if ext in (".txt", ".md", ".markdown", ".log"):
                try:
                    return data.decode("utf-8")
                except UnicodeDecodeError:
                    return data.decode("gbk", errors="ignore")
            return extract_text(tmp_path, settings_service.get())
        finally:
            tmp_path.unlink(missing_ok=True)


document_service = DocumentService()
