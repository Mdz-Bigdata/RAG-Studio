"""批量导入：把大量文档一次性灌进知识库（千万级段落用它，不要一篇篇在网页上传）。

输入两种：
- JSONL：每行一篇文档 {"id": 可选, "title": ..., "text": 正文（Markdown / 纯文本）, 以及可选的元数据
  "source" / "version" / "valid_from" / "valid_to" / "tags"（列表或逗号分隔）/ "department" / "access_level" / "allowed_departments"}
- 目录：按文件类型解析（与网页上传同一套解析器），不处理文档里的图片。

切片与网页上传完全一致；向量化跨文档成批做（--embed-workers > 1 时并发请求 API 向量模型），jieba 分词多进程，
写入成批追加，ANN / 全文索引最后一次建好。中断后重跑同一条命令即可续传（已导入的文档跳过，写到一半的那批会先清掉）。

用法（项目根目录）：
    # 导入 Web 应用里的知识库：用应用当前的策略配置与激活的向量模型；文档会出现在「知识库」页
    .venv/bin/python scripts/bulk_import.py --kb <kb_id> docs.jsonl --embed-workers 8 --tokenize-workers 8
    # 导入到独立的索引目录（CLI / 压测用），配置来自 config.yaml
    .venv/bin/python scripts/bulk_import.py --index-dir data/index_big docs_dir/
    # 批量重建 Web 知识库（改了入库参数 / 换了向量模型，而知识库太大不自动重建时）：按文档行里保存的正文 / 上传文件
    # 重新入库到 <索引目录>.rebuild，建好后替换；旧索引改名留底（确认无误后自行删除）。重建期间不要往这个知识库上传
    .venv/bin/python scripts/bulk_import.py --kb <kb_id> --rebuild

千万级知识库先把策略配置改成 config.yaml 里 storage 段注释的那组（SQLite 块存储 + LanceDB + jieba 全文索引），
否则导入后检索仍会把全部段落装进内存。导入 Web 知识库期间不要在网页上改这个知识库（两个进程同时写同一个索引）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import uuid
from collections.abc import Iterator
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("bulk_import")
META_KEYS = ("source", "version", "valid_from", "valid_to", "tags", "department", "access_level", "allowed_departments")


def stable_id(namespace: str, key: str) -> str:
    """同一份输入每次得到同一个文档 id：重跑时能认出已经导入过的文档。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag-studio:{namespace}:{key}"))


def metadata_of(rec: dict, title: str, file_format: str):
    from prism.schemas import DocMetadata

    md = DocMetadata(title=title, file_format=file_format)
    for key in META_KEYS:
        v = rec.get(key)
        if v in (None, ""):
            continue
        if key in ("valid_from", "valid_to"):
            v = date.fromisoformat(str(v)[:10])
        elif key in ("tags", "allowed_departments") and isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        elif key == "access_level":
            v = max(0, min(3, int(v)))
        setattr(md, key, v)
    return md


def jsonl_records(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                logger.warning("第 %d 行不是合法 JSON，跳过", n)
                continue
            if not (rec.get("text") or "").strip():
                continue
            yield rec


def swap(live: Path, staged: Path) -> None:
    """建好的新索引换上去；旧索引改名留底。运行中的服务按 index_meta.json 的修改时间发现变化，下次访问时加载新索引。"""
    if live.exists():
        old = live.with_name(f"{live.name}.old-{time.strftime('%Y%m%d%H%M%S')}")
        live.rename(old)
        logger.info("旧索引留底：%s（确认无误后自行删除）", old)
    staged.rename(live)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, nargs="?", help="JSONL 文件或文档目录（--rebuild 时不需要）")
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--kb", help="Web 应用的知识库 id")
    target.add_argument("--index-dir", type=Path, help="独立索引目录（配置来自 --config）")
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--batch-chunks", type=int, default=4096, help="攒够这么多个 child 写一批")
    ap.add_argument("--embed-batch", type=int, default=256, help="每次向量化请求的条数")
    ap.add_argument("--embed-workers", type=int, default=1, help="并发向量化请求数（API 向量模型时调大；本地模型保持 1）")
    ap.add_argument("--tokenize-workers", type=int, default=os.cpu_count() or 1, help="jieba 分词进程数（LanceDB jieba 全文索引时用）")
    ap.add_argument("--limit", type=int, default=0, help="只导入前 N 篇（试跑）")
    ap.add_argument("--rebuild", action="store_true", help="按知识库已有的文档行整库重建（需要 --kb）")
    args = ap.parse_args()
    if args.rebuild and not args.kb:
        ap.error("--rebuild 需要 --kb")
    if not args.rebuild and args.input is None:
        ap.error("需要输入文件或目录")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from prism.index import Index, build_embedder
    from prism.ingest import IngestPipeline
    from prism.ingest.bulk import BulkIndexer

    on_commit = None
    if args.kb:
        from app.config import app_config
        from app.db import Document, Repository, SessionLocal
        from app.services.index_service import index_service
        from app.services.model_service import model_service
        from app.services.settings_service import settings_service

        with SessionLocal() as db:
            if Repository(db).get_kb(args.kb) is None:
                sys.exit(f"知识库不存在：{args.kb}")
            registered = {d for (d,) in db.query(Document.id).filter(Document.kb_id == args.kb)}
        settings = settings_service.get()
        embedder = model_service.embedder()
        live_dir = index_service.directory(args.kb)
        directory = live_dir.with_name(live_dir.name + ".rebuild") if args.rebuild else live_dir
        namespace = args.kb
        upload_dir = app_config.upload_dir / args.kb
        texts: dict[str, str] = {}
        files: dict[str, Path] = {}

        def on_commit(batch) -> None:
            if args.rebuild:                # 重建：文档行本来就在，只更新段落数（切片参数可能改了）
                with SessionLocal() as db:
                    for doc_id, _md, chunks in batch:
                        db.query(Document).filter(Document.id == doc_id).update({"chunk_count": sum(1 for c in chunks if c.kind == "child"),
                                                                                 "parent_count": sum(1 for c in chunks if c.kind == "parent")})
                    db.commit()
                return
            """登记文档行：整库重建时要靠 file_path（目录导入：硬链接进上传目录）或 markdown_content（JSONL 正文）重新入库。"""
            with SessionLocal() as db:
                for doc_id, md, chunks in batch:
                    src = files.pop(doc_id, None)
                    path = None
                    if src is not None:
                        upload_dir.mkdir(parents=True, exist_ok=True)
                        path = upload_dir / f"{doc_id}{src.suffix.lower()}"
                        if not path.exists():
                            try:
                                os.link(src, path)          # 同一块磁盘：不占额外空间
                            except OSError:
                                shutil.copy2(src, path)
                    db.merge(Document(
                        id=doc_id, kb_id=args.kb, file_name=(src.name if src else (md.title or doc_id))[:255], file_path=str(path) if path else None,
                        file_size=path.stat().st_size if path else None, file_format=md.file_format, status="COMPLETED", parser=md.parser,
                        markdown_content=texts.pop(doc_id, None), chunk_count=sum(1 for c in chunks if c.kind == "child"),
                        parent_count=sum(1 for c in chunks if c.kind == "parent"), title=(md.title or "")[:255] or None, version=md.version,
                        valid_from=md.valid_from.isoformat() if md.valid_from else None, valid_to=md.valid_to.isoformat() if md.valid_to else None,
                        tags=",".join(md.tags) or None, department=md.department, access_level=md.access_level,
                        allowed_departments=",".join(md.allowed_departments) or None))
                    registered.add(doc_id)
                db.commit()
    else:
        from prism.config import Settings

        settings = Settings.load(args.config)
        embedder = build_embedder(settings.embedding)
        directory = args.index_dir
        namespace = str(args.index_dir.resolve())
        registered = None
        texts, files = {}, {}

    if settings.storage.chunk_store != "sqlite" or settings.vector_store.backend != "lancedb":
        logger.warning("当前配置不是千万级配置（storage.chunk_store=%s, vector_store.backend=%s）：导入能完成，但检索时全部段落会装进内存",
                       settings.storage.chunk_store, settings.vector_store.backend)
    index = Index.load(settings, directory, embedder=embedder, strict=False) if Index.exists(directory) else Index(settings, embedder=embedder, directory=directory)
    pipeline = IngestPipeline(settings, index=index)
    bulk = BulkIndexer(index, batch_chunks=args.batch_chunks, embed_batch=args.embed_batch, embed_workers=args.embed_workers,
                       tokenize_workers=args.tokenize_workers, on_commit=on_commit)

    def done(doc_id: str) -> bool:
        return bulk.has(doc_id) and (registered is None or args.rebuild or doc_id in registered)

    def from_input():
        """(文档 id, 显示名, 解析函数)：JSONL 每行一篇 / 目录里每个文件一篇。"""
        if args.input.is_file():
            for rec in jsonl_records(args.input):
                key = str(rec.get("id") or rec.get("source") or hashlib.sha1(((rec.get("title") or "") + "\n" + rec["text"]).encode("utf-8")).hexdigest())
                yield stable_id(namespace, key), key, lambda doc_id, rec=rec: parse_record(doc_id, rec, None)
        else:
            exts = pipeline.registry.supported_extensions()
            for p in sorted(args.input.rglob("*")):
                if p.is_file() and p.suffix.lower().lstrip(".") in exts:
                    yield stable_id(namespace, str(p)), str(p), lambda doc_id, p=p: parse_record(doc_id, {"source": str(p)}, p)

    def parse_record(doc_id: str, rec: dict, path: Path | None):
        if path is None:
            title = (rec.get("title") or rec["text"].strip().splitlines()[0])[:200]
            md = metadata_of(rec, title, "md")
            doc = pipeline.parse_markdown_text(rec["text"], doc_id=doc_id, source=rec.get("source") or title, metadata=md)
            texts[doc_id] = rec["text"]
        else:
            md = metadata_of(rec, path.stem, path.suffix.lower().lstrip("."))
            doc = pipeline.parse(path, md)
            doc.doc_id = doc_id
            files[doc_id] = path
        for key in META_KEYS:          # 记录里显式给的元数据优先于文件自带的（Markdown front matter 等）
            if rec.get(key) not in (None, ""):
                setattr(doc.metadata, key, getattr(md, key))
        return doc

    def from_kb():
        """重建：按文档行逐篇读（yield_per：几十万行不一次装进内存）。校对过的 / 没有原文件的用保存的 Markdown，否则重新解析上传的文件——
        与 document_service.reindex_kb 的选择一致。"""
        from app.services.document_service import _doc_metadata

        with SessionLocal() as db:
            for d in db.query(Document).filter(Document.kb_id == args.kb, Document.status == "COMPLETED").yield_per(500):
                md = _doc_metadata(d)
                edited, markdown, path = "edited" in (d.parser or ""), d.markdown_content, Path(d.file_path) if d.file_path else None

                def parse(doc_id, md=md, edited=edited, markdown=markdown, path=path, name=d.file_name):
                    if (edited or path is None or not path.exists()) and markdown:
                        return pipeline.parse_markdown_text(markdown, doc_id=doc_id, source=name, metadata=md)
                    if path is None or not path.exists():
                        raise FileNotFoundError("既没有保存的正文，也找不到上传的文件")
                    doc = pipeline.parse(path, md)
                    doc.doc_id = doc_id
                    return doc
                yield d.id, d.file_name, parse

    t0 = time.perf_counter()
    seen = skipped = failed = 0
    last_log = t0
    for doc_id, key, parse in (from_kb() if args.rebuild else from_input()):
        if args.limit and seen >= args.limit:
            break
        seen += 1
        if done(doc_id):
            skipped += 1
            continue
        try:
            doc = parse(doc_id)
            doc, chunks, tables = pipeline.chunk_document(doc)
            bulk.add(doc_id, doc.metadata, chunks, tables=tables)
        except Exception as exc:
            failed += 1
            logger.warning("导入失败 %s：%s: %s", key[:80], type(exc).__name__, exc)
        if time.perf_counter() - last_log > 30:
            last_log = time.perf_counter()
            st = bulk.stats
            logger.info("已处理 %d 篇（跳过 %d、失败 %d），写入 %d 个 child，%.0f child/s", seen, skipped, failed, st.children, st.children / (last_log - t0))
    logger.info("全部读完，写最后一批并建索引…")
    stats = bulk.finish()
    if args.rebuild:
        swap(live_dir, directory)
        from app.services.reindex_service import reindex_service

        reindex_service._mark_pending(args.kb, None)
    elapsed = time.perf_counter() - t0
    print(json.dumps({"input": str(args.input), "index": str(live_dir if args.rebuild else directory), "read": seen, "skipped": skipped, "failed": failed, "elapsedSeconds": round(elapsed, 1),
                      "childPerSecond": round(stats.children / elapsed, 1) if elapsed else None, **stats.as_dict()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
