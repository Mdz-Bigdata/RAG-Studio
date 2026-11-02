"""OKF 知识包导入：Git 仓库或本地目录 → 每个条目一篇知识库文档（带 okf 元数据、「OKF」标签）→ 走普通的解析 / 切片 / 索引。

- 来源是 Git 地址：克隆到 data/okf/<kb_id>/repo/<来源哈希>，再次同步时 fetch + reset 到远端分支（GitOps：改动都在仓库的提交历史里）；
  来源是本地目录：直接扫描，不复制；
- 条目的文档 id 由「知识库 + 条目路径」决定（uuid5），同一条目再次同步仍是同一篇文档：内容哈希没变就不重新入库，
  变了就重写文件、重新解析；知识包里删掉的条目从知识库删除；
- 来源记在应用设置 okf:<kb_id> 里，「知识库」页可以一键重新同步；清单（每个条目的 id / 哈希）记在 data/okf/<kb_id>/manifest.json。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prism.okf import OKFCatalog, okf_markdown, scan_bundle

from ..config import app_config
from ..db import Repository, SessionLocal
from .document_service import document_service
from .index_service import index_service

logger = logging.getLogger(__name__)

_GIT_URL = re.compile(r"^(https?://|git@|ssh://|git://)|\.git$")
_SAFE = re.compile(r"[^\w.\-]+")


def is_git_source(source: str) -> bool:
    return bool(_GIT_URL.search(source.strip()))


def _git(args: list[str], cwd: Path | None = None, timeout: int = 600) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise ValueError("服务器上没有安装 git，无法从 Git 仓库导入 OKF 知识包") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"git {args[0]} 超时（{timeout}s）") from exc
    if r.returncode != 0:
        raise ValueError(f"git {args[0]} 失败：{(r.stderr or r.stdout).strip()[-400:]}")
    return r.stdout


class OKFService:
    def root(self, kb_id: str) -> Path:
        return app_config.data_dir / "okf" / kb_id

    def _manifest_path(self, kb_id: str) -> Path:
        return self.root(kb_id) / "manifest.json"

    def manifest(self, kb_id: str) -> dict[str, Any]:
        p = self._manifest_path(kb_id)
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except ValueError:
            return {}

    def _save_manifest(self, kb_id: str, data: dict[str, Any]) -> None:
        p = self._manifest_path(kb_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _setting_key(kb_id: str) -> str:
        return f"okf:{kb_id}"

    def source_of(self, kb_id: str) -> dict[str, Any] | None:
        with SessionLocal() as db:
            return Repository(db).get_setting(self._setting_key(kb_id))

    # --- 取知识包 -------------------------------------------------------------------------
    def _checkout(self, kb_id: str, source: str, ref: str | None) -> tuple[Path, str | None]:
        """Git 来源：克隆 / 更新到本地，返回 (目录, 提交号)；本地目录：原样返回。"""
        source = source.strip()
        if not is_git_source(source):
            p = Path(source).expanduser()
            if not p.is_dir():
                raise ValueError(f"目录不存在：{source}")
            return p, None
        dest = self.root(kb_id) / "repo" / hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
        if (dest / ".git").is_dir():
            _git(["fetch", "--depth", "1", "origin", *([ref] if ref else [])], cwd=dest)
            _git(["reset", "--hard", "FETCH_HEAD" if ref else "origin/HEAD"], cwd=dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            _git(["clone", "--depth", "1", *(["--branch", ref] if ref else []), source, str(dest)])
        commit = _git(["rev-parse", "HEAD"], cwd=dest).strip()
        return dest, commit

    # --- 导入 / 同步 ----------------------------------------------------------------------
    def import_bundle(self, kb_id: str, source: str, *, ref: str | None = None, subdir: str | None = None) -> dict[str, Any]:
        """导入或重新同步。返回 {created, updated, unchanged, removed, total, commit, entries}。"""
        with SessionLocal() as db:
            if Repository(db).get_kb(kb_id) is None:
                raise ValueError("知识库不存在")
        base, commit = self._checkout(kb_id, source, ref)
        entries = scan_bundle(base, subdir or None)
        if not entries:
            raise ValueError("知识包里没有带 front matter（含 title）的 Markdown 文件")
        manifest = self.manifest(kb_id)
        old: dict[str, dict[str, Any]] = manifest.get("entries") or {}
        folder = app_config.upload_dir / kb_id
        folder.mkdir(parents=True, exist_ok=True)
        counts = {"created": 0, "updated": 0, "unchanged": 0, "removed": 0}
        changed: list[str] = []
        new_entries: dict[str, dict[str, Any]] = {}
        label = f"{source.strip()}" + (f"@{ref}" if ref else "") + (f"/{subdir}" if subdir else "")
        with SessionLocal() as db:
            repo = Repository(db)
            for e in entries:
                doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag-studio:okf:{kb_id}:{e.path}"))
                prev = old.get(e.path)
                doc = repo.get_document(doc_id)
                new_entries[e.path] = {"docId": doc_id, "sha1": e.sha1, "title": e.title, "type": e.type, "owner": e.owner}
                if prev and prev.get("sha1") == e.sha1 and doc is not None and doc.status == "COMPLETED":
                    counts["unchanged"] += 1
                    continue
                path = folder / f"{doc_id}.md"
                path.write_text(okf_markdown(e, label), encoding="utf-8")
                name = f"{_SAFE.sub('_', Path(e.path).name) or 'entry.md'}"
                fields = dict(file_name=name[:255], file_path=str(path), file_size=path.stat().st_size, file_format="md", status="PENDING",
                              title=e.title[:255], tags=",".join(sorted({*e.tags, "OKF"})))
                if doc is None:
                    repo.create_document(id=doc_id, kb_id=kb_id, **fields)
                    counts["created"] += 1
                else:
                    repo.update_document(doc, **fields)
                    counts["updated"] += 1
                changed.append(doc_id)
            repo.set_setting(self._setting_key(kb_id), {"source": source.strip(), "ref": ref or None, "subdir": subdir or None})
        for path, info in old.items():
            if path not in new_entries:
                try:
                    document_service.delete(info["docId"])
                    counts["removed"] += 1
                except KeyError:
                    pass
        failed: list[str] = []
        for doc_id in changed:
            try:
                document_service.process(doc_id, reason="okf")
            except Exception as exc:
                logger.exception("OKF 条目入库失败 %s", doc_id)
                failed.append(f"{doc_id}: {type(exc).__name__}: {exc}"[:200])
        self._save_manifest(kb_id, {"source": source.strip(), "ref": ref or None, "subdir": subdir or None, "commit": commit,
                                    "syncedAt": datetime.now(timezone.utc).isoformat(), "entries": new_entries})
        return {**counts, "total": len(entries), "commit": commit, "failed": failed, "entries": list(new_entries.values())}

    def sync(self, kb_id: str) -> dict[str, Any]:
        src = self.source_of(kb_id)
        if not src:
            raise ValueError("这个知识库还没有导入过 OKF 知识包")
        return self.import_bundle(kb_id, src["source"], ref=src.get("ref"), subdir=src.get("subdir"))

    def status(self, kb_id: str) -> dict[str, Any]:
        m = self.manifest(kb_id)
        src = self.source_of(kb_id)
        entries = list((m.get("entries") or {}).values())
        return {"source": (src or {}).get("source"), "ref": (src or {}).get("ref"), "subdir": (src or {}).get("subdir"), "commit": m.get("commit"),
                "syncedAt": m.get("syncedAt"), "count": len(entries), "entries": entries, "isGit": bool(src and is_git_source(src["source"]))}

    def match(self, kb_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """调试 / 界面预览：这个问题会命中哪些条目。"""
        catalog = OKFCatalog(index_service.get(kb_id).store.docs)
        return [m.__dict__ for m in catalog.match(query, limit=limit)]

    def forget(self, kb_id: str, *, delete_docs: bool = True) -> dict[str, Any]:
        """取消关联：可选把条目文档一并删除。"""
        m = self.manifest(kb_id)
        removed = 0
        if delete_docs:
            for info in (m.get("entries") or {}).values():
                try:
                    document_service.delete(info["docId"])
                    removed += 1
                except KeyError:
                    pass
        with SessionLocal() as db:
            Repository(db).set_setting(self._setting_key(kb_id), None)
        p = self._manifest_path(kb_id)
        p.unlink(missing_ok=True)
        return {"removed": removed}


okf_service = OKFService()
