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
import threading
import time
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


_KBS_KEY = "okf:kbs"              # 导入过 OKF 知识包的知识库列表（自动同步按它轮询）
_AUTO_KEY = "okf:auto_sync"        # {"minutes": N}：0 = 关闭


class OKFService:
    auto_sync_tick = 30.0            # 后台线程每隔多少秒看一次配置

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sync_lock = threading.Lock()
        self._auto_thread: threading.Thread | None = None
        self._auto_stop = threading.Event()
        self._auto_last: dict[str, Any] | None = None

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
            repo.set_setting(_KBS_KEY, sorted({*(repo.get_setting(_KBS_KEY) or []), kb_id}))
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
            repo = Repository(db)
            repo.set_setting(self._setting_key(kb_id), None)
            repo.set_setting(_KBS_KEY, [k for k in (repo.get_setting(_KBS_KEY) or []) if k != kb_id])
        p = self._manifest_path(kb_id)
        p.unlink(missing_ok=True)
        return {"removed": removed}


    # --- 自动同步（Git 来源）------------------------------------------------------------
    def auto_sync_settings(self) -> dict[str, Any]:
        with SessionLocal() as db:
            cfg = Repository(db).get_setting(_AUTO_KEY) or {}
        return {"minutes": int(cfg.get("minutes") or 0), "running": self.auto_sync_running(), "last": self._auto_last}

    def update_auto_sync(self, minutes: int) -> dict[str, Any]:
        minutes = max(0, min(1440, int(minutes)))
        with SessionLocal() as db:
            Repository(db).set_setting(_AUTO_KEY, {"minutes": minutes})
        return self.auto_sync_settings()

    def start_auto_sync(self) -> None:
        """应用启动时调用：后台线程按 okf:auto_sync.minutes 定时检查远端提交（0 = 关闭，线程照跑、只看配置）。"""
        with self._lock:
            if self.auto_sync_running():
                return
            self._auto_stop = threading.Event()
            self._auto_thread = threading.Thread(target=self._auto_sync_loop, args=(self._auto_stop,), name="okf-auto-sync", daemon=True)
            self._auto_thread.start()

    def stop_auto_sync(self) -> None:
        self._auto_stop.set()

    def auto_sync_running(self) -> bool:
        return self._auto_thread is not None and self._auto_thread.is_alive() and not self._auto_stop.is_set()

    def _auto_sync_loop(self, stop: threading.Event) -> None:
        last: float | None = None
        while not stop.wait(self.auto_sync_tick):
            try:
                minutes = self.auto_sync_settings()["minutes"]
                if minutes <= 0:
                    last = None
                    continue
                if last is not None and time.monotonic() - last < minutes * 60:
                    continue
                last = time.monotonic()
                self.auto_sync_round()
            except Exception:
                logger.exception("OKF 自动同步出错")

    @staticmethod
    def remote_commit(source: str, ref: str | None) -> str | None:
        """远端分支 / 标签当前的提交号（git ls-remote，不下载内容）；拿不到返回 None。"""
        try:
            out = _git(["ls-remote", source, ref or "HEAD"], timeout=120)
        except ValueError as exc:
            logger.warning("OKF：查询远端提交失败 %s：%s", source, exc)
            return None
        for line in out.splitlines():
            sha, _, name = line.partition("\t")
            if ref is None or name in (f"refs/heads/{ref}", f"refs/tags/{ref}", ref):
                return sha.strip() or None
        return None

    def auto_sync_round(self) -> dict[str, Any] | None:
        """同步一轮：Git 来源的知识库先比远端提交号，和上次导入的不同才重新导入（本地目录来源不参与自动同步）。"""
        if not self._sync_lock.acquire(blocking=False):
            logger.info("OKF 自动同步：上一轮还没结束，跳过")
            return None
        try:
            with SessionLocal() as db:
                registered = set(Repository(db).get_setting(_KBS_KEY) or [])
            # 登记表之外再按磁盘上的清单找一遍：登记表是后加的，之前导入过的知识库不在里面；清单有、来源也在，就该轮询
            on_disk = {p.parent.name for p in (app_config.data_dir / "okf").glob("*/manifest.json")}
            kb_ids = sorted(registered | on_disk)
            started = datetime.now(timezone.utc)
            out: dict[str, Any] = {"kbs": 0, "checked": 0, "updated": 0, "unchanged": 0, "errors": []}
            for kb_id in kb_ids:
                src = self.source_of(kb_id)
                if not src or not is_git_source(src["source"]):
                    continue
                out["kbs"] += 1
                remote = self.remote_commit(src["source"], src.get("ref"))
                if remote is None:
                    out["errors"].append(f"{kb_id}: 查不到远端提交")
                    continue
                out["checked"] += 1
                if remote == self.manifest(kb_id).get("commit"):
                    out["unchanged"] += 1
                    continue
                try:
                    r = self.import_bundle(kb_id, src["source"], ref=src.get("ref"), subdir=src.get("subdir"))
                    out["updated"] += 1
                    out["errors"] += [f"{kb_id}: {f}" for f in r.get("failed", [])]
                except Exception as exc:
                    logger.exception("OKF 自动同步失败：知识库 %s", kb_id)
                    out["errors"].append(f"{kb_id}: {type(exc).__name__}: {exc}"[:300])
            out["errors"] = out["errors"][:10]
            self._auto_last = {"at": started.isoformat(), **out}
            if out["updated"] or out["errors"]:
                logger.info("OKF 自动同步：检查 %d 个知识库，重新导入 %d 个，错误 %d", out["checked"], out["updated"], len(out["errors"]))
            return self._auto_last
        finally:
            self._sync_lock.release()


okf_service = OKFService()
