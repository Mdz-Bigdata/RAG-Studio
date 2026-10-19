"""飞书文档服务：应用凭据、同步进知识库、远程增删改查、文档智能体。

- 同步：飞书文档 → 块 → Markdown（带 front matter：标题 / 标签「飞书」/ 来源链接）→ 存成上传目录里的 .md → 走普通的解析 / 切片 / 索引；
  按 revision_id 判断飞书上有没有改过，没改过跳过。之后检索、问答、评测、Agent 湖都把它当普通文档看。
- 图片：read_images 打开时下载文档里的图片，OCR 出图中文字、多模态模型（有的话）给出描述，写在图片位置下面——以截图 / 幻灯片为主的文档
  靠它才能被检索、总结；结果按图片 token 缓存在 data/feishu_media。同步时图片文件也放在 .md 旁边，图片索引（以图搜图 / 文搜图）照常收录。
- 远程增删改查：新建（可直接入库）/ 读 / 追加 / 整篇替换 / 定点改字 / 删除；改动过的文档如果已经入库，写完自动重新同步。
- 智能体：prism.feishu.FeishuAgent + 这里注入的工具；写操作默认只给出待确认的操作，确认后经 execute_action 执行。
- 自动同步：auto_sync_minutes > 0 时后台线程定时对所有接入了飞书文档的知识库跑 sync_kb；配置了事件订阅（Verification Token）时，
  飞书推送的 drive.file.edit_v1 事件只触发那一篇文档的同步。两者共用一把锁，同一时间只有一轮同步在跑。
凭据存在 app_settings（键 feishu），也可用环境变量 FEISHU_APP_ID / FEISHU_APP_SECRET / FEISHU_DOMAIN 兜底；接口只回传脱敏的 Secret / Token。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
import yaml
from pydantic import BaseModel, Field, field_validator

from prism.feishu import FeishuAgent, FeishuClient, FeishuError, Tool, blocks_to_html, blocks_to_markdown, markdown_to_blocks, parse_ref
from prism.feishu.convert import IMAGE, ImageRenderer, block_text, has_table, replace_in_elements, text_field

from ..config import app_config
from ..db import FeishuDoc, Repository, SessionLocal

logger = logging.getLogger(__name__)

SETTING_KEY = "feishu"
MEDIA_DIR = "feishu_media"                      # 图片缓存：data/feishu_media/<token>.<ext>（+ 识别结果 <token>.json）
ASSETS_DIR = "feishu-assets"                    # 同步时图片放在知识库上传目录的 feishu-assets/<document_id>/ 下
MASKED = "********"
SECRET_FIELDS = ("app_secret", "event_verification_token", "event_encrypt_key")     # 接口只回传脱敏值；回传脱敏值表示不修改
SCOPE_MISSING = {99991672, 99991679}           # 应用没开通所需权限
_PROBE = "ScopeProbe000000000000"         # 拼成 27 位的 token：长度不对会先报字段校验错误（99992402），探不到权限
SCOPE_PROBES: tuple[tuple[str, str, Callable[[FeishuClient, Any], Any]], ...] = (     # (名称, 要开通的权限, 探测调用)
    ("drive", "drive:drive（或 drive:drive:readonly）", lambda c, s: c.list_folder(s.folder_token or None)),
    ("docx", "docx:document", lambda c, s: c.get_document("doxcn" + _PROBE)),
    ("wiki", "wiki:wiki:readonly", lambda c, s: c.resolve(parse_ref(f"https://x.feishu.cn/wiki/wikcn{_PROBE}"))),
    ("media", "docs:document.media:download", lambda c, s: c.download_media("boxcn" + _PROBE)),
)
Schedule = Callable[[list[str]], None]          # 把需要解析入库的本地文档 id 交给调用方安排（接口里用 BackgroundTasks）


class FeishuSettings(BaseModel):
    app_id: str = ""
    app_secret: str = ""
    domain: Literal["feishu", "lark"] = "feishu"     # lark = 海外版 open.larksuite.com
    web_base: str = ""                                # 文档链接前缀，如 https://xxx.feishu.cn；留空用 feishu.cn
    folder_token: str = ""                            # 新建文档默认放进的文件夹，也是「浏览文件夹」的默认目录
    default_kb_id: str = ""                           # 智能体 / 新建文档默认同步进的知识库
    use_convert_api: bool = True                      # 写入带表格的 Markdown 时用官方转换接口（保留表格结构）
    agent_allow_write: bool = False                   # 智能体直接写飞书（否则写操作需在页面上确认）
    # 新建文档的链接分享：应用新建的文档默认只有应用能看，打开链接分享你才能在飞书里看到它（closed = 不改）
    new_doc_share: Literal["tenant_editable", "tenant_readable", "closed"] = "tenant_editable"
    agent_max_steps: int = 6
    read_images: bool = True                          # 读图片：下载后 OCR + 多模态模型描述，写进正文（图片为主的文档靠它才能分析）
    auto_sync_minutes: int = Field(default=0, ge=0, le=1440)  # 定时同步间隔（分钟），0 = 关闭
    event_verification_token: str = ""                # 事件订阅的 Verification Token（开发者后台「事件与回调」→「加密策略」）
    event_encrypt_key: str = ""                       # 同上的 Encrypt Key；配置了才能解密加密推送

    @field_validator("web_base")
    @classmethod
    def _origin(cls, v: str) -> str:
        """只留 https://xxx.feishu.cn：贴了带 /wiki/…、/docx/… 的完整链接也行。"""
        v = (v or "").strip()
        if not v:
            return ""
        u = urlparse(v if "://" in v else f"https://{v}")
        return f"{u.scheme}://{u.netloc}" if u.netloc else v.rstrip("/")


def _safe_name(title: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", title).strip() or "未命名文档"
    return name[:120]


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() + "Z" if dt else None


class FeishuService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._client: tuple[tuple, FeishuClient] | None = None
        self.transport: httpx.BaseTransport | None = None     # 测试注入 httpx.MockTransport
        self._sync_lock = threading.Lock()                     # 定时同步 / 事件同步互斥：同一时间只跑一轮
        self._auto_stop = threading.Event()
        self._auto_thread: threading.Thread | None = None
        self._auto_last: dict[str, Any] | None = None
        self._seen_events: OrderedDict[str, None] = OrderedDict()   # 飞书 3 秒内没收到 200 会重推，按 event_id 去重

    # ------------------------------------------------------------------ 配置
    def settings(self) -> FeishuSettings:
        with SessionLocal() as db:
            stored = Repository(db).get_setting(SETTING_KEY, {}) or {}
        s = FeishuSettings.model_validate(stored)
        s.app_id = s.app_id or os.environ.get("FEISHU_APP_ID", "")
        s.app_secret = s.app_secret or os.environ.get("FEISHU_APP_SECRET", "")
        if "domain" not in stored and os.environ.get("FEISHU_DOMAIN") in ("feishu", "lark"):
            s.domain = os.environ["FEISHU_DOMAIN"]  # type: ignore[assignment]
        return s

    def public_settings(self) -> dict[str, Any]:
        s = self.settings()
        with SessionLocal() as db:
            stored = Repository(db).get_setting(SETTING_KEY, {}) or {}
        return {**s.model_dump(), **{k: MASKED if getattr(s, k) else "" for k in SECRET_FIELDS}, "configured": bool(s.app_id and s.app_secret),
                "fromEnv": bool(s.app_id) and not stored.get("app_id"), "autoSync": {"running": self.auto_sync_running(), "last": self._auto_last}}

    def update_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        with SessionLocal() as db:
            repo = Repository(db)
            stored = repo.get_setting(SETTING_KEY, {}) or {}
            clean = {k: v for k, v in patch.items() if k in FeishuSettings.model_fields and not (k in SECRET_FIELDS and v == MASKED)}
            merged = FeishuSettings.model_validate({**stored, **clean}).model_dump()
            repo.set_setting(SETTING_KEY, merged)
        with self._lock:
            self._client = None
        return self.public_settings()

    def client(self) -> FeishuClient:
        s = self.settings()
        if not (s.app_id and s.app_secret):
            raise FeishuError("还没有配置飞书应用：在「飞书文档」页填写 App ID 与 App Secret")
        key = (s.app_id, s.app_secret, s.domain, id(self.transport))
        with self._lock:
            if self._client is None or self._client[0] != key:
                # FEISHU_API_BASE：经代理 / 私有化部署访问开放平台时覆盖接口地址（默认按 domain 取 open.feishu.cn / open.larksuite.com）
                self._client = (key, FeishuClient(s.app_id, s.app_secret, domain=s.domain, base_url=os.environ.get("FEISHU_API_BASE") or None, transport=self.transport))
            return self._client[1]

    def test(self) -> dict[str, Any]:
        """换 token，再逐项探测权限：用不存在的资源调一次接口，报「未开通权限」就是缺，其它错误（不存在等）说明权限已开通。"""
        s = self.settings()
        out: dict[str, Any] = {"ok": False, "token": False, "folder": None, "scopes": {}}
        try:
            client = self.client()
            client.tenant_token(refresh=True)
            out["token"] = True
        except FeishuError as exc:
            out["error"] = str(exc)
            return out
        missing, errors = [], []
        for name, need, probe in SCOPE_PROBES:
            try:
                result = probe(client, s)
                if name == "drive":
                    out["folder"] = {"token": s.folder_token or None, "files": len(result)}
                ok = True
            except FeishuError as exc:
                ok = exc.code not in SCOPE_MISSING
                if ok and name == "drive":
                    errors.append(str(exc))           # 权限有，但默认文件夹不对等
            out["scopes"][name] = ok
            if not ok:
                missing.append(need)
        if missing:
            errors.insert(0, "缺少权限：" + "、".join(missing) + "——在开发者后台「权限管理」开通，再到「版本管理与发布」创建版本并发布")
        out["ok"] = not errors
        if errors:
            out["error"] = "；".join(errors)
        return out

    # ------------------------------------------------------------------ 解析引用
    def resolve(self, ref: str, kb_id: str | None = None) -> tuple[str, str]:
        """链接 / document_id / 已接入文档的标题 → (document_id, 链接)。"""
        ref = (ref or "").strip()
        try:
            parsed = parse_ref(ref)
        except ValueError:
            if "://" in ref:
                raise
            link = self._link_by_title(ref, kb_id)
            if link is None:
                raise ValueError(f"找不到文档：{ref}（请给出飞书链接、document_id，或已接入文档的标题）") from None
            return link.document_id, link.url or self.client().web_url(link.document_id, self.settings().web_base or None)
        client = self.client()
        target = client.resolve(parsed)
        if target.kind != "docx":
            label = {"doc": "旧版文档", "sheet": "电子表格", "bitable": "多维表格", "folder": "文件夹", "mindnote": "思维笔记"}.get(target.kind, target.kind)
            raise FeishuError(f"只支持新版文档（docx），这是{label}" + ("：请在飞书里先升级为新版文档" if target.kind == "doc" else ""))
        url = ref if "://" in ref else client.web_url(target.token, self.settings().web_base or None)
        return target.token, url

    def _link_by_title(self, title: str, kb_id: str | None) -> FeishuDoc | None:
        if not title:
            return None
        with SessionLocal() as db:
            links = Repository(db).list_feishu_docs(kb_id)
        exact = [lk for lk in links if (lk.title or "") == title]
        fuzzy = [lk for lk in links if title in (lk.title or "")]
        return (exact or fuzzy or [None])[0]

    # ------------------------------------------------------------------ 远程：查
    def read(self, ref: str, kb_id: str | None = None, *, view: bool = False) -> dict[str, Any]:
        """默认给智能体 / 分析用：Markdown 里带图片的 OCR 文字与描述。

        view=True 给页面展示：html 按飞书原样排版（图片走 /feishu/media 代理），markdown 是不含识别文字的原文（编辑全文时用），不跑 OCR。
        """
        document_id, url = self.resolve(ref, kb_id)
        client = self.client()
        meta = client.get_document(document_id)
        blocks = client.list_blocks(document_id)
        title = (meta.get("title") or "").strip()
        out: dict[str, Any] = {"documentId": document_id, "title": title or "未命名文档", "revisionId": meta.get("revision_id"), "url": url, "blocks": len(blocks)}
        if view:
            md = blocks_to_markdown(blocks, title=title)
            out["html"] = blocks_to_html(blocks, title=title, image_src=lambda token: f"{app_config.api_prefix}/feishu/media/{token}")
        else:
            md = blocks_to_markdown(blocks, title=title, image=self._image_renderer(client, blocks))
        return {**out, "markdown": md, "chars": len(md)}

    def media_file(self, token: str, client: FeishuClient | None = None) -> Path:
        """文档里的图片 token → 本地缓存文件（没有就从飞书下载）。"""
        from prism.multimodal.extract import open_image

        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", token or ""):
            raise ValueError("无效的图片 token")
        media = app_config.data_dir / MEDIA_DIR
        cached = next((p for p in media.glob(f"{token}.*") if p.suffix != ".json"), None) if media.is_dir() else None
        if cached is not None:
            return cached
        data = (client or self.client()).download_media(token)
        if open_image(data) is None:
            raise FeishuError("图片无法解码")
        media.mkdir(parents=True, exist_ok=True)
        path = media / f"{token}.{_image_ext(data)}"
        path.write_bytes(data)
        return path

    def list_remote(self, folder_token: str | None = None) -> dict[str, Any]:
        token = folder_token if folder_token is not None else (self.settings().folder_token or None)
        if token and "://" in token:
            ref = parse_ref(token)
            if ref.kind != "folder":
                raise ValueError("这不是文件夹链接")
            token = ref.token
        files = self.client().list_folder(token or None)
        with SessionLocal() as db:
            linked: dict[str, list[str]] = {}
            for lk in Repository(db).list_feishu_docs():
                linked.setdefault(lk.document_id, []).append(lk.kb_id)
        return {"folderToken": token, "files": [{"token": f.get("token"), "name": f.get("name"), "type": f.get("type"), "url": f.get("url"),
                                                   "modifiedTime": f.get("modified_time"), "linkedKbs": linked.get(f.get("token") or "", [])} for f in files]}

    # ------------------------------------------------------------------ 远程：增 / 改 / 删
    def create(self, title: str, markdown: str = "", *, folder_token: str | None = None, kb_id: str | None = None, schedule: Schedule | None = None) -> dict[str, Any]:
        title = (title or "").strip() or "未命名文档"
        client = self.client()
        doc = client.create_document(title, folder_token if folder_token is not None else (self.settings().folder_token or None))
        document_id = doc.get("document_id")
        if not document_id:
            raise FeishuError("飞书没有返回新文档的 document_id")
        body = _strip_title(markdown, title)
        written = self._insert_markdown(client, document_id, body) if body.strip() else 0
        url = client.web_url(document_id, self.settings().web_base or None)
        out: dict[str, Any] = {"documentId": document_id, "title": title, "url": url, "blocks": written}
        share = self.settings().new_doc_share
        if share != "closed":
            try:
                client.set_link_share(document_id, share)
                out["share"] = share
            except FeishuError as exc:
                logger.warning("设置新文档的链接分享失败 %s：%s", document_id, exc)
                out["shareError"] = ("新文档在应用自己的云空间里，目前只有应用能打开：开通 docs:permission.setting:write_only（或 drive:drive）后会自动对组织内开放链接，"
                                     "或在「应用配置」里设置默认文件夹（你分享给应用的文件夹）" if exc.code in SCOPE_MISSING else f"设置链接分享失败：{exc}")
        if kb_id:
            out["sync"] = self.sync_document(kb_id, document_id, url=url, force=True, schedule=schedule)
        return out

    def write(self, ref: str, markdown: str, *, mode: Literal["append", "replace"] = "append", kb_id: str | None = None, schedule: Schedule | None = None) -> dict[str, Any]:
        if not (markdown or "").strip():
            raise ValueError("要写入的内容为空")
        document_id, url = self.resolve(ref, kb_id)
        client = self.client()
        removed = 0
        if mode == "replace":
            meta = client.get_document(document_id)
            markdown = _strip_title(markdown, meta.get("title") or "")
            root = next((b for b in client.list_blocks(document_id) if b.get("block_id") == document_id), None)
            removed = len((root or {}).get("children") or [])
            client.delete_children(document_id, 0, removed)
        written = self._insert_markdown(client, document_id, markdown)
        return {"documentId": document_id, "url": url, "mode": mode, "removedBlocks": removed, "blocks": written, "resynced": self._after_write(document_id, schedule)}

    def replace_text(self, ref: str, find: str, replace: str, *, kb_id: str | None = None, schedule: Schedule | None = None) -> dict[str, Any]:
        if not find:
            raise ValueError("要查找的文字为空")
        document_id, url = self.resolve(ref, kb_id)
        client = self.client()
        total, changed, missed = 0, 0, 0
        for b in client.list_blocks(document_id):
            field = text_field(b)
            if not field:
                continue
            elements = (b.get(field) or {}).get("elements") or []
            new, n = replace_in_elements(elements, find, replace)
            if n:
                client.update_text(document_id, b["block_id"], new)
                total += n
                changed += 1
            elif find in block_text(b):
                missed += 1                      # 匹配跨了加粗 / 链接等样式边界，按段替换会丢样式，不处理
        out = {"documentId": document_id, "url": url, "replaced": total, "blocks": changed, "skippedStyledMatches": missed}
        if total:
            out["resynced"] = self._after_write(document_id, schedule)
        return out

    def delete_remote(self, ref: str, *, kb_id: str | None = None, delete_local: bool = True) -> dict[str, Any]:
        document_id, url = self.resolve(ref, kb_id)
        self.client().delete_file(document_id, "docx")
        removed = []
        with SessionLocal() as db:
            links = [lk.id for lk in Repository(db).feishu_docs_by_document(document_id)]
        for link_id in links:
            if delete_local:
                self.unlink(link_id, delete_local=True)
                removed.append(link_id)
        return {"documentId": document_id, "url": url, "deleted": True, "unlinked": removed}

    def _insert_markdown(self, client: FeishuClient, document_id: str, markdown: str) -> int:
        if self.settings().use_convert_api and has_table(markdown):
            try:
                first, blocks = client.convert_markdown(markdown)
                if first and blocks:
                    return client.insert_descendants(document_id, first, blocks)
            except FeishuError as exc:
                logger.info("飞书 Markdown 转换接口不可用，改用内置转换（表格按行写成文本）：%s", exc)
        return client.append_children(document_id, markdown_to_blocks(markdown))

    def _after_write(self, document_id: str, schedule: Schedule | None) -> list[str]:
        """飞书上改过的文档，如果已经接入某个知识库，就重新同步（让检索看到最新内容）。返回重新同步的知识库。"""
        with SessionLocal() as db:
            links = [(lk.kb_id, lk.url) for lk in Repository(db).feishu_docs_by_document(document_id)]
        done = []
        for kb_id, url in links:
            try:
                self.sync_document(kb_id, document_id, url=url, force=True, schedule=schedule)
                done.append(kb_id)
            except Exception as exc:
                logger.warning("写入后重新同步失败 %s → %s：%s", document_id, kb_id, exc)
        return done

    # ------------------------------------------------------------------ 同步进知识库
    def links(self, kb_id: str | None = None) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            repo = Repository(db)
            out = []
            for lk in repo.list_feishu_docs(kb_id):
                local = repo.get_document(lk.local_doc_id) if lk.local_doc_id else None
                out.append({"id": lk.id, "kbId": lk.kb_id, "documentId": lk.document_id, "title": lk.title, "url": lk.url, "revisionId": lk.revision_id,
                            "status": lk.status if (local is not None or lk.status == "FAILED") else "MISSING", "errorMessage": lk.error_message,
                            "localDocId": lk.local_doc_id if local is not None else None, "localStatus": local.status if local else None,
                            "chunkCount": (local.chunk_count or 0) if local else 0, "syncedAt": _iso(lk.synced_at), "createdAt": _iso(lk.created_at)})
            return out

    def import_refs(self, kb_id: str, refs: list[str], *, schedule: Schedule | None = None) -> list[dict[str, Any]]:
        results = []
        for ref in dict.fromkeys(r.strip() for r in refs if r and r.strip()):
            try:
                document_id, url = self.resolve(ref, kb_id)
            except (ValueError, FeishuError) as exc:
                results.append({"ref": ref, "ok": False, "error": str(exc)})
                continue
            try:
                results.append({"ref": ref, "ok": True, **self.sync_document(kb_id, document_id, url=url, schedule=schedule)})
            except (ValueError, FeishuError) as exc:
                results.append({"ref": ref, "ok": False, "documentId": document_id, "error": str(exc)})
        return results

    def sync_link(self, link_id: str, *, force: bool = False, schedule: Schedule | None = None) -> dict[str, Any]:
        with SessionLocal() as db:
            lk = Repository(db).get_feishu_doc(link_id)
            if lk is None:
                raise KeyError(link_id)
            kb_id, document_id, url = lk.kb_id, lk.document_id, lk.url
        return self.sync_document(kb_id, document_id, url=url, force=force, schedule=schedule)

    def sync_kb(self, kb_id: str, *, force: bool = False, schedule: Schedule | None = None) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            ids = [lk.id for lk in Repository(db).list_feishu_docs(kb_id)]
        out = []
        for link_id in ids:
            try:
                out.append({"ok": True, **self.sync_link(link_id, force=force, schedule=schedule)})
            except (FeishuError, ValueError) as exc:
                out.append({"ok": False, "linkId": link_id, "error": str(exc)})
        return out

    def sync_document(self, kb_id: str, document_id: str, *, url: str | None = None, force: bool = False, schedule: Schedule | None = None) -> dict[str, Any]:
        """把一篇飞书文档同步进知识库。status：created / updated / unchanged。schedule 为空时就地解析入库（阻塞到完成）。"""
        with SessionLocal() as db:
            repo = Repository(db)
            if repo.get_kb(kb_id) is None:
                raise ValueError("知识库不存在")
            link = repo.find_feishu_doc(kb_id, document_id)
            local = repo.get_document(link.local_doc_id) if link and link.local_doc_id else None
            prev_revision = link.revision_id if link else None
            new_link = link is None
            local_ok = local is not None and local.status == "COMPLETED"
        client = self.client()
        try:
            meta = client.get_document(document_id)
            revision = meta.get("revision_id")
            title = (meta.get("title") or "").strip() or "未命名文档"
            if local_ok and not force and revision is not None and prev_revision == revision:
                return {"status": "unchanged", "linkId": link.id, "documentId": document_id, "title": title, "localDocId": local.id, "revisionId": revision}
            blocks = client.list_blocks(document_id)
            folder = app_config.upload_dir / kb_id
            assets = folder / ASSETS_DIR / document_id
            markdown = blocks_to_markdown(blocks, title=title, image=self._image_renderer(client, blocks, assets=(assets, f"{ASSETS_DIR}/{document_id}")))
        except FeishuError as exc:
            self._mark_failed(kb_id, document_id, url, str(exc))
            raise
        url = url or (link.url if link else None) or client.web_url(document_id, self.settings().web_base or None)
        content = _front_matter(title, document_id, url) + markdown

        with SessionLocal() as db:
            repo = Repository(db)
            link = repo.find_feishu_doc(kb_id, document_id)
            local = repo.get_document(link.local_doc_id) if link and link.local_doc_id else None
            folder.mkdir(parents=True, exist_ok=True)
            if local is None:
                local_id = str(uuid.uuid4())
                path = folder / f"{local_id}.md"
                path.write_text(content, encoding="utf-8")
                repo.create_document(id=local_id, kb_id=kb_id, file_name=f"{_safe_name(title)}.md", file_path=str(path), file_size=path.stat().st_size,
                                     file_format="md", status="PENDING", title=title, tags="飞书")
                status = "created"
            else:
                local_id = local.id
                path = Path(local.file_path) if local.file_path and local.file_path.endswith(".md") else folder / f"{local_id}.md"
                path.write_text(content, encoding="utf-8")
                repo.update_document(local, file_name=f"{_safe_name(title)}.md", file_path=str(path), file_size=path.stat().st_size, file_format="md", title=title, status="PENDING")
                status = "updated"
            link = repo.save_feishu_doc(link, kb_id=kb_id, document_id=document_id, title=title, url=url, revision_id=revision, local_doc_id=local_id,
                                        status="SYNCED", error_message=None, synced_at=datetime.utcnow())
            link_id = link.id
        if new_link and self.settings().event_verification_token:
            self._subscribe(client, document_id)
        if schedule is not None:
            schedule([local_id])
        else:
            self.process_many([local_id])
        return {"status": status, "linkId": link_id, "documentId": document_id, "title": title, "localDocId": local_id, "revisionId": revision, "chars": len(markdown)}

    def _mark_failed(self, kb_id: str, document_id: str, url: str | None, error: str) -> None:
        with SessionLocal() as db:
            repo = Repository(db)
            link = repo.find_feishu_doc(kb_id, document_id)
            repo.save_feishu_doc(link, kb_id=kb_id, document_id=document_id, url=url or (link.url if link else None), status="FAILED", error_message=error[:900],
                                 **({} if link else {"title": document_id}))

    @staticmethod
    def process_many(local_ids: list[str]) -> None:
        from .document_service import document_service

        for doc_id in local_ids:
            document_service.process(doc_id, reason="feishu_sync")

    def unlink(self, link_id: str, *, delete_local: bool = True) -> dict[str, Any]:
        with SessionLocal() as db:
            repo = Repository(db)
            lk = repo.get_feishu_doc(link_id)
            if lk is None:
                raise KeyError(link_id)
            local_id, kb_id, document_id = lk.local_doc_id, lk.kb_id, lk.document_id
            repo.delete_feishu_doc(lk)
        removed = False
        if delete_local and local_id:
            from .document_service import document_service

            try:
                document_service.delete(local_id)
                removed = True
            except KeyError:
                pass
            shutil.rmtree(app_config.upload_dir / kb_id / ASSETS_DIR / document_id, ignore_errors=True)
        return {"linkId": link_id, "localDeleted": removed}

    # ------------------------------------------------------------------ 图片：下载 + OCR / 看图描述
    def _image_renderer(self, client: FeishuClient, blocks: list[dict[str, Any]], *, assets: tuple[Path, str] | None = None) -> ImageRenderer | None:
        """准备好文档里每张图的文字（并发下载 / 识别），返回给 blocks_to_markdown 的图片渲染回调。

        assets = (目录, .md 里引用它的相对路径)：同步时把图片文件放过去，Markdown 里用 ![](相对路径) 引用，图片索引会收录。
        """
        if not self.settings().read_images:
            return None
        tokens = list(dict.fromkeys(t for b in blocks if b.get("block_type") == IMAGE and (t := (b.get("image") or {}).get("token"))))
        if not tokens:
            return None
        describer = self._describer()
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="feishu-image") as pool:
            infos = dict(zip(tokens, pool.map(lambda t: self._image_info(client, t, describer), tokens)))
        if assets is not None:
            shutil.rmtree(assets[0], ignore_errors=True)
            assets[0].mkdir(parents=True, exist_ok=True)

        def render(block: dict[str, Any], caption: str) -> str | None:
            info = infos.get((block.get("image") or {}).get("token"))
            if info is None:
                return None
            label = re.sub(r"[\[\]\n]+", " ", caption).strip() or "图片"
            if info.get("error"):
                return f"[{label}（没能读取：{info['error'][:120]}）]"
            out = [f"[{label}]"]
            if assets is not None:
                src = Path(info["path"])
                shutil.copyfile(src, assets[0] / src.name)
                out = [f"![{label}]({assets[1]}/{src.name})"]
            if info.get("description"):
                out.append("> 图片内容：" + "\n> ".join(ln.strip() for ln in info["description"].splitlines() if ln.strip()))
            if info.get("ocr"):
                out.append("> 图中文字：" + info["ocr"][:2000])
            return "\n".join(out)

        return render

    @staticmethod
    def _describer() -> Any:
        from prism.multimodal.describe import ImageDescriber

        from .model_service import model_service
        from .settings_service import settings_service

        try:
            vlm = model_service.vision_llm(aux=True)
        except Exception as exc:
            logger.info("没有可用的多模态模型，飞书图片只做 OCR：%s", exc)
            vlm = None
        return ImageDescriber(settings_service.get().multimodal, vlm)       # 不给 cache_path：结果由 _image_info 按 token 缓存

    def _image_info(self, client: FeishuClient, token: str, describer: Any) -> dict[str, Any]:
        """{path, ocr, description} 或 {error}。按图片 token 缓存；换了 OCR 引擎 / 多模态模型才重新识别。"""
        from prism.multimodal.extract import open_image

        meta_path = app_config.data_dir / MEDIA_DIR / f"{token}.json"
        caps = describer.capabilities
        engine = f"{caps['ocr']}|{caps['vlm']}"
        try:
            cached = json.loads(meta_path.read_text(encoding="utf-8"))
            if cached.get("engine") == engine and Path(cached["path"]).is_file():
                return cached
        except (OSError, ValueError, KeyError):
            pass
        try:
            path = self.media_file(token, client)
        except FeishuError as exc:
            logger.warning("下载飞书图片失败 %s：%s", token, exc)
            return {"error": "应用缺少下载图片的权限（docs:document.media:download 或 drive:drive）" if exc.code in SCOPE_MISSING else str(exc)}
        data = path.read_bytes()
        image = open_image(data)
        if image is None:
            return {"error": "图片无法解码"}
        ocr, description = "", ""
        if min(image.size) >= describer.cfg.min_image_side:          # 图标 / 分隔线不识别
            ocr, description = describer.describe(image, hashlib.sha1(data).hexdigest())
        info = {"path": str(path), "ocr": ocr, "description": description, "engine": engine}
        meta_path.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        return info

    # ------------------------------------------------------------------ 自动同步：定时
    auto_sync_tick = 30.0                                     # 每隔多久看一次配置、是否到点（秒）；改间隔不用重启

    def start_auto_sync(self) -> None:
        """应用启动时调用：起后台线程，按 auto_sync_minutes 定时同步（0 = 关闭，线程照跑、只看配置）。"""
        with self._lock:
            if self.auto_sync_running():
                return
            self._auto_stop = threading.Event()
            self._auto_thread = threading.Thread(target=self._auto_sync_loop, args=(self._auto_stop,), name="feishu-auto-sync", daemon=True)
            self._auto_thread.start()

    def stop_auto_sync(self) -> None:
        self._auto_stop.set()

    def auto_sync_running(self) -> bool:
        return self._auto_thread is not None and self._auto_thread.is_alive() and not self._auto_stop.is_set()

    def _auto_sync_loop(self, stop: threading.Event) -> None:
        last: float | None = None              # 上一轮开始的时间；None = 下次看配置时就跑（启动后 / 刚打开时先补一轮）
        while not stop.wait(self.auto_sync_tick):
            try:
                minutes = self.settings().auto_sync_minutes
                if minutes <= 0:
                    last = None
                    continue
                if last is not None and time.monotonic() - last < minutes * 60:
                    continue
                last = time.monotonic()
                self.auto_sync_round()
            except Exception:
                logger.exception("飞书自动同步出错")

    def auto_sync_round(self) -> dict[str, Any] | None:
        """同步一轮：所有接入了飞书文档的知识库按 revision 增量同步。已有一轮在跑 / 没配置应用时跳过，返回 None。"""
        if not self._sync_lock.acquire(blocking=False):
            logger.info("飞书自动同步：上一轮还没结束，跳过")
            return None
        try:
            s = self.settings()
            if not (s.app_id and s.app_secret):
                return None
            with SessionLocal() as db:
                kb_ids = sorted({lk.kb_id for lk in Repository(db).list_feishu_docs()})
            started = datetime.utcnow()
            out: dict[str, Any] = {"kbs": len(kb_ids), "checked": 0, "updated": 0, "failed": 0, "errors": []}
            for kb_id in kb_ids:
                try:
                    results = self.sync_kb(kb_id)
                except Exception as exc:
                    logger.exception("飞书自动同步失败：知识库 %s", kb_id)
                    out["errors"].append(f"{kb_id}: {type(exc).__name__}: {exc}"[:300])
                    continue
                out["checked"] += len(results)
                out["updated"] += sum(1 for r in results if r["ok"] and r.get("status") != "unchanged")
                out["failed"] += sum(1 for r in results if not r["ok"])
                out["errors"] += [f"{r.get('linkId')}: {r.get('error')}"[:300] for r in results if not r["ok"]]
            out["errors"] = out["errors"][:10]
            self._auto_last = {"at": _iso(started), **out}
            if out["updated"] or out["failed"] or out["errors"]:
                logger.info("飞书自动同步：检查 %d 篇，更新 %d 篇，失败 %d 篇", out["checked"], out["updated"], out["failed"] + len(out["errors"]))
            return self._auto_last
        finally:
            self._sync_lock.release()

    # ------------------------------------------------------------------ 自动同步：事件订阅
    def handle_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        """飞书事件回调。返回 {"challenge"}（URL 校验）、{"documentId"}（要同步的文档，调用方放后台跑 sync_changed）或 {"ignored"}。

        Verification Token 没配置 / 不匹配抛 PermissionError；加密推送需要配置 Encrypt Key，解不开抛 ValueError。
        """
        s = self.settings()
        if "encrypt" in payload:
            if not s.event_encrypt_key:
                raise ValueError("收到加密的事件推送，但没有配置 Encrypt Key")
            payload = _decrypt_event(str(payload["encrypt"]), s.event_encrypt_key)
        header = payload.get("header") or {}
        if not s.event_verification_token:
            raise PermissionError("还没有配置飞书事件订阅的 Verification Token")
        if not hmac.compare_digest(str(header.get("token") or payload.get("token") or ""), s.event_verification_token):
            raise PermissionError("Verification Token 不匹配")
        if payload.get("type") == "url_verification":
            return {"challenge": payload.get("challenge", "")}

        event = payload.get("event") or {}
        event_type = header.get("event_type") or event.get("type")
        if event_type != "drive.file.edit_v1":
            return {"ignored": f"不处理的事件：{event_type}"}
        event_id = header.get("event_id") or payload.get("uuid")
        if event_id and not self._first_seen(str(event_id)):
            return {"ignored": "重复推送"}
        document_id = str(event.get("file_token") or "")
        if not document_id or event.get("file_type", "docx") != "docx":
            return {"ignored": "不是新版文档"}
        with SessionLocal() as db:
            if not Repository(db).feishu_docs_by_document(document_id):
                return {"ignored": "没有接入知识库的文档"}
        return {"documentId": document_id}

    def _first_seen(self, event_id: str) -> bool:
        with self._lock:
            if event_id in self._seen_events:
                return False
            self._seen_events[event_id] = None
            while len(self._seen_events) > 1000:
                self._seen_events.popitem(last=False)
            return True

    def sync_changed(self, document_id: str) -> list[dict[str, Any]]:
        """事件触发：只把这一篇同步进接入了它的知识库（按 revision 增量）。等正在跑的一轮结束后再跑，异常只记日志。"""
        with self._sync_lock:
            with SessionLocal() as db:
                links = [(lk.kb_id, lk.url) for lk in Repository(db).feishu_docs_by_document(document_id)]
            out = []
            for kb_id, url in links:
                try:
                    out.append({"ok": True, "kbId": kb_id, **self.sync_document(kb_id, document_id, url=url)})
                except Exception as exc:
                    logger.warning("飞书事件同步失败 %s → %s：%s", document_id, kb_id, exc)
                    out.append({"ok": False, "kbId": kb_id, "documentId": document_id, "error": str(exc)})
            return out

    def subscribe_all(self) -> dict[str, Any]:
        """给所有已接入的文档订阅变更事件（飞书只推送订阅过的文档；新接入的文档在配置了 Token 后会自动订阅）。"""
        with SessionLocal() as db:
            ids = sorted({lk.document_id for lk in Repository(db).list_feishu_docs()})
        client = self.client()
        failed = {did: err for did in ids if (err := self._subscribe(client, did))}
        return {"total": len(ids), "subscribed": len(ids) - len(failed), "failed": failed}

    @staticmethod
    def _subscribe(client: FeishuClient, document_id: str) -> str | None:
        try:
            client.subscribe_file(document_id, "docx")
            return None
        except FeishuError as exc:
            logger.warning("订阅飞书文档变更失败 %s：%s", document_id, exc)
            return str(exc)

    # ------------------------------------------------------------------ 知识库里的上传文档（智能体读取 / 写进飞书）
    def kb_docs(self, kb_id: str) -> list[dict[str, Any]]:
        """知识库里不是从飞书同步来的文档（上传的 PDF / Word / Markdown 等）。"""
        with SessionLocal() as db:
            repo = Repository(db)
            synced = {lk.local_doc_id for lk in repo.list_feishu_docs(kb_id)}
            return [{"docId": d.id, "title": d.title or Path(d.file_name or "").stem or d.id, "fileName": d.file_name, "status": d.status}
                    for d in repo.list_documents(kb_id) if d.id not in synced]

    def kb_doc(self, kb_id: str | None, ref: str) -> dict[str, Any]:
        """知识库文档的全文（解析后的 Markdown，去掉 front matter）。ref = 文档 id / 标题 / 文件名，只在当前知识库里找。"""
        if not kb_id:
            raise ValueError("没有选择知识库")
        ref = (ref or "").strip()
        with SessionLocal() as db:
            docs = Repository(db).list_documents(kb_id)
            doc = next((d for d in docs if d.id == ref), None) or next((d for d in docs if ref in (d.title, d.file_name, Path(d.file_name or "").stem)), None) \
                or next((d for d in docs if ref and ref in (d.title or "")), None)
            if doc is None:
                raise ValueError(f"当前知识库里没有这篇文档：{ref}")
            if not (doc.markdown_content or "").strip():
                raise ValueError(f"《{doc.title or doc.file_name}》还没有解析出正文（状态：{doc.status}）")
            markdown = re.sub(r"\A---\n.*?\n---\n+", "", doc.markdown_content, flags=re.S)
            title = doc.title or Path(doc.file_name or "").stem
            return {"docId": doc.id, "title": title, "fileName": doc.file_name, "markdown": markdown, "chars": len(markdown)}

    # ------------------------------------------------------------------ 智能体
    def tools(self, kb_id: str | None) -> dict[str, Tool]:
        def need(args: dict[str, Any], key: str) -> str:
            v = str(args.get(key) or "").strip()
            if not v:
                raise ValueError(f"缺少参数 {key}")
            return v

        def list_docs(args: dict[str, Any]) -> dict[str, Any]:
            docs = [{"title": d["title"], "documentId": d["documentId"], "url": d["url"], "status": d["status"], "chunks": d["chunkCount"], "syncedAt": d["syncedAt"]}
                    for d in self.links(kb_id)]
            q = str(args.get("query") or "").strip()
            if q:
                docs = [d for d in docs if q in (d["title"] or "")] or docs
            out: dict[str, Any] = {"docs": docs}
            if args.get("folder"):
                out["folder"] = [f for f in self.list_remote(args.get("folder_token") or None)["files"] if f["type"] == "docx"]
            return out

        def search(args: dict[str, Any]) -> list[dict[str, Any]]:
            if not kb_id:
                raise ValueError("没有选择知识库，无法检索")
            from .retrieval_service import retrieval_service

            res = retrieval_service.retrieve(kb_id, need(args, "query"), top_k=int(args.get("top_k") or 6), use_transform=False)
            by_local = {d["localDocId"]: d for d in self.links(kb_id) if d["localDocId"]}
            names = retrieval_service.doc_names(kb_id)
            hits = []
            for c in res.contexts:
                link = by_local.get(c.chunk.doc_id)
                hits.append({"title": (link or {}).get("title") or names.get(c.chunk.doc_id), "documentId": (link or {}).get("documentId"), "url": (link or {}).get("url"),
                             "label": c.citation_label(), "text": c.text_for_llm().strip()[:900], "score": round(float(c.score), 4)})
            return hits

        def read_doc(args: dict[str, Any]) -> dict[str, Any]:
            doc = self.read(need(args, "document_id"), kb_id)
            limit = int(args.get("max_chars") or 12000)
            return {**doc, "markdown": doc["markdown"][:limit], "truncated": doc["chars"] > limit}

        def read_kb_doc(args: dict[str, Any]) -> dict[str, Any]:
            doc = self.kb_doc(kb_id, need(args, "doc_id"))
            limit = int(args.get("max_chars") or 12000)
            return {**doc, "markdown": doc["markdown"][:limit], "truncated": doc["chars"] > limit}

        def body(args: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
            """写入的正文：给了 source_doc_id 就原样取知识库文档的正文（不经模型转抄），否则用 markdown 参数。"""
            source = str(args.get("source_doc_id") or "").strip()
            if source:
                doc = self.kb_doc(kb_id, source)
                return doc["markdown"], doc
            return need(args, "markdown"), None

        def create_doc(args: dict[str, Any]) -> dict[str, Any]:
            markdown, source = body(args) if (args.get("source_doc_id") or args.get("markdown")) else ("", None)
            title = str(args.get("title") or "").strip() or (source["title"] if source else "")
            # 从知识库文档导出的默认不再同步回来：知识库里已经有这篇了
            sync = args.get("sync", source is None)
            return self.create(title or need(args, "title"), markdown, folder_token=args.get("folder_token") or None, kb_id=kb_id if sync else None)

        def append_doc(args: dict[str, Any]) -> dict[str, Any]:
            return self.write(need(args, "document_id"), body(args)[0], mode="append", kb_id=kb_id)

        def replace_doc(args: dict[str, Any]) -> dict[str, Any]:
            return self.write(need(args, "document_id"), body(args)[0], mode="replace", kb_id=kb_id)

        def replace_text(args: dict[str, Any]) -> dict[str, Any]:
            return self.replace_text(need(args, "document_id"), need(args, "find"), str(args.get("replace") or ""), kb_id=kb_id)

        def delete_doc(args: dict[str, Any]) -> dict[str, Any]:
            return self.delete_remote(need(args, "document_id"), kb_id=kb_id)

        def sync_doc(args: dict[str, Any]) -> Any:
            if not kb_id:
                raise ValueError("没有选择知识库，无法同步")
            ref = str(args.get("document_id") or "").strip()
            if not ref:
                return self.sync_kb(kb_id, force=bool(args.get("force")))
            document_id, url = self.resolve(ref, kb_id)
            return self.sync_document(kb_id, document_id, url=url, force=bool(args.get("force")))

        return {
            "list_docs": Tool("列出已接入当前知识库的飞书文档（标题 / document_id / 链接 / 同步状态）；folder=true 时再列出飞书文件夹里的文档。参数：{\"query\": \"标题关键词，可省略\", \"folder\": false}", list_docs),
            "search": Tool("在当前知识库（含已同步的飞书文档）里做 RAG 检索，返回相关片段及其所属飞书文档。参数：{\"query\": \"...\", \"top_k\": 6}", search),
            "read_doc": Tool("读取一篇飞书文档的最新全文（Markdown）。参数：{\"document_id\": \"链接 / document_id / 已接入文档的标题\", \"max_chars\": 12000}", read_doc),
            "read_kb_doc": Tool("读取当前知识库里一篇上传文档（PDF / Word / Markdown 等，不是飞书文档）的全文。参数：{\"doc_id\": \"知识库文档 id 或标题\", \"max_chars\": 12000}", read_kb_doc),
            "create_doc": Tool("新建飞书文档并写入内容，默认同步进当前知识库。参数：{\"title\": \"...\", \"markdown\": \"正文 Markdown\"}；"
                               "要把知识库文档的正文原样写进去时改给 {\"source_doc_id\": \"知识库文档 id\", \"title\": \"可省略，默认用原标题\"}，不要自己抄写正文", create_doc, write=True),
            "append_doc": Tool("在文档末尾追加内容。参数：{\"document_id\": \"...\", \"markdown\": \"...\"}；追加知识库文档的正文时用 source_doc_id 代替 markdown", append_doc, write=True),
            "replace_doc": Tool("用新内容整篇替换文档正文（标题不变；原正文会被删除，图片等非文字块也会被删）。参数：{\"document_id\": \"...\", \"markdown\": \"...\"}；"
                                "用知识库文档的正文替换时用 source_doc_id 代替 markdown", replace_doc, write=True),
            "replace_text": Tool("在文档里把一段文字替换成另一段（保留样式，适合改错字 / 改数字 / 改名称）。参数：{\"document_id\": \"...\", \"find\": \"原文\", \"replace\": \"新文字\"}", replace_text, write=True),
            "delete_doc": Tool("删除飞书文档（进飞书回收站），同时从知识库移除。参数：{\"document_id\": \"...\"}", delete_doc, write=True),
            "sync_doc": Tool("把飞书文档的最新内容重新同步进知识库；不给 document_id 时同步全部已接入文档。参数：{\"document_id\": \"可省略\", \"force\": false}", sync_doc),
        }

    def agent(self, kb_id: str | None, *, allow_write: bool | None = None) -> FeishuAgent:
        from .model_service import model_service

        s = self.settings()
        # 当前知识库的文档交给智能体（飞书文档 + 上传的文档）：用户说「这篇文章」时它知道指哪篇
        docs = [{"title": d["title"], "documentId": d["documentId"], "url": d["url"]} for d in self.links(kb_id)] if kb_id else []
        docs += [{"title": d["title"], "docId": d["docId"], "kind": "kb"} for d in self.kb_docs(kb_id)] if kb_id else []
        return FeishuAgent(self.tools(kb_id), llm=model_service.llm(), max_steps=s.agent_max_steps,
                           allow_write=s.agent_allow_write if allow_write is None else allow_write, docs=docs)

    def stream(self, kb_id: str | None, question: str, *, allow_write: bool | None = None) -> Iterator[str]:
        kb_id = kb_id or self.settings().default_kb_id or None
        try:
            for ev in self.agent(kb_id, allow_write=allow_write).run(question):
                yield _sse(ev["type"], ev)
        except Exception as exc:
            logger.exception("飞书智能体运行失败")
            yield _sse("error", {"type": "error", "agent": "feishu", "content": f"{type(exc).__name__}: {exc}", "data": {}, "ms": 0})

    def execute_action(self, kb_id: str | None, tool: str, args: dict[str, Any]) -> Any:
        """执行智能体给出的待确认写操作。"""
        t = self.tools(kb_id or self.settings().default_kb_id or None).get(tool)
        if t is None or not t.write:
            raise ValueError(f"不能执行的操作：{tool}")
        return t.fn(args or {})


def _image_ext(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "png"
    if data.startswith(b"GIF8"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "jpg"


def _decrypt_event(encrypted: str, key: str) -> dict[str, Any]:
    """飞书加密推送：AES-256-CBC，密钥 = SHA256(Encrypt Key)，密文前 16 字节是 IV，PKCS7 填充。"""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    try:
        raw = base64.b64decode(encrypted)
        decryptor = Cipher(algorithms.AES(hashlib.sha256(key.encode()).digest()), modes.CBC(raw[:16])).decryptor()
        plain = decryptor.update(raw[16:]) + decryptor.finalize()
        return json.loads(plain[:-plain[-1]].decode("utf-8"))
    except Exception as exc:
        raise ValueError("事件推送解密失败：请检查 Encrypt Key") from exc


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _front_matter(title: str, document_id: str, url: str) -> str:
    meta = {"title": title, "tags": ["飞书"], "source": "feishu", "feishu_document_id": document_id, "feishu_url": url}
    return "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) + "---\n\n"


def _strip_title(markdown: str, title: str) -> str:
    """正文开头的「# 标题」与文档标题相同时去掉（飞书的标题单独存放，不去掉会显示两遍）。"""
    text = (markdown or "").lstrip()
    m = re.match(r"#\s+(.+?)\s*\n", text + "\n")
    if m and title and m.group(1).strip() == title.strip():
        return text[m.end():].lstrip("\n")
    return markdown or ""


feishu_service = FeishuService()
