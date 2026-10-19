"""飞书开放平台客户端（新版文档 docx + 云空间 drive + 知识库 wiki），自建应用 tenant_access_token 鉴权。

只用到的接口：
    POST   /auth/v3/tenant_access_token/internal                       换取 tenant_access_token（2 小时有效，提前 5 分钟刷新）
    GET    /docx/v1/documents/{id}                                     文档标题 / revision
    GET    /docx/v1/documents/{id}/blocks                              全部块（分页）
    POST   /docx/v1/documents                                          新建文档（可指定文件夹）
    POST   /docx/v1/documents/{id}/blocks/{block}/children             追加子块（每次最多 50 个）
    POST   /docx/v1/documents/blocks/convert                           Markdown → 块（官方转换，表格等结构更完整）
    POST   /docx/v1/documents/{id}/blocks/{block}/descendant           插入带嵌套的块（配合 convert）
    DELETE /docx/v1/documents/{id}/blocks/{block}/children/batch_delete 删除一段子块
    PATCH  /docx/v1/documents/{id}/blocks/{block}                      改一个块的文字
    DELETE /drive/v1/files/{token}?type=docx                           删除文档（进回收站）
    GET    /drive/v1/files                                             列文件夹
    GET    /wiki/v2/spaces/get_node                                    知识库节点 → 实际文档 token
    POST   /drive/v1/files/{token}/subscribe?file_type=docx            订阅文档变更（之后才会推送 drive.file.edit_v1 事件）
    GET    /drive/v1/medias/{token}/download                           下载文档里的图片 / 附件（二进制）
    PATCH  /drive/v1/permissions/{token}/public?type=docx              文档的链接分享权限（应用新建的文档默认只有应用自己能看）

应用需要开通权限 docx:document、drive:drive（或 drive:drive:readonly）、wiki:wiki:readonly；读图片还要 docs:document.media:download（有 drive:drive 也行），
并且文档要对应用可见：在文档「…→ 添加文档应用」里加上这个应用，或把文档放进应用有权限的文件夹。
http 传输可注入（测试用 httpx.MockTransport），不碰真实网络。
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

DOMAINS = {"feishu": "https://open.feishu.cn", "lark": "https://open.larksuite.com"}
WEB_BASES = {"feishu": "https://feishu.cn", "lark": "https://larksuite.com"}

_TOKEN_EXPIRED = {99991661, 99991663, 99991664, 99991668}   # token 无效 / 过期：刷新后重试一次
_RATE_LIMITED = {99991400}                                   # 频率限制：退避重试
_MAX_CHILDREN = 50                                           # children 接口单次最多 50 个块
_MAX_DESCENDANTS = 1000                                      # descendant 接口单次最多 1000 个块


class FeishuError(RuntimeError):
    def __init__(self, message: str, *, code: int | None = None, status: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass
class DocRef:
    """一条飞书链接解析出的引用：kind = docx / wiki / folder / doc（旧版文档）/ sheet / bitable ..."""

    kind: str
    token: str


_URL_KINDS = {"docx": "docx", "wiki": "wiki", "docs": "doc", "doc": "doc", "sheets": "sheet", "base": "bitable", "mindnotes": "mindnote", "folder": "folder", "file": "file"}


def parse_ref(text: str) -> DocRef:
    """飞书 / Lark 链接或裸 token → DocRef。裸 token 视为 docx 文档。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("链接为空")
    if re.fullmatch(r"[A-Za-z0-9]{10,64}", text):
        return DocRef("docx", text)
    parsed = urlparse(text if "://" in text else f"https://{text}")
    parts = [p for p in parsed.path.split("/") if p]
    for i, part in enumerate(parts[:-1]):
        kind = _URL_KINDS.get(part)
        if kind and re.fullmatch(r"[A-Za-z0-9]{10,64}", parts[i + 1]):
            return DocRef(kind, parts[i + 1])
    raise ValueError(f"无法识别的飞书链接：{text}")


class FeishuClient:
    def __init__(self, app_id: str, app_secret: str, *, domain: str = "feishu", base_url: str | None = None, timeout: float = 30.0,
                 transport: httpx.BaseTransport | None = None, max_retries: int = 3) -> None:
        if not app_id or not app_secret:
            raise FeishuError("未配置飞书应用的 App ID / App Secret")
        self.app_id = app_id
        self.app_secret = app_secret
        self.domain = domain if domain in DOMAINS else "feishu"
        self.base_url = (base_url or DOMAINS[self.domain]).rstrip("/")
        self.max_retries = max_retries
        self._http = httpx.Client(base_url=f"{self.base_url}/open-apis", timeout=timeout, transport=transport)
        self._token: tuple[str, float] | None = None
        self._token_lock = threading.Lock()

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------ 鉴权 / 请求
    def tenant_token(self, *, refresh: bool = False) -> str:
        with self._token_lock:
            if not refresh and self._token and self._token[1] > time.time():
                return self._token[0]
            resp = self._http.post("/auth/v3/tenant_access_token/internal", json={"app_id": self.app_id, "app_secret": self.app_secret})
            body = _json(resp)
            if body.get("code") != 0 or not body.get("tenant_access_token"):
                raise FeishuError(f"获取 tenant_access_token 失败：{body.get('msg') or resp.status_code}", code=body.get("code"), status=resp.status_code)
            ttl = int(body.get("expire") or 7200)
            self._token = (body["tenant_access_token"], time.time() + max(60, ttl - 300))
            return self._token[0]

    def request(self, method: str, path: str, *, params: dict[str, Any] | None = None, json: Any = None) -> dict[str, Any]:
        """返回响应里的 data；code != 0 抛 FeishuError（token 过期自动刷新重试，限流退避重试）。"""
        refreshed = False
        for attempt in range(self.max_retries + 1):
            token = self.tenant_token()
            resp = self._http.request(method, path, params={k: v for k, v in (params or {}).items() if v is not None},
                                      json=json, headers={"Authorization": f"Bearer {token}"})
            body = _json(resp)
            code = body.get("code")
            if code == 0:
                return body.get("data") or {}
            if code in _TOKEN_EXPIRED and not refreshed:
                refreshed = True
                self._token = None
                continue
            if (code in _RATE_LIMITED or resp.status_code == 429) and attempt < self.max_retries:
                time.sleep(min(4.0, 0.5 * 2 ** attempt))
                continue
            raise FeishuError(_explain(code, body.get("msg") or resp.reason_phrase, path), code=code, status=resp.status_code)
        raise FeishuError(f"飞书接口重试后仍失败：{method} {path}")

    def download_media(self, token: str) -> bytes:
        """文档里图片块的 image.token → 图片字节。成功时返回二进制；失败时响应是 JSON 错误，按 request 的规则重试 / 抛错。"""
        path = f"/drive/v1/medias/{token}/download"
        refreshed = False
        for attempt in range(self.max_retries + 1):
            resp = self._http.get(path, headers={"Authorization": f"Bearer {self.tenant_token()}"})
            if resp.status_code == 200 and "json" not in resp.headers.get("content-type", ""):
                return resp.content
            body = _json(resp)
            code = body.get("code")
            if code in _TOKEN_EXPIRED and not refreshed:
                refreshed = True
                self._token = None
                continue
            if (code in _RATE_LIMITED or resp.status_code == 429) and attempt < self.max_retries:
                time.sleep(min(4.0, 0.5 * 2 ** attempt))
                continue
            raise FeishuError(_explain(code, body.get("msg") or resp.reason_phrase, path), code=code, status=resp.status_code)
        raise FeishuError(f"飞书接口重试后仍失败：GET {path}")

    # ------------------------------------------------------------------ 文档读取
    def get_document(self, document_id: str) -> dict[str, Any]:
        """{document_id, revision_id, title}"""
        return self.request("GET", f"/docx/v1/documents/{document_id}").get("document") or {}

    def list_blocks(self, document_id: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token = None
        while True:
            data = self.request("GET", f"/docx/v1/documents/{document_id}/blocks", params={"page_size": 500, "page_token": page_token, "document_revision_id": -1})
            items.extend(data.get("items") or [])
            if not data.get("has_more") or not data.get("page_token"):
                return items
            page_token = data["page_token"]

    def resolve(self, ref: DocRef) -> DocRef:
        """知识库节点 → 实际的文档 token / 类型；其它类型原样返回。"""
        if ref.kind != "wiki":
            return ref
        node = self.request("GET", "/wiki/v2/spaces/get_node", params={"token": ref.token}).get("node") or {}
        if not node.get("obj_token"):
            raise FeishuError(f"知识库节点 {ref.token} 没有对应的文档")
        return DocRef(str(node.get("obj_type") or "docx"), node["obj_token"])

    def list_folder(self, folder_token: str | None = None) -> list[dict[str, Any]]:
        """文件夹下的文件：[{token, name, type, url, created_time, modified_time, owner_id}]；不给 folder_token 列应用的根目录。"""
        files: list[dict[str, Any]] = []
        page_token = None
        while True:
            data = self.request("GET", "/drive/v1/files", params={"folder_token": folder_token or None, "page_size": 200, "page_token": page_token})
            files.extend(data.get("files") or [])
            if not data.get("has_more") or not data.get("next_page_token"):
                return files
            page_token = data["next_page_token"]

    # ------------------------------------------------------------------ 文档写入
    def create_document(self, title: str, folder_token: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"title": title[:800]}
        if folder_token:
            body["folder_token"] = folder_token
        return self.request("POST", "/docx/v1/documents", json=body).get("document") or {}

    def append_children(self, document_id: str, blocks: list[dict[str, Any]], *, parent_id: str | None = None, index: int = -1) -> int:
        """在 parent（默认文档根块）下追加块，按 50 个一批。返回写入的块数。"""
        parent = parent_id or document_id
        written = 0
        for i in range(0, len(blocks), _MAX_CHILDREN):
            batch = blocks[i:i + _MAX_CHILDREN]
            self.request("POST", f"/docx/v1/documents/{document_id}/blocks/{parent}/children",
                         params={"document_revision_id": -1, "client_token": str(uuid.uuid4())},
                         json={"children": batch, "index": -1 if index < 0 else index + written})
            written += len(batch)
        return written

    def convert_markdown(self, markdown: str) -> tuple[list[str], list[dict[str, Any]]]:
        """官方 Markdown → 块转换：(第一层块 id, 全部块)。表格块带的只读 merge_info 要去掉才能插入。"""
        data = self.request("POST", "/docx/v1/documents/blocks/convert", json={"content_type": "markdown", "content": markdown})
        blocks = data.get("blocks") or []
        for b in blocks:
            table = b.get("table")
            if isinstance(table, dict) and isinstance(table.get("property"), dict):
                table["property"].pop("merge_info", None)
        return list(data.get("first_level_block_ids") or []), blocks

    def insert_descendants(self, document_id: str, first_level: list[str], blocks: list[dict[str, Any]], *, index: int = -1) -> int:
        if len(blocks) > _MAX_DESCENDANTS:
            raise FeishuError(f"一次最多插入 {_MAX_DESCENDANTS} 个块（本次 {len(blocks)} 个）")
        self.request("POST", f"/docx/v1/documents/{document_id}/blocks/{document_id}/descendant",
                     params={"document_revision_id": -1, "client_token": str(uuid.uuid4())},
                     json={"children_id": first_level, "descendants": blocks, "index": index})
        return len(blocks)

    def delete_children(self, document_id: str, start: int, end: int, *, parent_id: str | None = None) -> None:
        """删除 parent 下 [start, end) 的子块。"""
        if end <= start:
            return
        self.request("DELETE", f"/docx/v1/documents/{document_id}/blocks/{parent_id or document_id}/children/batch_delete",
                     params={"document_revision_id": -1, "client_token": str(uuid.uuid4())}, json={"start_index": start, "end_index": end})

    def update_text(self, document_id: str, block_id: str, elements: list[dict[str, Any]]) -> None:
        self.request("PATCH", f"/docx/v1/documents/{document_id}/blocks/{block_id}", params={"document_revision_id": -1, "client_token": str(uuid.uuid4())},
                     json={"update_text_elements": {"elements": elements}})

    def delete_file(self, token: str, kind: str = "docx") -> None:
        self.request("DELETE", f"/drive/v1/files/{token}", params={"type": kind})

    def set_link_share(self, token: str, entity: str, kind: str = "docx") -> None:
        """链接分享范围：tenant_readable / tenant_editable（组织内获得链接的人可读 / 可编辑）/ closed。"""
        self.request("PATCH", f"/drive/v1/permissions/{token}/public", params={"type": kind}, json={"link_share_entity": entity})

    def subscribe_file(self, token: str, kind: str = "docx") -> None:
        """订阅文档变更：飞书只给订阅过的文档推送 drive.file.edit_v1 等事件。"""
        self.request("POST", f"/drive/v1/files/{token}/subscribe", params={"file_type": kind})

    def web_url(self, document_id: str, web_base: str | None = None) -> str:
        return f"{(web_base or WEB_BASES[self.domain]).rstrip('/')}/docx/{document_id}"


def _json(resp: httpx.Response) -> dict[str, Any]:
    try:
        body = resp.json()
    except ValueError:
        raise FeishuError(f"飞书返回了非 JSON 响应（HTTP {resp.status_code}）：{resp.text[:200]}", status=resp.status_code) from None
    return body if isinstance(body, dict) else {}


_HINTS = {
    99991672: "应用没有开通所需权限（docx:document / drive:drive / wiki:wiki:readonly；读图片要 docs:document.media:download），在开发者后台「权限管理」开通并发布版本",
    99991679: "应用没有开通所需权限，在开发者后台「权限管理」开通并发布版本",
    1770032: "应用没有这篇文档的权限：在文档右上角「…→ 添加文档应用」里加上本应用",
    1770002: "文档不存在或已删除",
    1061004: "应用没有该文件 / 文件夹的权限",
    131006: "应用没有知识库节点的权限：把应用加为知识空间成员",
}


def _explain(code: Any, msg: str, path: str) -> str:
    hint = _HINTS.get(code) if isinstance(code, int) else None
    return f"飞书接口错误 {code}：{msg}（{path}）" + (f"——{hint}" if hint else "")
