"""OKF（Open Knowledge Format，开放知识格式）知识包：一个目录（通常是 Git 仓库）里的若干 Markdown 文件，
每个文件一个概念——一项指标、一个 API、一张表、一份运维手册——文件头是 YAML front matter：

    ---
    type: metric
    title: "Monthly Churn Rate"
    description: "Official formula for calculating monthly customer churn."
    owner: "data-engineering"
    tags: [revenue, kpi, board-report]
    timestamp: 2026-06-20T10:00:00Z
    aliases: [月流失率, churn]          # 可选：本系统扩展，命中别名也算命中
    ---
    # Monthly Churn Rate
    ...

这里只做两件事：扫描知识包得到条目列表（scan_bundle），以及把条目写成本系统能识别的 Markdown（okf_markdown）——
front matter 里补一段 okf 元数据和「OKF」标签，解析后落在 DocMetadata.extra["okf"]，检索时据此认出"这是权威条目"。
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+\.md)\)")
_SKIP_DIRS = {".git", ".github", "node_modules", ".venv", "__pycache__"}
OKF_TAG = "OKF"


@dataclass
class OKFEntry:
    path: str                          # 相对知识包根目录
    title: str
    text: str                          # 文件原文（含 front matter）
    sha1: str
    type: str = ""
    description: str = ""
    owner: str = ""
    timestamp: str = ""
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)   # 正文里链到的其他条目（文件名 stem）
    meta: dict[str, Any] = field(default_factory=dict)  # front matter 原样

    def names(self) -> list[str]:
        """检索时按这些名字认条目：标题 + 别名，去重、去空。"""
        out: list[str] = []
        for n in [self.title, *self.aliases]:
            n = " ".join(str(n).split())
            if n and n.casefold() not in {x.casefold() for x in out}:
                out.append(n)
        return out


def _as_list(v: Any) -> list[str]:
    if isinstance(v, (list, tuple, set)):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return []


def _as_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return str(v).strip()


def parse_entry(path: Path, root: Path) -> OKFEntry | None:
    """一个 OKF 文件 → 条目。没有 front matter 或没有 title 的不算条目（普通 Markdown 也可能混在仓库里）。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("OKF：读不了 %s：%s", path, exc)
        return None
    m = _FRONT.match(text)
    if not m:
        return None
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        logger.warning("OKF：%s 的 front matter 不是合法 YAML：%s", path.name, exc)
        return None
    if not isinstance(meta, dict) or not _as_text(meta.get("title")):
        return None
    body = text[m.end():]
    aliases = _as_list(meta.get("aliases")) + _as_list(meta.get("synonyms"))
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    if stem and stem.casefold() != _as_text(meta["title"]).casefold():
        aliases.append(stem)
    return OKFEntry(
        path=str(path.relative_to(root)).replace("\\", "/"), title=_as_text(meta["title"]), text=text,
        sha1=hashlib.sha1(text.encode("utf-8")).hexdigest(), type=_as_text(meta.get("type")), description=_as_text(meta.get("description")),
        owner=_as_text(meta.get("owner")), timestamp=_as_text(meta.get("timestamp") or meta.get("updated") or meta.get("date")),
        tags=_as_list(meta.get("tags")), aliases=aliases, related=[Path(link).stem for _text, link in _LINK.findall(body)], meta=meta,
    )


def scan_bundle(root: Path, subdir: str | None = None) -> list[OKFEntry]:
    """扫描知识包：root（或 root/subdir）下全部 .md，按路径排序。"""
    base = (root / subdir) if subdir else root
    if not base.is_dir():
        raise FileNotFoundError(f"OKF 知识包目录不存在：{base}")
    entries: list[OKFEntry] = []
    for p in sorted(base.rglob("*.md")):
        if any(part in _SKIP_DIRS for part in p.relative_to(base).parts):
            continue
        e = parse_entry(p, base)
        if e is not None:
            entries.append(e)
    return entries


def okf_markdown(entry: OKFEntry, source: str) -> str:
    """条目 → 入库用的 Markdown：front matter 补上 okf 元数据与「OKF」标签，正文原样。
    解析后 title / tags 进 DocMetadata，okf 一段落在 DocMetadata.extra["okf"]。"""
    meta = {k: v for k, v in entry.meta.items() if k not in ("tags",)}
    meta["title"] = entry.title
    meta["tags"] = sorted({*entry.tags, OKF_TAG})
    meta["okf"] = {
        "type": entry.type, "owner": entry.owner, "timestamp": entry.timestamp, "description": entry.description,
        "path": entry.path, "source": source, "aliases": entry.aliases, "related": entry.related, "sha1": entry.sha1,
    }
    for k, v in list(meta.items()):           # datetime 之类写回 YAML 再读会变成别的类型：统一成字符串
        if isinstance(v, (datetime, date)):
            meta[k] = v.isoformat()
    body = entry.text[_FRONT.match(entry.text).end():] if _FRONT.match(entry.text) else entry.text
    return "---\n" + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip() + "\n---\n" + body.lstrip("\n")
