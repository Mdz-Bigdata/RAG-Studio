"""语料质量体检：在"能不能检索到"之前，先看"库里的东西干不干净"。

入库清洗（cleaning.py）只看单篇文档里的块；很多问题要放到整库才看得出来，或者清洗阈值放过了：

切片级（以 parent——阅读粒度——为单位，剔除时连同其 child 一起移出检索）
    garbled         乱码 / 编码错误 / 私用区字符（扫描件 OCR 失败、字体映射错误）
    low_info        低信息量：目录行（"第一章……3"）、页码、纯数字 / 符号
    too_short       过短：只有一个标题或半句话，召回了也答不了问题，还挤占 top-K
    oversized       超长：非表格 / 代码却远超 parent 上限，通常是断行修复失败把整页并成一段
    duplicate       完全重复：同一段内容在多份文档（或同一文档多处）出现，权重虚高、挤占上下文
    near_duplicate  近似重复（MinHash + LSH，Jaccard ≥ 阈值）：新旧版本的同一条款、模板化段落

文档级
    expired           已过失效日期（检索时已被过滤，建议下线）
    version_conflict  同名文档多个版本同时生效：新旧制度"反复横跳"的根源
    empty             没有产生任何检索切片
    missing_meta      版本 / 生效日期 / 部门 / 标签全空：检索前无法做任何硬过滤
    poor_parse        三成以上切片有问题：解析质量差，建议换解析档位或人工校对

剔除以内容指纹（归一化正文的哈希）记录，重新解析 / 重建索引时同样生效，可随时恢复。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import zlib
from collections import defaultdict
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..index import Index
    from ..schemas import Chunk, DocMetadata

CHUNK_ISSUES: dict[str, tuple[str, str, str]] = {
    # type: (标签, 严重度, 处置建议)
    "garbled": ("乱码 / 编码错误", "high", "剔除；或换 MinerU 档位 / 开启 OCR 后重新解析"),
    "duplicate": ("完全重复", "medium", "剔除重复副本，保留一份"),
    "near_duplicate": ("近似重复", "medium", "多为新旧版本的同一条款：剔除旧版副本，或给旧文档设置失效日期"),
    "low_info": ("低信息量（目录 / 页码 / 符号）", "medium", "剔除"),
    "too_short": ("内容过短", "low", "剔除，或调大 chunking.parent_min_tokens 让短块并入相邻块"),
    "oversized": ("超长切片", "low", "检查 PDF 断行修复或切片参数；表格 / 代码块不受影响"),
}
DOC_ISSUES: dict[str, tuple[str, str]] = {
    "expired": ("已过期", "medium"),
    "version_conflict": ("多版本同时生效", "high"),
    "empty": ("没有检索切片", "high"),
    "poor_parse": ("解析质量差", "high"),
    "missing_meta": ("缺少元数据", "low"),
}
_SEVERITY_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.3}

_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)
_CJK = re.compile(r"[一-鿿㐀-䶿]")
_INFO = re.compile(r"[一-鿿㐀-䶿A-Za-z]")
_SENT_END = re.compile(r"[。！？!?；;]|\.\s|\.$")
_MOJIBAKE = re.compile(r"[ÃÂâ][\x80-\xbf€™œ]|\ufffd|锟斤拷|烫烫|屯屯")   # UTF-8 被当成 Latin-1 解码 / 替换字符
_TOC_LINE = re.compile(r"^\s*(?:第?[\d一二三四五六七八九十]+[章节条.、]?\s*)?.{1,50}?(?:[.·…\-—_ ]{4,}|\s{3,})\s*\d{1,4}\s*$")
_VERSIONISH = re.compile(r"[（(【\[]?\s*(?:20\d{2}\s*年?\s*(?:版|修订版?|修正)?|[Vv]?\d+(?:\.\d+)+|修订版?|最新版?|正式版|试行|草案|征求意见稿|第[一二三四五六七八九十\d]+版)\s*[）)】\]]?")


def normalize(text: str) -> str:
    return _PUNCT.sub("", text).lower()


def fingerprint(text: str) -> str:
    """内容指纹：去掉空白与标点后的哈希。切片 id 随重新解析变化，指纹不变。"""
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:16]


def exclusion_key(doc_id: str, text: str) -> str:
    """剔除清单的键：「文档 id:内容指纹」。只按指纹会把别的文档里内容相同的段落（要保留的副本）一起剔掉。"""
    return f"{doc_id}:{fingerprint(text)}"


# ----------------------------------------------------------------------------------------
def _garbled(text: str) -> float:
    """异常字符占比：私用区 / 未分配 / 控制字符（字体映射错误、OCR 失败）+ 典型的 UTF-8 乱码序列。"""
    t = re.sub(r"\s+", "", text)
    if not t:
        return 0.0
    bad = sum(1 for ch in t if unicodedata.category(ch) in ("Co", "Cn", "Cc", "Cs"))
    return min(1.0, (bad + 3 * len(_MOJIBAKE.findall(t))) / len(t))


def _low_info(text: str) -> str | None:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 3 and sum(1 for ln in lines if _TOC_LINE.match(ln)) / len(lines) >= 0.5:
        return "目录页"
    t = re.sub(r"\s+", "", text)
    if len(t) >= 8 and len(_INFO.findall(t)) / len(t) < 0.25:
        return "几乎全是数字 / 符号"
    return None


def _shingles(norm: str, k: int = 5) -> np.ndarray:
    if len(norm) < k:
        return np.array([zlib.crc32(norm.encode("utf-8"))], dtype=np.uint64)
    return np.unique(np.fromiter((zlib.crc32(norm[i:i + k].encode("utf-8")) for i in range(len(norm) - k + 1)), dtype=np.uint64))


class _MinHash:
    P = np.uint64((1 << 31) - 1)

    def __init__(self, n: int = 64, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, (1 << 31) - 1, n, dtype=np.uint64)
        self.b = rng.integers(0, (1 << 31) - 1, n, dtype=np.uint64)

    def __call__(self, sh: np.ndarray) -> np.ndarray:
        sh = sh % self.P
        return ((self.a[:, None] * sh[None, :] + self.b[:, None]) % self.P).min(axis=1)


def _title_key(title: str) -> str:
    return normalize(_VERSIONISH.sub("", title or ""))


def _version_tuple(v: str | None) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v or ""))[:4]


# ----------------------------------------------------------------------------------------
class CorpusHealthChecker:
    def __init__(self, index: "Index", *, near_dup_threshold: float = 0.85, garbled_threshold: float = 0.15, min_chars: int = 36) -> None:
        self.index = index
        self.near_dup_threshold = near_dup_threshold
        self.garbled_threshold = garbled_threshold
        self.min_chars = min_chars
        ck = index.settings.chunking
        self.parent_max_tokens = ck.parent_max_tokens

    def run(self, on_date: date | None = None, titles: dict[str, str] | None = None) -> dict[str, Any]:
        on_date = on_date or date.today()
        store = self.index.store
        titles = titles or {}
        docs: dict[str, "DocMetadata"] = store.docs
        parents = sorted((c for c in store.chunks.values() if c.kind == "parent"), key=lambda c: (c.doc_id, c.order))
        children_by_doc: dict[str, int] = defaultdict(int)
        for c in store.chunks.values():
            if c.kind == "child":
                children_by_doc[c.doc_id] += 1

        def doc_title(doc_id: str) -> str:
            md = docs.get(doc_id)
            return titles.get(doc_id) or (md.title if md and md.title else "") or (md.source.rsplit("/", 1)[-1] if md else doc_id)

        flagged: dict[str, dict[str, Any]] = {}

        def flag(p: "Chunk", issue: str, detail: str, ref: "Chunk | None" = None) -> None:
            item = flagged.setdefault(p.chunk_id, {
                "chunkId": p.chunk_id, "docId": p.doc_id, "docTitle": doc_title(p.doc_id), "breadcrumb": p.breadcrumb_text,
                "issues": [], "details": [], "preview": p.text[:220], "tokens": p.token_count, "fingerprint": fingerprint(p.text),
                "page": None if p.page_start is None else p.page_start + 1, "duplicateOf": None,
            })
            if issue not in item["issues"]:
                item["issues"].append(issue)
                item["details"].append(detail)
            if ref is not None and item["duplicateOf"] is None:
                item["duplicateOf"] = {"chunkId": ref.chunk_id, "docId": ref.doc_id, "docTitle": doc_title(ref.doc_id), "breadcrumb": ref.breadcrumb_text}

        # --- 单块检查 ------------------------------------------------------------------
        norms: dict[str, str] = {}
        for p in parents:
            atomic = any(t in ("table", "code", "equation", "image") for t in p.block_types)
            norm = normalize(p.text)
            norms[p.chunk_id] = norm
            g = _garbled(p.text)
            moji = len(_MOJIBAKE.findall(p.text))
            if g >= self.garbled_threshold or moji >= 2:     # "锟斤拷"之类的典型乱码出现两次就足以确认
                flag(p, "garbled", f"异常字符占比 {g:.0%}" + (f"，{moji} 处典型乱码序列" if moji else ""))
            if not atomic:
                if why := _low_info(p.text):
                    flag(p, "low_info", why)
                elif len(norm) < self.min_chars // 3 or (len(norm) < self.min_chars and not _SENT_END.search(p.text)):
                    # 一句完整的话（"病假需提供二级以上医院的诊断证明。"）再短也能回答问题；只标记半句话 / 孤立标题
                    flag(p, "too_short", f"正文仅 {len(norm)} 字，且不成句")
                if p.token_count > self.parent_max_tokens * 2:
                    flag(p, "oversized", f"{p.token_count} token，超过 parent 上限 {self.parent_max_tokens} 的两倍")

        # --- 重复 / 近似重复 ------------------------------------------------------------
        def rank(c: "Chunk") -> tuple:
            """保留谁：现行文档 > 过期文档；生效日期新 > 旧；版本高 > 低；同文档内保留第一次出现。"""
            md = docs.get(c.doc_id)
            eff = 1 if (md is None or md.is_effective(on_date)) else 0
            vf = md.valid_from.toordinal() if md and md.valid_from else 0
            return (-eff, -vf, tuple(-x for x in _version_tuple(md.version if md else None)), c.doc_id, c.order)

        by_fp: dict[str, list["Chunk"]] = defaultdict(list)
        for p in parents:
            if len(norms[p.chunk_id]) >= 12:
                by_fp[fingerprint(p.text)].append(p)
        exact_dup: set[str] = set()
        for group in by_fp.values():
            if len(group) > 1:
                keep = min(group, key=rank)
                for c in group:
                    if c is not keep:
                        flag(c, "duplicate", f"与《{doc_title(keep.doc_id)}》中的段落完全相同", keep)
                        exact_dup.add(c.chunk_id)

        cands = [p for p in parents if len(norms[p.chunk_id]) >= max(self.min_chars, 60) and p.chunk_id not in exact_dup]
        if len(cands) > 1:
            mh = _MinHash()
            sh = {p.chunk_id: _shingles(norms[p.chunk_id]) for p in cands}
            sigs = np.stack([mh(sh[p.chunk_id]) for p in cands])
            buckets: dict[tuple, list[int]] = defaultdict(list)
            bands, rows = 16, 4
            for i in range(len(cands)):
                for b in range(bands):
                    buckets[(b, *sigs[i, b * rows:(b + 1) * rows].tolist())].append(i)
            parent_of = list(range(len(cands)))

            def find(x: int) -> int:
                while parent_of[x] != x:
                    parent_of[x] = parent_of[parent_of[x]]
                    x = parent_of[x]
                return x

            checked: set[tuple[int, int]] = set()
            sims: dict[int, float] = {}
            for members in buckets.values():
                if len(members) < 2 or len(members) > 50:
                    continue
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        i, j = members[x], members[y]
                        if (i, j) in checked:
                            continue
                        checked.add((i, j))
                        a, b_ = sh[cands[i].chunk_id], sh[cands[j].chunk_id]
                        inter = len(np.intersect1d(a, b_, assume_unique=True))
                        jac = inter / (len(a) + len(b_) - inter)
                        if jac >= self.near_dup_threshold:
                            parent_of[find(i)] = find(j)
                            sims[i] = max(sims.get(i, 0), jac)
                            sims[j] = max(sims.get(j, 0), jac)
            groups: dict[int, list[int]] = defaultdict(list)
            for i in range(len(cands)):
                groups[find(i)].append(i)
            for members in groups.values():
                if len(members) < 2:
                    continue
                keep = min((cands[i] for i in members), key=rank)
                for i in members:
                    if cands[i] is not keep:
                        flag(cands[i], "near_duplicate", f"与《{doc_title(keep.doc_id)}》的段落相似度 {sims.get(i, self.near_dup_threshold):.0%}", keep)

        # --- 文档级 ----------------------------------------------------------------------
        issue_parents: dict[str, int] = defaultdict(int)
        parents_by_doc: dict[str, int] = defaultdict(int)
        for p in parents:
            parents_by_doc[p.doc_id] += 1
        for item in flagged.values():
            issue_parents[item["docId"]] += 1

        doc_rows: dict[str, dict[str, Any]] = {}
        for doc_id, md in docs.items():
            row = {
                "docId": doc_id, "title": doc_title(doc_id), "version": md.version, "validFrom": md.valid_from.isoformat() if md.valid_from else None,
                "validTo": md.valid_to.isoformat() if md.valid_to else None, "parents": parents_by_doc.get(doc_id, 0), "children": children_by_doc.get(doc_id, 0),
                "flagged": issue_parents.get(doc_id, 0), "issues": [],
            }
            doc_rows[doc_id] = row

            def add(issue: str, detail: str, row=row) -> None:
                label, sev = DOC_ISSUES[issue]
                row["issues"].append({"type": issue, "label": label, "severity": sev, "detail": detail})

            if md.valid_to and md.valid_to < on_date:
                add("expired", f"已于 {md.valid_to.isoformat()} 失效，检索时自动过滤")
            if row["children"] == 0:
                add("empty", "解析后没有产生检索切片：可能是扫描件未 OCR、或内容全部被清洗掉")
            elif row["flagged"] >= 3 and row["flagged"] / max(1, row["parents"]) >= 0.3:
                add("poor_parse", f"{row['flagged']}/{row['parents']} 个段落有质量问题：建议换解析档位、开启 OCR，或人工校对")
            if not (md.version or md.valid_from or md.department or md.tags):
                add("missing_meta", "版本 / 生效日期 / 部门 / 标签全空：无法按时效或范围过滤，可在「元数据」页自动抽取")

        # 同名多版本同时生效
        by_title: dict[str, list[str]] = defaultdict(list)
        for doc_id, md in docs.items():
            key = _title_key(doc_title(doc_id))
            if len(key) >= 4 and md.is_effective(on_date):
                by_title[key].append(doc_id)
        conflicts: list[dict[str, Any]] = []
        for key, ids in by_title.items():
            if len(ids) < 2:
                continue
            ordered = sorted(ids, key=lambda d: (docs[d].valid_from or date.min, _version_tuple(docs[d].version), docs[d].created_at), reverse=True)
            conflicts.append({"key": key, "keepDocId": ordered[0], "docs": [{**{k: doc_rows[d][k] for k in ("docId", "title", "version", "validFrom", "validTo")}, "keep": d == ordered[0]} for d in ordered]})
            for d in ordered[1:]:
                label, sev = DOC_ISSUES["version_conflict"]
                doc_rows[d]["issues"].append({"type": "version_conflict", "label": label, "severity": sev, "detail": f"与《{doc_rows[ordered[0]]['title']}》疑似同一文档的不同版本，且都在生效期内"})

        # --- 汇总 ----------------------------------------------------------------------
        chunks = sorted(flagged.values(), key=lambda x: (-max(_SEVERITY_WEIGHT[CHUNK_ISSUES[i][1]] for i in x["issues"]), x["docTitle"], x["chunkId"]))
        for item in chunks:
            item["severity"] = max(item["issues"], key=lambda i: _SEVERITY_WEIGHT[CHUNK_ISSUES[i][1]])
            item["severity"] = CHUNK_ISSUES[item["severity"]][1]
        issue_counts = {k: sum(1 for c in chunks if k in c["issues"]) for k in CHUNK_ISSUES}
        doc_issue_counts = {k: sum(1 for d in doc_rows.values() if any(i["type"] == k for i in d["issues"])) for k in DOC_ISSUES}

        n = len(parents)
        weighted = sum(max(_SEVERITY_WEIGHT[CHUNK_ISSUES[i][1]] for i in c["issues"]) for c in chunks)
        score = 100.0 * (1 - weighted / n) if n else 100.0
        score -= min(20, 5 * len(conflicts)) + min(10, 2 * doc_issue_counts["expired"]) + min(15, 5 * (doc_issue_counts["empty"] + doc_issue_counts["poor_parse"]))
        score = max(0, round(score))
        for row in doc_rows.values():
            p = row["parents"]
            row["score"] = round(100 * (1 - row["flagged"] / p)) if p else 0

        return {
            "score": score,
            "grade": "优" if score >= 90 else "良" if score >= 75 else "中" if score >= 60 else "差",
            "totals": {"docs": len(docs), "parents": n, "children": sum(children_by_doc.values()), "flagged": len(chunks)},
            "issueCounts": issue_counts,
            "docIssueCounts": doc_issue_counts,
            "issueLabels": {k: {"label": v[0], "severity": v[1], "action": v[2]} for k, v in CHUNK_ISSUES.items()},
            "chunks": chunks,
            "docs": sorted(doc_rows.values(), key=lambda d: (-len(d["issues"]), d["score"], d["title"])),
            "conflicts": conflicts,
            "recommendations": self._recommend(issue_counts, doc_issue_counts, conflicts, n),
        }

    @staticmethod
    def _recommend(ic: dict[str, int], dc: dict[str, int], conflicts: list, n: int) -> list[str]:
        rec: list[str] = []
        if conflicts:
            rec.append(f"{len(conflicts)} 组文档疑似同一制度的多个版本同时生效 → 给旧版本设置失效日期（检索时自动过滤），避免新旧规定一起被召回")
        dup = ic["duplicate"] + ic["near_duplicate"]
        if dup:
            rec.append(f"{dup} 个段落重复或近似重复 → 剔除副本：重复内容会挤占 top-K，让模型只看到同一段话的多个拷贝")
        if ic["garbled"]:
            rec.append(f"{ic['garbled']} 个段落含乱码 → 对来源文档换 MinerU 档位 / 开启 OCR 重新解析，或直接剔除")
        if ic["low_info"] + ic["too_short"]:
            rec.append(f"{ic['low_info'] + ic['too_short']} 个段落是目录、页码或只有半句话 → 剔除；数量多时调大 chunking.parent_min_tokens")
        if dc["poor_parse"] or dc["empty"]:
            rec.append(f"{dc['poor_parse'] + dc['empty']} 份文档解析质量差或没有切片 → 在知识库页「校对」或「重新解析」")
        if dc["expired"]:
            rec.append(f"{dc['expired']} 份文档已过期 → 检索时已自动过滤，确认无用后删除")
        if dc["missing_meta"]:
            rec.append(f"{dc['missing_meta']} 份文档缺少元数据 → 在「元数据」页自动抽取版本、生效日期与部门")
        if not rec:
            rec.append("语料质量良好：没有发现需要处理的问题" if n else "知识库还没有切片")
        return rec
