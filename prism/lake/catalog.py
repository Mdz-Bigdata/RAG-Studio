"""Agent 湖目录：让分析智能体先"理解数据"，再去查数据。

思路来自火山引擎 LAS 的 Lance 数据检索 Agent（LanceDB + DuckDB）：表 / 列 / 指标 / Few-Shot 示例 / 历史查询都是元数据，
每一条配上向量，分析前先按意图做一次 catalog discovery——只把与问题相关的列、meta 里的键、指标口径和示例喂给模型，
而不是把整张表的说明一股脑塞进提示词。四类目录项：

- column   lake 表的列：类型、说明、真实取值（低基数列列出全部取值和计数，数值列给范围和空值数）
- meta     meta JSON 里按 kind 出现的键：取值示例 + 对应的 json_extract_string 写法（SQL 写错最多的地方）
- metric   指标口径：拒答率、低置信率、缺口率、延迟分位、点踩率……每条带固定的 SQL，模型照抄而不是自己定义
- example  问题 → SQL 的示例：内置几条，再加上湖里以前分析成功的 SQL（历史查询记录就是最贴合这个库的 Few-Shot）

人工维护的条目（自定义指标 / 示例 / 列备注）也存在湖里：kind = note、agent = catalog、meta.catalog 标明类型，
跟着湖一起有版本、可检索、可回填；保存时 SQL 会先在当前快照上试跑一遍，跑不通不收。
目录随快照（版本 / 分支）建一次并缓存；向量用湖自己的 embedder，与湖里的记录同一个空间。
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Any

import numpy as np

from .analytics import COLUMN_DOCS, _json_obj

_LOW_CARDINALITY = 24          # 不同取值不超过这么多的文本列，目录里列出全部取值
_META_SAMPLE_ROWS = 300        # 每种 kind 扫多少行 meta 来发现键
_META_VALUE_CHARS = 40
_SPLIT = re.compile(r"[\s，。；：、？！「」“”‘’/（）()\[\]{}<>=\-·|,.;:?!'\"]+")

# 列的口语别名：问题里出现这些词就把该列挑进目录（向量相似度之外的关键词命中，离线哈希向量下也能用）
COLUMN_ALIASES: dict[str, list[str]] = {
    "latency_ms": ["延迟", "耗时", "慢", "快", "p95", "p50", "毫秒"], "score": ["分数", "置信", "相关度", "重排分", "低分"],
    "feedback": ["点赞", "点踩", "反馈", "差评", "好评"], "status": ["拒答", "状态", "失败", "没检索到", "未检索到", "出错"],
    "day": ["每天", "每日", "趋势", "日期", "哪天", "最近", "本周", "上周"], "ts": ["时间", "最近", "小时", "之前", "之后"],
    "kb_id": ["知识库", "哪个库", "分库"], "session_id": ["会话", "追问", "多轮"], "user_id": ["用户", "谁"],
    "kind": ["类型", "记录"], "meta": ["策略", "改写", "命中文档", "引用"], "content": ["问题", "内容", "问了"], "title": ["标题"],
    "parent_id": ["关联", "对应的回答", "对应的检索"], "agent": ["智能体"], "refs": ["引用", "文档"], "embed_model": ["向量模型"],
}

# 指标口径：名称、一句话定义、固定 SQL（表名 lake）。low_score 由配置带入
METRICS: list[dict[str, str]] = [
    {"name": "每日提问量", "aliases": "每天 每日 趋势 提问量 多少提问", "unit": "条/天", "definition": "kind = 'question' 的记录按 day 计数",
     "sql": "SELECT day, count(*) AS questions FROM lake WHERE kind = 'question' GROUP BY day ORDER BY day"},
    {"name": "拒答率", "aliases": "拒答 答不上 没答", "unit": "%", "definition": "status = 'refused' 的回答数 / 全部回答数（kind = 'answer'）",
     "sql": "SELECT round(100.0 * count(*) FILTER (WHERE status = 'refused') / count(*), 1) AS refusal_rate_pct, count(*) AS answers FROM lake WHERE kind = 'answer'"},
    {"name": "低置信率", "aliases": "置信 低分 分数低", "unit": "%", "definition": "检索最高分 score < {low_score} 的检索次数 / 有分数的检索次数（kind = 'retrieval'）；status = 'no_context' 单独算未检索到",
     "sql": "SELECT round(100.0 * count(*) FILTER (WHERE score < {low_score}) / count(*), 1) AS low_confidence_pct, count(*) AS retrievals FROM lake WHERE kind = 'retrieval' AND score IS NOT NULL"},
    {"name": "未检索到率", "aliases": "没检索到 未检索到 检索不到 no_context", "unit": "%", "definition": "status = 'no_context' 的检索次数 / 检索次数",
     "sql": "SELECT round(100.0 * count(*) FILTER (WHERE status = 'no_context') / count(*), 1) AS no_context_pct, count(*) AS retrievals FROM lake WHERE kind = 'retrieval'"},
    {"name": "知识缺口率", "aliases": "缺口 补语料 答不上", "unit": "%", "definition": "有缺口的问题数 / 问题数。缺口 = 未检索到 或 score < {low_score} 或 回答拒答 或 被点踩（与知识缺口页同一口径，但不扣除已复测解决的）",
     "sql": ("WITH q AS (SELECT q.id, q.day, q.content AS question, r.status AS r_status, r.score, a.status AS a_status, a.feedback\n"
             "  FROM lake q LEFT JOIN lake r ON r.parent_id = q.id AND r.kind = 'retrieval' LEFT JOIN lake a ON a.parent_id = q.id AND a.kind = 'answer'\n"
             "  WHERE q.kind = 'question')\n"
             "SELECT count(*) AS questions, count(*) FILTER (WHERE r_status = 'no_context' OR score < {low_score} OR a_status = 'refused' OR feedback < 0) AS gaps,\n"
             "       round(100.0 * count(*) FILTER (WHERE r_status = 'no_context' OR score < {low_score} OR a_status = 'refused' OR feedback < 0) / count(*), 1) AS gap_rate_pct FROM q")},
    {"name": "检索延迟分位", "aliases": "延迟 耗时 p95 p50 慢", "unit": "ms", "definition": "kind = 'retrieval' 的 latency_ms 的 p50 / p95（quantile_cont）",
     "sql": "SELECT round(quantile_cont(latency_ms, 0.5)) AS p50_ms, round(quantile_cont(latency_ms, 0.95)) AS p95_ms, count(*) AS retrievals FROM lake WHERE kind = 'retrieval' AND latency_ms IS NOT NULL"},
    {"name": "点踩率", "aliases": "点踩 差评 反馈 点赞", "unit": "%", "definition": "feedback = -1 的回答数 / 有反馈（feedback <> 0）的回答数",
     "sql": "SELECT round(100.0 * count(*) FILTER (WHERE feedback < 0) / nullif(count(*) FILTER (WHERE feedback <> 0), 0), 1) AS downvote_pct, count(*) FILTER (WHERE feedback <> 0) AS rated FROM lake WHERE kind = 'answer'"},
    {"name": "检索策略对比", "aliases": "策略 hyde 混合 对比 哪种", "unit": "", "definition": "按 meta.$.strategy 分组：次数、平均最高分、延迟中位数",
     "sql": "SELECT json_extract_string(meta, '$.strategy') AS strategy, count(*) AS n, round(avg(score), 4) AS avg_score, round(median(latency_ms)) AS p50_ms FROM lake WHERE kind = 'retrieval' GROUP BY 1 ORDER BY n DESC"},
    {"name": "被检索最多的文档", "aliases": "文档 热门 引用最多 命中最多", "unit": "次", "definition": "展开 retrieval 的 meta.$.contexts[*].fileName 计数",
     "sql": "SELECT file_name, count(*) AS retrieved FROM (SELECT unnest(json_extract_string(meta, '$.contexts[*].fileName')) AS file_name FROM lake WHERE kind = 'retrieval') GROUP BY 1 ORDER BY retrieved DESC LIMIT 10"},
    {"name": "会话追问率", "aliases": "追问 会话 多轮", "unit": "%", "definition": "提问数 ≥ 2 的会话数 / 会话数（kind = 'question'）",
     "sql": "WITH s AS (SELECT session_id, count(*) AS n FROM lake WHERE kind = 'question' AND session_id IS NOT NULL GROUP BY session_id)\nSELECT round(100.0 * count(*) FILTER (WHERE n >= 2) / count(*), 1) AS follow_up_pct, count(*) AS sessions FROM s"},
]

# 内置 Few-Shot：问题 → SQL。写法要点：先按 kind 过滤；问题和它的检索 / 回答靠 parent_id 关联；meta 用 json_extract_string
EXAMPLES: list[dict[str, str]] = [
    {"question": "每天有多少提问，平均检索分多少", "sql": "SELECT q.day, count(*) AS questions, round(avg(r.score), 4) AS avg_score FROM lake q LEFT JOIN lake r ON r.parent_id = q.id AND r.kind = 'retrieval' WHERE q.kind = 'question' GROUP BY q.day ORDER BY q.day"},
    {"question": "最近 7 天被点踩或拒答的回答是哪些问题", "sql": "SELECT a.day, q.content AS question, a.status, a.feedback FROM lake a JOIN lake q ON q.id = a.parent_id WHERE a.kind = 'answer' AND (a.status <> 'ok' OR a.feedback < 0) AND a.ts >= epoch(now()) - 7 * 86400 ORDER BY a.ts DESC"},
    {"question": "最慢的 10 次检索", "sql": "SELECT day, title AS question, latency_ms, score FROM lake WHERE kind = 'retrieval' ORDER BY latency_ms DESC NULLS LAST LIMIT 10"},
    {"question": "各知识库各有多少问题，缺口多少", "sql": "SELECT q.kb_id, count(*) AS questions, count(*) FILTER (WHERE r.status = 'no_context' OR a.status = 'refused') AS gaps FROM lake q LEFT JOIN lake r ON r.parent_id = q.id AND r.kind = 'retrieval' LEFT JOIN lake a ON a.parent_id = q.id AND a.kind = 'answer' WHERE q.kind = 'question' GROUP BY q.kb_id ORDER BY questions DESC"},
    {"question": "用了 HyDE 的检索平均分比没用的高吗", "sql": "SELECT json_extract_string(meta, '$.strategy') AS strategy, count(*) AS n, round(avg(score), 4) AS avg_score FROM lake WHERE kind = 'retrieval' GROUP BY 1"},
    {"question": "同一个问题被问过几次", "sql": "SELECT content AS question, count(*) AS asked FROM lake WHERE kind = 'question' GROUP BY content HAVING count(*) > 1 ORDER BY asked DESC LIMIT 20"},
]


def _terms(text: str) -> list[str]:
    """把一句话拆成可做子串匹配的词：按标点 / 空白切，再把 ≥ 4 字的中文段按 2 字窗口展开（"检索延迟分位" → 检索 / 延迟 / 分位 …）。"""
    out: list[str] = []
    for part in _SPLIT.split(text or ""):
        part = part.strip()
        if len(part) < 2:
            continue
        out.append(part)
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", part):
            out += [part[i:i + 2] for i in range(0, len(part) - 1)]
    return out


def _keyword_hit(entry: dict[str, Any], intent: str) -> bool:
    low = (intent or "").lower()
    return any(len(k) >= 2 and k.lower() in low for k in entry.get("keywords", []))


def _short(v: Any, n: int = _META_VALUE_CHARS) -> str:
    s = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _meta_paths(obj: Any, prefix: str = "$") -> list[tuple[str, Any]]:
    """meta JSON → (json path, 示例值)。字典展开一层，列表记成 [*]（列表元素是字典时再展开一层）。"""
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}"
            if isinstance(v, dict):
                out.append((path, v))
                out += [(f"{path}.{k2}", v2) for k2, v2 in v.items() if not isinstance(v2, (dict, list))]
            elif isinstance(v, list):
                out.append((f"{path}[*]", v[:2]))
                first = next((x for x in v if isinstance(x, dict)), None)
                if first is not None:
                    out += [(f"{path}[*].{k2}", v2) for k2, v2 in first.items() if not isinstance(v2, (dict, list))]
            else:
                out.append((path, v))
    return out


CATALOG_AGENT = "catalog"           # 人工维护的目录条目在湖里的 agent 标记
CUSTOM_TYPES = ("metric", "example", "column")


def custom_entries(lake: Any, view=None) -> list[dict[str, Any]]:
    """湖里人工维护的目录条目：note 记录，agent = catalog，meta.catalog ∈ metric / example / column。"""
    try:
        rows, _ = lake.list(where=lake.where_clause(kinds=["note"], extra=f"agent = '{CATALOG_AGENT}'"), limit=500, view=view)
    except Exception:
        return []
    out = []
    for r in rows:
        meta = _json_obj(r.get("meta"))
        ctype = meta.get("catalog")
        if ctype not in CUSTOM_TYPES:
            continue
        out.append({"id": r["id"], "type": ctype, "name": r.get("title") or "", "content": r.get("content") or "", "meta": meta, "day": r.get("day"), "user_id": r.get("user_id")})
    return out


def custom_record(entry: dict[str, Any], *, user_id: str = "default") -> dict[str, Any]:
    """表单 → 湖记录。metric: name / definition / sql / unit；example: name（问题）/ sql；column: column / note。"""
    ctype = str(entry.get("type") or "")
    if ctype not in CUSTOM_TYPES:
        raise ValueError(f"目录条目类型只能是 {' / '.join(CUSTOM_TYPES)}")
    name = str(entry.get("name") or entry.get("column") or "").strip()
    if not name:
        raise ValueError("名称不能为空")
    meta: dict[str, Any] = {"catalog": ctype}
    if ctype == "column":
        note = str(entry.get("note") or "").strip()
        if not note:
            raise ValueError("列备注不能为空")
        meta["column"] = name
        content = note
    else:
        sql = str(entry.get("sql") or "").strip().rstrip(";")
        if not sql:
            raise ValueError("SQL 不能为空")
        meta["sql"] = sql
        if ctype == "metric":
            meta["definition"] = str(entry.get("definition") or "").strip() or name
            meta["unit"] = str(entry.get("unit") or "").strip()
            content = f"{name} = {meta['definition']}\n{sql}"
        else:
            content = f"{name}\n{sql}"
    return {"kind": "note", "agent": CATALOG_AGENT, "user_id": user_id, "title": name[:300], "content": content, "meta": meta}


def column_entries(df, columns: list[dict[str, str]], notes: dict[str, list[str]] | None = None, labels: dict[str, dict[str, str]] | None = None) -> list[dict[str, Any]]:
    """每一列一条：说明（+ 人工备注）+ 真实取值概况。labels：某列取值的可读名（kb_id → 知识库名），写在取值旁边，SQL 里仍用原值过滤。"""
    entries = []
    n = len(df)
    labels = labels or {}
    for c in columns:
        name, ctype = c["name"], c["type"]
        if name == "vector":
            continue
        desc_parts = [COLUMN_DOCS.get(name, "派生列"), *(notes or {}).get(name, [])]
        e: dict[str, Any] = {"kind": "column", "name": name, "type": ctype, "description": "；".join(desc_parts), "samples": [], "range": None, "nulls": None, "notes": (notes or {}).get(name, [])}
        if name in df.columns and n:
            col = df[name]
            nulls = int(col.isna().sum()) if hasattr(col, "isna") else 0
            e["nulls"] = nulls
            if ctype in ("string", "large_string") or col.dtype == object:
                vals = col.dropna().astype(str)
                vals = vals[vals != ""]
                if name in ("meta", "refs", "content", "title", "id", "parent_id", "session_id", "embed_model"):
                    e["samples"] = [_short(v, 60) for v in vals.head(2).tolist()]
                else:
                    counts = Counter(vals.tolist())
                    named = labels.get(name) or {}
                    show = (lambda v: f"{v}＝{named[v]}" if v in named else v)
                    if len(counts) <= _LOW_CARDINALITY:
                        e["samples"] = [f"{show(v)}（{k}）" for v, k in counts.most_common(_LOW_CARDINALITY)]
                    else:
                        e["samples"] = [show(v) for v, _ in counts.most_common(5)]
                        e["range"] = f"{len(counts)} 个不同取值"
                    if named and name in COLUMN_ALIASES:
                        e["keywords"] = list(named.values())      # 问题里提到知识库名也能命中这一列（下面会再并入列名和别名）
            else:
                try:
                    num = col.dropna().astype(float)
                    if len(num):
                        lo, hi = float(num.min()), float(num.max())
                        if name == "ts":
                            e["range"] = f"{time.strftime('%Y-%m-%d', time.localtime(lo))} ~ {time.strftime('%Y-%m-%d', time.localtime(hi))}"
                        else:
                            e["range"] = f"{round(lo, 4)} ~ {round(hi, 4)}"
                except (TypeError, ValueError):
                    pass
        desc = e["description"]
        extra = ("；取值 " + " / ".join(e["samples"][:8])) if e["samples"] else ""
        extra += f"；范围 {e['range']}" if e["range"] else ""
        extra += f"；空值 {e['nulls']} 条" if e["nulls"] else ""
        e["doc"] = f"{name} ({ctype}): {desc}{extra}"
        e["text"] = f"列 {name} {desc} {' '.join(e['samples'][:8])}"
        e["keywords"] = [name, name.replace("_id", ""), *COLUMN_ALIASES.get(name, []), *e.get("keywords", [])]
        entries.append(e)
    return entries


def meta_entries(df) -> list[dict[str, Any]]:
    """meta JSON 里按 kind 出现的键：路径、示例值、SQL 写法。"""
    if not len(df) or "meta" not in df.columns:
        return []
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for kind, group in df.groupby("kind"):
        for raw in group["meta"].head(_META_SAMPLE_ROWS).tolist():
            for path, value in _meta_paths(_json_obj(raw)):
                key = (str(kind), path)
                item = found.setdefault(key, {"count": 0, "samples": []})
                item["count"] += 1
                s = _short(value)
                if s not in item["samples"] and len(item["samples"]) < 3 and s not in ("", "null", "{}", "[]"):
                    item["samples"].append(s)
    entries = []
    for (kind, path), item in sorted(found.items()):
        leaf = path.endswith("[*]") or "." in path.rsplit("[*]", 1)[-1] or "[*]" not in path
        access = f"json_extract_string(meta, '{path}')" if leaf else f"json_extract(meta, '{path}')"
        if path.endswith("[*]"):
            access = f"json_extract(meta, '{path[:-3]}')（列表）"
        elif "[*]." in path:
            access = f"unnest(json_extract_string(meta, '{path}'))（展开列表里的字段；要聚合时先在子查询里 unnest 再 GROUP BY）"
        samples = "、".join(item["samples"]) if item["samples"] else "—"
        leaf_name = re.split(r"[.\[]", path.rstrip("]"))[-1].strip("*.") or path
        entries.append({"kind": "meta", "name": path, "recordKind": kind, "count": item["count"], "samples": item["samples"], "access": access,
                        "doc": f"{kind} 的 meta {path}：示例 {samples}；SQL 用 {access}",
                        "text": f"{kind} meta {path} {' '.join(item['samples'])}", "keywords": [leaf_name, *_terms(leaf_name), *item["samples"][:3]]})
    return entries


def metric_entries(low_score: float, custom: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    out = []
    for m in METRICS:
        definition, sql = m["definition"].format(low_score=low_score), m["sql"].format(low_score=low_score)
        out.append({"kind": "metric", "name": m["name"], "unit": m["unit"], "definition": definition, "sql": sql, "source": "builtin",
                    "doc": f"{m['name']}{'（' + m['unit'] + '）' if m['unit'] else ''} = {definition}\n  SQL：{sql}", "text": f"指标 {m['name']} {definition}",
                    "keywords": [m["name"], *_terms(m["name"]), *m.get("aliases", "").split()]})
    for c in custom or []:
        if c["type"] != "metric":
            continue
        meta, name = c["meta"], c["name"]
        definition, unit, sql = str(meta.get("definition") or name), str(meta.get("unit") or ""), str(meta.get("sql") or "")
        out.append({"kind": "metric", "name": name, "unit": unit, "definition": definition, "sql": sql, "source": "custom", "id": c["id"],
                    "doc": f"{name}{'（' + unit + '）' if unit else ''} = {definition}\n  SQL：{sql}", "text": f"指标 {name} {definition}",
                    "keywords": [name, *_terms(name), *_terms(definition)]})
    return out


def example_entries(extra: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    out = []
    for ex in list(EXAMPLES) + list(extra or []):
        out.append({"kind": "example", "name": ex["question"], "sql": ex["sql"], "source": ex.get("source", "builtin"), **({"id": ex["id"]} if ex.get("id") else {}),
                    "doc": f"{ex['question']} → {ex['sql']}", "text": f"示例 {ex['question']}", "keywords": _terms(ex["question"])})
    return out


def custom_examples(custom: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"question": c["name"], "sql": str(c["meta"].get("sql") or ""), "source": "custom", "id": c["id"]} for c in custom if c["type"] == "example"]


def column_notes(custom: list[dict[str, Any]]) -> dict[str, list[str]]:
    notes: dict[str, list[str]] = {}
    for c in custom:
        if c["type"] == "column" and c["meta"].get("column"):
            notes.setdefault(str(c["meta"]["column"]), []).append(c["content"])
    return notes


def learned_examples(lake: Any, question: str, *, top_k: int = 3, view=None) -> list[dict[str, str]]:
    """湖里以前分析成功的 SQL：按问题相似度找 analysis 记录，取它名下执行成功（status ok）的 sql 工具调用。"""
    if not question.strip():
        return []
    try:
        hits = lake.search(question, top_k=max(top_k * 2, 4), where=lake.where_clause(kinds=["analysis"]), view=view)
    except Exception:
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for h in hits:
        calls, _ = lake.list(where=lake.where_clause(kinds=["tool_call"], parent_ids=[h["id"]]), limit=20, view=view)
        for c in calls:
            if c.get("title") != "sql" or c.get("status") != "ok":
                continue
            sql = str((_json_obj(c.get("meta")).get("args") or {}).get("query") or "").strip()
            if not sql or sql in seen:
                continue
            seen.add(sql)
            out.append({"question": str(h.get("title") or h.get("content", "")).split("\n")[0][:80], "sql": sql, "source": "history"})
            if len(out) >= top_k:
                return out
    return out


class LakeCatalog:
    """一份快照的目录：构建一次，按意图检索。"""

    def __init__(self, entries: list[dict[str, Any]], embedder: Any | None = None) -> None:
        self.entries = entries
        self.embedder = embedder
        self.custom: list[dict[str, Any]] = []
        self._vectors: np.ndarray | None = None

    @classmethod
    def build(cls, df, columns: list[dict[str, str]], *, low_score: float = 0.05, embedder: Any | None = None, custom: list[dict[str, Any]] | None = None,
              labels: dict[str, dict[str, str]] | None = None) -> "LakeCatalog":
        custom = custom or []
        entries = (column_entries(df, columns, column_notes(custom), labels) + meta_entries(df) + metric_entries(low_score, custom)
                   + example_entries(custom_examples(custom)))
        cat = cls(entries, embedder)
        cat.custom = custom
        return cat

    def _embed(self) -> np.ndarray | None:
        if self.embedder is None:
            return None
        if self._vectors is None:
            vecs = np.asarray(self.embedder.embed_documents([e["text"] for e in self.entries]), dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            self._vectors = vecs / np.where(norms == 0, 1, norms)
        return self._vectors

    def discover(self, intent: str, *, per_kind: dict[str, int] | None = None, extra_examples: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """按意图给每类目录项打分：向量余弦 + 关键词命中（列名 / 指标名 / 键名出现在问题里加分）。空意图返回每类的前几条。"""
        limits = {"column": 8, "meta": 6, "metric": 3, "example": 3} | (per_kind or {})
        entries = self.entries + example_entries(extra_examples)[len(EXAMPLES):] if extra_examples else self.entries
        scores = np.zeros(len(entries), dtype=np.float32)
        hits = np.zeros(len(entries), dtype=np.float32)
        if intent.strip():
            vecs = self._embed()
            if vecs is not None and len(vecs) == len(self.entries):
                q = np.asarray(self.embedder.embed_query(intent), dtype=np.float32)
                q = q / (np.linalg.norm(q) or 1.0)
                scores[: len(self.entries)] = vecs @ q
            for i, e in enumerate(entries):
                if _keyword_hit(e, intent):
                    hits[i] = 1.0             # 关键词命中排在前面：向量相似度只在命中的条目之间、以及没命中的条目之间排序
                if e.get("source") == "history":
                    scores[i] += 0.1          # 本库自己的成功查询优先于内置示例
                elif e.get("source") == "custom":
                    scores[i] += 0.2          # 人工维护的口径 / 示例最优先
        groups: dict[str, list[dict[str, Any]]] = {"column": [], "meta": [], "metric": [], "example": []}
        order = np.lexsort((-scores, -hits)) if intent.strip() else np.arange(len(entries))
        for i in order:
            e = entries[int(i)]
            bucket = groups[e["kind"]]
            if len(bucket) < limits.get(e["kind"], 0):
                bucket.append({k: v for k, v in e.items() if k not in ("text", "keywords")} | {"relevance": round(float(scores[int(i)]) + float(hits[int(i)]), 4)})
        # 列：始终带上写 SQL 必需的骨架列，哪怕问题里没提到
        must = ("kind", "parent_id")
        names = {c["name"] for c in groups["column"]}
        for e in self.entries:
            if e["kind"] == "column" and e["name"] in must and e["name"] not in names:
                groups["column"].append({k: v for k, v in e.items() if k not in ("text", "keywords")} | {"relevance": 0.0})
        return {"intent": intent, "columns": groups["column"], "metaKeys": groups["meta"], "metrics": groups["metric"], "examples": groups["example"],
                "totals": dict(Counter(e["kind"] for e in self.entries)), "custom": [{k: v for k, v in c.items() if k != "meta"} | {"meta": c["meta"]} for c in self.custom]}

    @staticmethod
    def prompt_doc(found: dict[str, Any]) -> str:
        """discover 的结果 → 提示词片段。"""
        parts = ["与问题相关的列（lake 表，其余列用 catalog 工具查）："] + [f"- {c['doc']}" for c in found["columns"]]
        if found["metaKeys"]:
            parts += ["meta JSON 里相关的键：", *[f"- {m['doc']}" for m in found["metaKeys"]]]
        if found["metrics"]:
            parts += ["指标口径（直接用这里的 SQL，不要自己另起定义；引用时写明口径和样本数）：", *[f"- {m['doc']}" for m in found["metrics"]]]
        if found["examples"]:
            parts += ["问题 → SQL 示例：", *[f"- {e['doc']}" for e in found["examples"]]]
        return "\n".join(parts)


def check_filter(where: str, columns: list[str]) -> str:
    """检索工具里由模型生成的结构化过滤条件（SQL WHERE 片段）：只允许比较 / 逻辑 / LIKE / IN / BETWEEN / IS NULL，
    标识符必须是 lake 的列；不允许子查询、函数调用以外的语句成分。"""
    where = (where or "").strip().strip(";").strip()
    if not where:
        return ""
    bare = re.sub(r"'(?:[^']|'')*'", " ", where)
    if ";" in bare or "--" in bare or "/*" in bare:
        raise ValueError("过滤条件里不能有分号或注释")
    if re.search(r"\b(select|union|insert|update|delete|drop|attach|pragma|create|alter|exec|call)\b", bare, re.I):
        raise ValueError("过滤条件只能是 WHERE 片段，不能包含子查询或语句")
    if re.search(r"\w\s*\(", bare):
        raise ValueError("过滤条件里不允许函数调用；时间范围用 days 参数")
    allowed = {"and", "or", "not", "in", "like", "ilike", "is", "null", "between", "true", "false"}
    cols = set(columns)
    for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", bare):
        low = ident.lower()
        if low in allowed or low in cols:
            continue
        if re.fullmatch(r"[0-9]+", ident):
            continue
        raise ValueError(f"过滤条件里的 {ident} 不是 lake 的列")
    return where


def is_empty_result(tool: str, result: Any) -> bool:
    """工具返回了"没有数据"：反思时据此提醒模型别靠改引号 / 大小写重试，而是换思路或直接说没有。"""
    if result is None:
        return True
    if tool == "sql":
        return not (result or {}).get("rows")
    if tool == "search":
        return not result
    if tool in ("knowledge_gaps",):
        return not (result or {}).get("gaps")
    return False


def rowcount(result: Any) -> int | None:
    if isinstance(result, dict) and "rowCount" in result:
        return int(result["rowCount"])
    if isinstance(result, list):
        return len(result)
    return None


__all__ = ["CATALOG_AGENT", "CUSTOM_TYPES", "EXAMPLES", "METRICS", "LakeCatalog", "check_filter", "column_entries", "custom_entries", "custom_record", "example_entries", "is_empty_result", "learned_examples", "meta_entries", "metric_entries", "rowcount"]
