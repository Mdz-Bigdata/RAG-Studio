"""Agent 湖分析：概况、检索质量与知识缺口、问题主题聚类、只读 SQL（DuckDB 直接查 Arrow）。

全部基于 agent_lake 的同一份数据（某个版本 / Tag / 分支的快照），不另建数仓。
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from typing import Any

import numpy as np

_STOP = set("的 了 是 吗 呢 啊 吧 么 什么 怎么 怎样 如何 哪些 哪个 多少 几 可以 能 能否 是否 一下 请问 请 我 我们 你 你们 他 她 它 和 与 及 或 在 有 没有 要 需要 会 对 把 被 给 从 到 为 为什么 吗？ 这个 那个 这些 那些 一个 还是 以及 如果 应该 怎么办 有没有".split())
_REFUSAL = re.compile(r"无法回答|未检索到相关|资料中没有|没有找到相关|根据现有资料无法")


def frame(arrow):
    """Arrow → pandas（不含向量列）。"""
    return arrow.to_pandas()


def _pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


def _json_list(text: Any) -> list[Any]:
    try:
        v = json.loads(text or "[]")
        return v if isinstance(v, list) else []
    except (TypeError, ValueError):
        return []


def _norm_question(text: Any) -> str:
    """问题去空白 / 标点后的比较键：同一个问题问了几次、或复测记录里的问题，都归到一起。"""
    return re.sub(r"[\s，。？！?!,.、：:；;\"'“”‘’（）()]+", "", str(text or "")).lower()


def _json_obj(text: Any) -> dict[str, Any]:
    try:
        v = json.loads(text or "{}")
        return v if isinstance(v, dict) else {}
    except (TypeError, ValueError):
        return {}


# ---------------------------------------------------------------------- 概况
def overview(df, days: int = 30) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0, "kinds": {}, "sessions": 0, "users": 0, "agents": 0, "daily": [], "firstTs": None, "lastTs": None}
    kinds = df["kind"].value_counts().to_dict()
    cutoff = time.time() - days * 86400
    recent = df[df["ts"] >= cutoff]
    daily_tbl = recent.groupby(["day", "kind"]).size().unstack(fill_value=0) if not recent.empty else None
    daily = []
    if daily_tbl is not None:
        for day, row in daily_tbl.sort_index().iterrows():
            daily.append({"day": day, **{k: int(v) for k, v in row.items()}})
    sessions = df.loc[df["session_id"] != "", "session_id"].nunique()
    return {
        "rows": int(len(df)), "kinds": {k: int(v) for k, v in kinds.items()},
        "sessions": int(sessions), "users": int(df["user_id"].nunique()), "agents": int(df["agent"].nunique()),
        "daily": daily, "firstTs": float(df["ts"].min()), "lastTs": float(df["ts"].max()),
    }


# ---------------------------------------------------------------------- 检索质量 / 知识缺口
def retrieval_quality(df, *, low_score: float = 0.05, days: int | None = None, gap_limit: int = 50) -> dict[str, Any]:
    """问答链路体检：检索延迟与置信度、拒答率、反馈；找出"没答上 / 答得没把握 / 被点踩"的问题（知识缺口）。
    缺口会结合复测结果（retest 记录）分成"仍未解决"（gaps）、"复测已解决"（resolvedGaps）和"答案在其它知识库"（misroutedGaps，选错库 / 路由问题），
    并列出缺口之后同一知识库里变更过的文档。"""
    if days:
        df = df[df["ts"] >= time.time() - days * 86400]
    q = df[df["kind"] == "question"]
    r = df[df["kind"] == "retrieval"]
    a = df[df["kind"] == "answer"]
    lat = [float(x) for x in r["latency_ms"].dropna().tolist()]
    scores = [float(x) for x in r["score"].dropna().tolist()]
    r_by_q = {row.parent_id: row for row in r.itertuples()}
    a_by_q = {row.parent_id: row for row in a.itertuples()}

    gaps: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []        # 用户标记「忽略」的问题（status=ignored）：不算缺口，单独列出便于恢复
    for row in q.itertuples():
        rr, aa = r_by_q.get(row.id), a_by_q.get(row.id)
        reasons = []
        if rr is not None and rr.status == "no_context":
            reasons.append("未检索到资料")
        elif rr is not None and rr.score is not None and not math.isnan(rr.score) and rr.score < low_score:
            reasons.append(f"检索置信度低（{rr.score:.3f}）")
        if aa is not None and (aa.status == "refused" or _REFUSAL.search(aa.content or "")):
            reasons.append("回答拒答")
        if aa is not None and aa.feedback < 0:
            reasons.append("用户点踩")
        if aa is not None and aa.status == "error":
            reasons.append("生成失败")
        if reasons:
            (ignored if row.status == "ignored" else gaps).append({"questionId": row.id, "question": row.content, "reasons": reasons, "ts": float(row.ts), "sessionId": row.session_id,
                         "kbId": row.kb_id, "score": None if rr is None or rr.score is None or math.isnan(rr.score) else round(float(rr.score), 4),
                         "answer": (aa.content[:200] if aa is not None else None), "answerId": aa.id if aa is not None else None})
    gaps.sort(key=lambda g: -g["ts"])
    ignored.sort(key=lambda g: -g["ts"])

    # 复测：同一知识库里同一个问题的最新一次复测晚于缺口发生时间，就以复测结果为准——"已解决"的移出未解决列表
    latest: dict[tuple[str, str], Any] = {}
    for row in df[df["kind"] == "retest"].itertuples():
        k = (row.kb_id, _norm_question(row.title))
        if k not in latest or row.ts > latest[k].ts:
            latest[k] = row
    doc_events = [(row.kb_id, float(row.ts), row.title) for row in df[df["kind"] == "doc_event"].itertuples()]
    open_gaps, resolved_gaps, misrouted_gaps = [], [], []
    for g in gaps:
        g["docChangesAfter"] = list(dict.fromkeys(t for kb, ts, t in doc_events if kb == g["kbId"] and ts > g["ts"]))[:5]
        rt = latest.get((g["kbId"], _norm_question(g["question"])))
        if rt is not None and rt.ts > g["ts"]:
            g["retest"] = {"status": rt.status, "score": None if rt.score is None or math.isnan(rt.score) else round(float(rt.score), 4), "ts": float(rt.ts),
                           "altKb": _json_obj(rt.meta).get("altKb")}
            if rt.status == "resolved":
                resolved_gaps.append(g)
                continue
            if rt.status == "wrong_kb":           # 别的知识库能答：选错库 / 路由问题，不算缺语料
                misrouted_gaps.append(g)
                continue
        open_gaps.append(g)

    docs: Counter[str] = Counter()
    doc_names: dict[str, str] = {}
    for meta in r["meta"].tolist():
        for c in _json_obj(meta).get("contexts", [])[:5]:
            key = c.get("docId") or c.get("fileName") or ""
            if key:
                docs[key] += 1
                doc_names[key] = c.get("fileName") or key
    cited: Counter[str] = Counter()
    for refs in a["refs"].tolist():
        for d in _json_list(refs):
            cited[str(d)] += 1

    strategies: dict[str, dict[str, Any]] = {}
    for row in r.itertuples():
        s = _json_obj(row.meta).get("strategy") or "—"
        st = strategies.setdefault(s, {"strategy": s, "count": 0, "scores": [], "latency": []})
        st["count"] += 1
        if row.score is not None and not math.isnan(row.score):
            st["scores"].append(row.score)
        if row.latency_ms is not None and not math.isnan(row.latency_ms):
            st["latency"].append(row.latency_ms)
    by_strategy = [{"strategy": s["strategy"], "count": s["count"], "avgScore": round(float(np.mean(s["scores"])), 4) if s["scores"] else None,
                    "p50LatencyMs": _pct(s["latency"], 50)} for s in sorted(strategies.values(), key=lambda s: -s["count"])]

    n_q, n_a = len(q), len(a)
    refused = sum(1 for row in a.itertuples() if row.status == "refused" or _REFUSAL.search(row.content or ""))
    up, down = int((a["feedback"] > 0).sum()), int((a["feedback"] < 0).sum())
    buckets = [("< 0.01", 0, 0.01), ("0.01~0.05", 0.01, 0.05), ("0.05~0.2", 0.05, 0.2), ("0.2~0.5", 0.2, 0.5), ("≥ 0.5", 0.5, float("inf"))]
    return {
        "questions": n_q, "answers": n_a, "retrievals": len(r),
        "latency": {"avg": round(float(np.mean(lat)), 1) if lat else None, "p50": _pct(lat, 50), "p95": _pct(lat, 95)},
        "score": {"avg": round(float(np.mean(scores)), 4) if scores else None, "lowCount": sum(1 for s in scores if s < low_score), "lowThreshold": low_score,
                  "buckets": [{"label": lbl, "count": sum(1 for s in scores if lo <= s < hi)} for lbl, lo, hi in buckets]},
        "refused": refused, "refusalRate": round(refused / n_a, 4) if n_a else None,
        "feedback": {"up": up, "down": down},
        "gapCount": len(open_gaps), "gapRate": round(len(open_gaps) / n_q, 4) if n_q else None, "gaps": open_gaps[:gap_limit],
        "historicalGapCount": len(gaps), "resolvedCount": len(resolved_gaps), "resolvedGaps": resolved_gaps[:gap_limit],
        "misroutedCount": len(misrouted_gaps), "misroutedGaps": misrouted_gaps[:gap_limit],
        "ignoredCount": len(ignored), "ignoredGaps": ignored[:gap_limit],
        "lastRetestTs": max((float(r.ts) for r in latest.values()), default=None),
        "topDocs": [{"docId": k, "fileName": doc_names.get(k, k), "retrieved": n, "cited": cited.get(k, 0)} for k, n in docs.most_common(10)],
        "byStrategy": by_strategy,
    }


# ---------------------------------------------------------------------- 路由
def routing(df, *, kb_names: dict[str, str] | None = None, low_score: float = 0.05) -> dict[str, Any]:
    """路由情况：自动路由兜底触发了几次（从哪个库换到哪个库）；"答案在其它知识库"的缺口之后再问时有没有被答对（已修复 / 仍在发生）；
    各知识库的提问量。kb_names：{知识库 id: 名称}，由调用方提供（核心库不读业务库）。"""
    names = kb_names or {}
    name = lambda kb: names.get(kb, kb or "（未指定）")   # noqa: E731
    r = df[df["kind"] == "retrieval"]
    a = df[df["kind"] == "answer"]
    q = df[df["kind"] == "question"]
    answer_by_q = {row.parent_id: row for row in a.itertuples()}
    fallbacks = []
    for row in r.sort_values("ts").itertuples():
        fb = _json_obj(row.meta).get("routeFallback")
        if isinstance(fb, dict) and fb:
            fallbacks.append({"question": row.title, "from": fb.get("fromName"), "to": fb.get("toName"), "fromScore": fb.get("fromScore"), "toScore": fb.get("toScore"),
                              "reason": fb.get("reason"), "time": time.strftime("%Y-%m-%d %H:%M", time.localtime(row.ts))})
    # 问题 → 每次提问的结果（知识库、是否有资料、是否兜底、回答状态），按问题文本归并
    asks: dict[str, list[dict[str, Any]]] = {}
    for row in r.itertuples():
        ans = answer_by_q.get(row.parent_id)
        fb = _json_obj(row.meta).get("routeFallback")
        asks.setdefault(_norm_question(row.title), []).append({"ts": float(row.ts), "kb": row.kb_id, "hasContext": row.status != "no_context",
                                                              "fallback": bool(isinstance(fb, dict) and fb), "answered": ans is not None and ans.status == "ok"})
    rq = retrieval_quality(df, low_score=low_score, gap_limit=10_000)
    misrouted: dict[str, dict[str, Any]] = {}
    for g in rq["misroutedGaps"]:
        key = _norm_question(g["question"])
        m = misrouted.setdefault(key, {"question": g["question"], "askedIn": set(), "answerInKb": ((g.get("retest") or {}).get("altKb") or {}).get("name"), "times": 0, "lastTs": 0.0})
        m["askedIn"].add(name(g["kbId"]))
        m["times"] += 1
        m["lastTs"] = max(m["lastTs"], g["ts"])
    out_mis = []
    for key, m in misrouted.items():
        later = [x for x in asks.get(key, []) if x["ts"] > m["lastTs"] and x["answered"] and x["hasContext"]]
        status = ("之后再问已答对" + ("（靠路由兜底）" if any(x["fallback"] for x in later) else "")) if later else "之后没有再被正确回答过"
        out_mis.append({"question": m["question"], "askedIn": sorted(m["askedIn"]), "answerInKb": m["answerInKb"], "times": m["times"],
                        "lastWrongAsk": time.strftime("%Y-%m-%d %H:%M", time.localtime(m["lastTs"])), "status": status})
    by_kb = q.groupby("kb_id").size().to_dict() if not q.empty else {}
    return {"fallbackCount": len(fallbacks), "fallbacks": fallbacks[-20:], "misroutedCount": rq["misroutedCount"], "misrouted": out_mis,
            "questionsByKb": {name(k): int(v) for k, v in sorted(by_kb.items(), key=lambda kv: -kv[1])}}


# ---------------------------------------------------------------------- 主题聚类
def _tokenize(text: str) -> list[str]:
    try:
        import jieba

        jieba.setLogLevel(60)
        toks = jieba.lcut(text)
    except Exception:
        toks = re.findall(r"[A-Za-z0-9_.-]+|[一-鿿]{2,}", text)
    return [t.strip().lower() for t in toks if len(t.strip()) >= 2 and t.strip().lower() not in _STOP and not re.fullmatch(r"[\W_]+", t.strip())]


def _kmeans(x: np.ndarray, k: int, iters: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """余弦 k-means（k-means++ 初始化，固定种子 → 结果可复现）。"""
    rng = np.random.default_rng(seed)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = 1.0 - np.max(x @ np.stack(centers).T, axis=1)
        d = np.clip(d, 0, None) ** 2
        centers.append(x[rng.choice(len(x), p=d / d.sum())] if d.sum() > 0 else x[rng.integers(len(x))])
    c = np.stack(centers)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        new = np.argmax(x @ c.T, axis=1)
        if _ and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = x[labels == j]
            if len(m):
                v = m.mean(axis=0)
                c[j] = v / (np.linalg.norm(v) + 1e-9)
    return labels, c


def topics(df_vec, *, k: int | None = None, gaps: set[str] | None = None, scores: dict[str, float] | None = None, with_ids: bool = False) -> dict[str, Any]:
    """把问题按语义聚成 k 个主题：主题词（c-TF-IDF）、规模、缺口率、平均检索分、代表问题。
    with_ids：每个主题带上全部问题 id（questionIds，页面上"删除该主题下的问题"用；给智能体的结果不带，免得撑爆上下文）。"""
    q = df_vec[df_vec["kind"] == "question"]
    q = q[q["content"].str.strip() != ""]
    n = len(q)
    if n < 2:
        return {"k": 0, "questions": n, "clusters": [], "assignments": {}}
    k = int(k or max(2, min(8, round(math.sqrt(n / 2)))))
    k = max(1, min(k, n))
    x = np.stack([np.asarray(v, dtype=np.float32) for v in q["vector"].tolist()])
    labels, centers = _kmeans(x, k)
    texts = q["content"].tolist()
    ids = q["id"].tolist()
    toks = [_tokenize(t) for t in texts]
    df_count: Counter[str] = Counter()
    for j in range(k):
        df_count.update({t for i, ts in enumerate(toks) if labels[i] == j for t in ts})
    clusters = []
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    gaps = gaps or set()
    scores = scores or {}
    for j in range(k):
        idx = [i for i in range(n) if labels[i] == j]
        if not idx:
            continue
        tf: Counter[str] = Counter(t for i in idx for t in toks[i])
        weighted = {t: c * math.log(1 + k / df_count[t]) for t, c in tf.items()}
        kw = [t for t, _ in sorted(weighted.items(), key=lambda kv: -kv[1])[:5]]
        sims = xn[idx] @ centers[j]
        order = [idx[i] for i in np.argsort(-sims)]
        cl_scores = [scores[ids[i]] for i in idx if ids[i] in scores]
        clusters.append({
            "label": " / ".join(kw[:3]) or texts[order[0]][:16], "keywords": kw, "size": len(idx),
            "gapRate": round(sum(1 for i in idx if ids[i] in gaps) / len(idx), 3),
            "avgScore": round(float(np.mean(cl_scores)), 4) if cl_scores else None,
            "samples": [{"id": ids[i], "question": texts[i]} for i in order[:4]],
            "ids": [ids[i] for i in idx],
        })
    clusters.sort(key=lambda c: -c["size"])
    assignments = {i: c["label"] for c in clusters for i in c["ids"]}
    for c in clusters:
        ids_ = c.pop("ids")
        if with_ids:
            c["questionIds"] = ids_
    return {"k": len(clusters), "questions": n, "clusters": clusters, "assignments": assignments}


# ---------------------------------------------------------------------- SQL
_FORBIDDEN = re.compile(r"\b(attach|detach|copy|export|import|install|load|pragma|create|insert|update|delete|drop|alter|call|set|reset|checkpoint|vacuum|use)\b", re.I)


def _strip_literals(sql: str) -> str:
    return re.sub(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S)


def check_sql(sql: str) -> str:
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        raise ValueError("SQL 为空")
    bare = _strip_literals(sql)
    if ";" in bare:
        raise ValueError("只允许一条语句")
    if not re.match(r"^\s*(select|with)\b", bare, re.I):
        raise ValueError("只允许 SELECT / WITH 查询")
    m = _FORBIDDEN.search(bare)
    if m:
        raise ValueError(f"只读分析不允许使用 {m.group(1).upper()}")
    return sql


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, float):
        return None if math.isnan(v) or math.isinf(v) else round(v, 6)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return _jsonable(float(v))
    return str(v)


def run_sql(tables: dict[str, Any], sql: str, *, limit: int = 500) -> dict[str, Any]:
    """在 DuckDB 里对 Arrow 表执行只读 SQL：禁用外部文件访问（read_csv / attach 等都读不到本机文件），结果截断到 limit 行。"""
    import duckdb

    sql = check_sql(sql)
    con = duckdb.connect(config={"enable_external_access": False, "lock_configuration": True})
    try:
        for name, arrow in tables.items():
            con.register(name, arrow)
        t0 = time.perf_counter()
        cur = con.execute(f"SELECT * FROM ({sql}) AS q LIMIT {int(limit) + 1}")
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        ms = (time.perf_counter() - t0) * 1000
    finally:
        con.close()
    truncated = len(rows) > limit
    return {"columns": cols, "rows": [[_jsonable(v) for v in r] for r in rows[:limit]], "rowCount": min(len(rows), limit), "truncated": truncated, "elapsedMs": round(ms, 1)}


# 列说明：schema_doc（整表）和 catalog（按意图挑列）共用
COLUMN_DOCS = {
    "id": "主键", "kind": "类型：question 问题 / retrieval 检索日志 / answer 回答 / analysis 湖分析 / tool_call 工具调用 / memory 长期记忆 / board 智能体黑板 / document 文档 / doc_event 文档变更 / retest 复测 / note 笔记",
    "session_id": "会话 id", "user_id": "用户", "agent": "产生这条记录的智能体", "kb_id": "知识库 id（SQL 里按 id 过滤，取值旁 ＝ 后面是知识库名）", "title": "标题（retrieval / answer 是问题原文；tool_call 是工具名）", "content": "正文",
    "meta": "JSON 元数据（retrieval: strategy / queries / contexts[{docId,fileName,score}] / timingsMs；answer: model / citations；tool_call: args / why）",
    "refs": "JSON 数组：引用的文档 id（answer）或关联记录 id", "parent_id": "父记录：retrieval / answer / retest 的 parent_id 指向 question，tool_call 指向 analysis",
    "status": "ok / no_context（没检索到）/ refused（拒答）/ error", "score": "retrieval：最终上下文的最高分（重排分，无关 < 0.01）",
    "latency_ms": "耗时毫秒", "feedback": "回答反馈：1 赞 / -1 踩 / 0 无", "ts": "Unix 时间戳（秒）", "day": "日期 YYYY-MM-DD", "embed_model": "向量模型",
}


def schema_doc(columns: list[dict[str, str]]) -> str:
    """给分析智能体 / 前端看的表结构说明。"""
    return "\n".join(f"- {c['name']} ({c['type']}): {COLUMN_DOCS.get(c['name'], '派生列')}" for c in columns)
