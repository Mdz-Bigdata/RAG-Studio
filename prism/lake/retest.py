"""缺口复测：湖里记录的是历史——补了语料之后，旧的缺口仍然留在统计里。复测把缺口问题（可选：当时检索分偏低的问题）
拿到"现在的"知识库重新检索（默认再生成一次回答看是否拒答），结果作为 retest 记录写回湖里；
检索质量统计据此把缺口分成"仍未解决"和"复测已解决"。

检索 / 回答由调用方注入（retest_fn），核心库不依赖 Web 服务：
    retest_fn(question, kb_id, check_answer) -> {"score": float | None, "contexts": [{docId, fileName, score}], "refused": bool | None, "answer": str | None,
                                                 "altKb": {kbId, name, score, fileName, answer} | None}
本库仍答不上、但 altKb（别的知识库）能答时记为 wrong_kb：是提问时选错了库 / 路由问题，不是缺语料。
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import analytics
from .store import AgentLake

logger = logging.getLogger(__name__)

RetestFn = Callable[[str, str, bool], dict[str, Any]]
STATUS_LABELS = {"resolved": "已解决", "improved": "有改善", "answerable": "能回答（检索分偏低）", "still_gap": "仍是缺口", "wrong_kb": "答案在其它知识库", "error": "复测失败"}


def _key(kb_id: str, question: str) -> tuple[str, str]:
    return (kb_id or "", analytics._norm_question(question))


def pick_targets(df, *, low_score: float, include_weak: bool = False, weak_score: float = 0.3, limit: int = 10) -> list[dict[str, Any]]:
    """要复测的问题：仍未解决的缺口（新的在前）+ 可选的"答了但检索分偏低"的问题；同一知识库里同一个问题只测一次。"""
    q = analytics.retrieval_quality(df, low_score=low_score, gap_limit=100_000)
    targets: dict[tuple[str, str], dict[str, Any]] = {}
    for g in q["gaps"]:
        k = _key(g["kbId"], g["question"])
        if k[1] and g["kbId"] and k not in targets:
            targets[k] = {"questionId": g["questionId"], "question": g["question"], "kbId": g["kbId"], "oldScore": g["score"], "reasons": g["reasons"], "type": "gap"}
    if include_weak:
        r = df[df["kind"] == "retrieval"]
        qs = df[df["kind"] == "question"].set_index("id")
        weak = []
        for row in r.itertuples():
            if row.score is None or (isinstance(row.score, float) and math.isnan(row.score)) or row.score >= weak_score or row.parent_id not in qs.index:
                continue
            qrow = qs.loc[row.parent_id]
            weak.append((row.ts, row.parent_id, qrow["content"], qrow["kb_id"], float(row.score)))
        for ts, qid, text, kb, score in sorted(weak, reverse=True):
            k = _key(kb, text)
            if k[1] and kb and k not in targets:
                targets[k] = {"questionId": qid, "question": text, "kbId": kb, "oldScore": round(score, 4), "reasons": [f"检索分偏低（{score:.3f}）"], "type": "weak"}
    return list(targets.values())[:limit]


def classify(target: dict[str, Any], result: dict[str, Any], *, low_score: float, weak_score: float) -> str:
    if result.get("error"):
        return "error"
    score, refused = result.get("score"), result.get("refused")
    if not result.get("contexts") or score is None or score < low_score or refused:
        return "wrong_kb" if result.get("altKb") else "still_gap"
    if target["type"] == "weak" and score < weak_score:
        old = target.get("oldScore") or 0.0
        if score > old + 0.05:
            return "improved"
        # 分数仍低、但检索到了资料且回答不拒答：多半是问法笼统（"图2 说明了什么"），重排分天然偏低——能答，不算缺口。
        # 没生成回答（check_answer=False，refused 为 None）时判断不了，仍按缺口处理。
        return "answerable" if refused is False else "still_gap"
    return "resolved"


def run_retest(lake: AgentLake, df, retest_fn: RetestFn, *, low_score: float = 0.05, include_weak: bool = False, weak_score: float = 0.3,
               check_answer: bool = True, limit: int = 10, workers: int = 3, record: bool = True, agent: str = "lake-retest") -> dict[str, Any]:
    targets = pick_targets(df, low_score=low_score, include_weak=include_weak, weak_score=weak_score, limit=limit)
    t0 = time.perf_counter()

    def one(t: dict[str, Any]) -> dict[str, Any]:
        t1 = time.perf_counter()
        try:
            res = retest_fn(t["question"], t["kbId"], check_answer) or {}
        except Exception as exc:          # 单条失败不影响其余
            logger.warning("缺口复测失败：%s（%s）", t["question"], exc)
            res = {"error": str(exc)}
        status = classify(t, res, low_score=low_score, weak_score=weak_score)
        return {**t, "status": status, "statusLabel": STATUS_LABELS[status], "newScore": None if res.get("score") is None else round(float(res["score"]), 4),
                "refused": res.get("refused"), "answer": (res.get("answer") or "")[:400], "contexts": (res.get("contexts") or [])[:3], "altKb": res.get("altKb"),
                "error": res.get("error"), "ms": round((time.perf_counter() - t1) * 1000, 1)}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(one, targets))
    if record and results:
        now = time.time()
        lake.append([{
            "kind": "retest", "agent": agent, "kb_id": r["kbId"], "parent_id": r["questionId"], "title": r["question"],
            "content": f"复测：{r['question']}\n结果：{r['statusLabel']}（检索分 {r['oldScore'] if r['oldScore'] is not None else '—'} → {r['newScore'] if r['newScore'] is not None else '—'}"
                       f"{'，回答拒答' if r['refused'] else ''}）" + (f"\n回答：{r['answer'][:200]}" if r["answer"] else "")
                       + (f"\n其它知识库能回答：{r['altKb']['name']}（检索分 {r['altKb']['score']}）" if r["altKb"] else ""),
            "status": r["status"], "score": r["newScore"], "latency_ms": r["ms"], "ts": now,
            "meta": {"oldScore": r["oldScore"], "reasons": r["reasons"], "type": r["type"], "refused": r["refused"], "checkAnswer": check_answer, "altKb": r["altKb"],
                     "contexts": r["contexts"], "error": r["error"]},
        } for r in results])
    counts = {s: sum(1 for r in results if r["status"] == s) for s in STATUS_LABELS}
    return {"tested": len(results), "counts": counts, "results": results, "elapsedMs": round((time.perf_counter() - t0) * 1000, 1), "checkAnswer": check_answer}
