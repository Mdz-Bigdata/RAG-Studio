#!/usr/bin/env python3
"""回答级评测：按线上问答的同一条链路（检索 → 图片 → 选模型 → 生成）回答评测集里的每个问题，再打分。

和「智能问答」的区别只有一点：不写会话、不写长期记忆、不写 Agent 湖（评测不能污染线上数据），
Agent 湖的历史问答参考默认也不带（它随线上使用不断变化，评测结果会不可复现；--experience 打开）。

用法：
    python scripts/answer_eval.py --set dev --judge                 # 调参时只看 dev
    python scripts/answer_eval.py --set test --judge                # 定稿后再看 test（不要拿 test 调参）
    python scripts/answer_eval.py --set dev --overrides '{"retrieval": {"rerank": {"min_score": 0.05}}}'
    python scripts/answer_eval.py --set dev --compare data/eval/answer_reports/<上一次>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prism.eval.answer_eval import JUDGE_SYSTEM, AnswerCase, build_judge_prompt, load_cases, parse_judge, score_case, summarize, to_markdown  # noqa: E402
from prism.generation import AnswerGenerator  # noqa: E402
from prism.generation.answer import parse_citations  # noqa: E402
from prism.generation.citecheck import check_citations  # noqa: E402
from prism.generation.prompts import format_context  # noqa: E402

SETS = {"dev": ROOT / "data/eval/answer_dev.jsonl", "test": ROOT / "data/eval/answer_test.jsonl"}
OUT = ROOT / "data/eval/answer_reports"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev", help="dev | test | all | 自定义 JSONL 路径")
    ap.add_argument("--judge", action="store_true", help="用 LLM 裁判逐条找无依据论断（慢一些）")
    ap.add_argument("--experience", action="store_true", help="带上 Agent 湖的历史问答参考（与线上一致，但不可复现）")
    ap.add_argument("--overrides", default=None, help="策略覆盖（JSON，结构同 config.yaml），只对本次评测生效")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--compare", type=Path, default=None, help="与之前的报告 JSON 对比")
    ap.add_argument("--only", default="", help="只跑问题里包含该子串的样例（调试用）")
    ap.add_argument("--rescore", type=Path, default=None, help="不调模型，用已保存报告里的回答按当前打分规则重算")
    args = ap.parse_args()
    if args.rescore:
        return rescore(args.rescore)

    logging.basicConfig(level=logging.WARNING)
    from app.db import Repository, SessionLocal
    from app.services.lake_service import lake_service
    from app.services.model_service import model_service
    from app.services.multimodal_service import multimodal_service
    from app.services.retrieval_service import retrieval_service
    from app.services.settings_service import settings_service

    if args.overrides:
        # 只在本进程内生效：替换缓存的设置对象，不写回 SQLite。只允许检索 / 生成 / 多模态参数——
        # 改入库参数（切片、向量模型等）会让索引服务判定索引过期并在后台整库重建
        over = json.loads(args.overrides)
        bad = set(over) - {"retrieval", "generation", "multimodal"}
        if bad:
            ap.error(f"--overrides 只允许 retrieval / generation / multimodal，不允许 {sorted(bad)}")
        settings_service._cached = settings_service.get().with_overrides(over)  # noqa: SLF001

    paths = list(SETS.values()) if args.set == "all" else [SETS.get(args.set) or Path(args.set)]
    cases: list[AnswerCase] = [c for p in paths for c in load_cases(p)]
    if args.only:
        cases = [c for c in cases if args.only in c.question]

    with SessionLocal() as db:
        kbs = {kb.name: kb.id for kb in Repository(db).list_kbs()}
    settings = settings_service.get()
    mm = settings.multimodal
    judge_llm = model_service.llm() if args.judge else None

    def run_one(case: AnswerCase) -> dict[str, Any]:
        kb_id = kbs.get(case.kb, case.kb)
        t0 = time.perf_counter()
        result = retrieval_service.retrieve(kb_id, case.question)
        images = []
        if mm.enabled:
            try:
                images, _ = multimodal_service.images_for_answer(kb_id, case.question, result.contexts, doc_ids=retrieval_service.visible_doc_ids(kb_id, case.question))
            except Exception as exc:  # 与线上一致：图片侧失败不影响文字回答
                logging.warning("图片检索失败：%s", exc)
        llm = model_service.llm()
        vision = model_service.vision_llm() if (images and mm.chat_attach_images > 0) else None
        if vision is not None and vision is not llm:
            llm = vision
        can_see = llm is not None and llm is vision
        payloads = multimodal_service.payloads(kb_id, images, mm.chat_attach_images) if (can_see and images) else []
        notes = ""
        if args.experience and result.contexts:
            notes, _ = lake_service.experience(case.question, kb_id=kb_id, session_id=None)
        gen = AnswerGenerator(settings.generation, llm)
        ans = gen.answer(result, images=images or None, image_payloads=payloads or None, reference_notes=notes or None)
        text = ans.text
        ctx_texts = [format_context(c) for c in result.contexts]
        img_texts = ["\n".join(x for x in (f"[图{i}] 来源：{h.record.label()}", h.record.caption, h.record.description, h.record.ocr_text) if x) for i, h in enumerate(images, start=1)]
        sources = ctx_texts + img_texts
        citations = parse_citations(text, len(ctx_texts))
        cited = bool(citations) or ("[图" in text)
        store = retrieval_service.retriever(kb_id).index.store
        expired = []
        for c in result.contexts:
            md = store.docs.get(c.chunk.doc_id)
            if md is not None and not md.is_effective(date.today()):
                expired.append(md.title or c.chunk.doc_id)
        row: dict[str, Any] = {
            "case": case.model_dump(),
            "answer": text,
            "score": score_case(case, text, sources, cited=cited),
            "citationIssues": check_citations(text, ctx_texts, question=case.question) if citations else [],
            "expiredInContext": sorted(set(expired)),
            "sources": sources,
            "contexts": [{"doc": (store.docs.get(c.chunk.doc_id).title if store.docs.get(c.chunk.doc_id) else c.chunk.doc_id), "score": round(c.score, 4), "weaved": c.weaved, "text": c.text_for_llm()[:300]} for c in result.contexts],
            "images": len(images),
            "notes": result.trace.notes,
            "model": ans.model,
            "seconds": round(time.perf_counter() - t0, 2),
        }
        if judge_llm is not None:
            try:
                resp = judge_llm.chat([{"role": "user", "content": build_judge_prompt(case.question, sources, text)}], system=JUDGE_SYSTEM, max_tokens=2048, thinking=False)
                row["judge"] = parse_judge(resp.text)
            except Exception as exc:
                row["judge"] = {"error": str(exc)[:200]}
        return row

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(run_one, cases))
    summary = summarize(rows)
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    summary["set"] = args.set
    summary["overrides"] = json.loads(args.overrides) if args.overrides else None
    summary["experience"] = args.experience

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}-{args.set if args.set in SETS or args.set == 'all' else Path(args.set).stem}{('-' + args.tag) if args.tag else ''}"
    (OUT / f"{name}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    md = to_markdown(summary, rows, title=f"回答评测 · {args.set}{(' · ' + args.tag) if args.tag else ''}")
    (OUT / f"{name}.md").write_text(md, encoding="utf-8")

    keys = ("accuracy", "answerAccuracy", "refusalAccuracy", "partialAccuracy", "premiseCorrection", "falseRefusalRate", "hallucinationRate", "ungroundedNumberRate", "citationIssueRate", "uncitedRate")
    prev = json.loads(args.compare.read_text(encoding="utf-8"))["summary"] if args.compare else None
    print(f"\n回答评测 {args.set}  n={summary['n']}  {summary['seconds']}s  → {OUT / name}.md")
    for k in keys:
        v = summary.get(k)
        line = f"  {k:22s} {'—' if v is None else f'{v * 100:6.1f}%'}"
        if prev is not None and prev.get(k) is not None and v is not None:
            line += f"   (之前 {prev[k] * 100:6.1f}%，{(v - prev[k]) * 100:+.1f})"
        print(line)
    print(f"  expiredInContext       {summary['expiredInContext']} 题")
    return 0


def rescore(path: Path) -> int:
    """打分规则改了之后，用已保存的回答重算（回答与资料不变，裁判结果沿用）。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    # 用评测集里当前的标注（按问题匹配）：标注修正后，旧报告按同一套规则重算才可比
    current = {c.question: c for p in SETS.values() if p.exists() for c in load_cases(p)}
    for r in data["rows"]:
        case = current.get(r["case"]["question"]) or AnswerCase.model_validate(r["case"])
        r["case"] = case.model_dump()
        sources = r.get("sources") or [c["text"] for c in r.get("contexts", [])]
        r["score"] = score_case(case, r["answer"], sources, cited=r["score"].get("cited"))
    summary = {**data["summary"], **summarize(data["rows"])}
    data["summary"] = summary
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    path.with_suffix(".md").write_text(to_markdown(summary, data["rows"], title=f"回答评测 · {summary.get('set')}（重算）"), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("byCategory",)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
