"""从知识库语料合成"难题"评估集：口语化 / 换说法 / 条件数值类问题，标注可校验。

真实黄金集被现有系统做满分后就失去了区分度。这里让文本模型逐段出题，要求问题避开原文措辞（检索最容易失手的情况），
并对每道题做三项校验，不过关重试一次、仍不过关丢弃：
  1. answer_keywords 必须逐字出现在出题段落里（标注可被评测程序判定）；
  2. 只保留在全库不超过 2 个段落里出现的关键词，一个都不剩就丢弃（评测判定"命中任一关键词即相关"，
     "人力资源部""每月"这类常见词会让错误段落也算命中）；
  3. 问题与原文的最长公共子串不超过 8 个字（确实换了说法，不是把原句改成问号）。

输出写到单独的文件（默认 data/eval/synthetic_qa.jsonl），不改动人工维护的黄金集；每条带 "synthetic": true 与出题段落 id，便于人工复核。

用法（在项目根目录，使用 Web 应用当前激活的文本模型与知识库索引，只读）：
    .venv/bin/python scripts/synth_eval.py <kb_id> [--out data/eval/synthetic_qa.jsonl] [--limit 40]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STYLES = [
    ("口语化", "像员工在群里随口问的那样，口语、可以有省略和语气词，不要用文档里的专业说法"),
    ("换说法", "换一种说法提问：用同义词和不同的句式，不要复用原文里超过 4 个字的连续片段"),
    ("条件数值", "问一个带具体条件或数字的问题（例如某个级别 / 金额 / 时限 / 错误码对应的处理），答案必须是原文里的具体规定"),
]

PROMPT = """下面是知识库中的一段内容（来自《{title}》，章节：{crumb}）：
<段落>
{text}
</段落>

请出一道这段内容能直接回答的问题，用于测试检索系统。要求：
- 风格：{style_desc}
- 问题只能依赖这一段就能回答，不要问段落里没有的信息
- 问题里不要出现文档标题和章节名
- 再给出 2~3 个"答案关键词"：必须是段落原文中逐字出现的词或数字（如"3 月 31 日""1500 元""SO-8817"），能体现这道题的答案

只输出 JSON：{{"question": "...", "answer_keywords": ["...", "..."]}}"""


def lcs_len(a: str, b: str) -> int:
    """最长公共子串长度（字符级，忽略空白）。"""
    a, b = re.sub(r"\s+", "", a), re.sub(r"\s+", "", b)
    best, prev = 0, [0] * (len(b) + 1)
    for ch in a:
        cur = [0] * (len(b) + 1)
        for j, bj in enumerate(b, 1):
            if ch == bj:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def parse(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    if not isinstance(data, dict) or not str(data.get("question") or "").strip():
        return None
    kws = [str(k).strip() for k in data.get("answer_keywords") or [] if str(k).strip()]
    return {"question": str(data["question"]).strip(), "answer_keywords": kws}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kb_id")
    ap.add_argument("--out", default=str(ROOT / "data" / "eval" / "synthetic_qa.jsonl"))
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    logging.disable(logging.WARNING)

    from app.db import Repository, SessionLocal
    from app.services import index_service, model_service

    llm = model_service.llm(aux=True)
    if llm is None:
        sys.exit("没有可用的文本模型：先在「模型配置」里激活一个文本模型")
    idx = index_service.live(args.kb_id)
    with SessionLocal() as db:
        docs = {d.id: d for d in Repository(db).list_documents(args.kb_id)}
    # 只从现行的普通文档出题：增强语料（FAQ）本身是合成的，过期文档检索时会被过滤
    sources = {doc_id for doc_id, md in idx.store.docs.items() if md.is_effective() and doc_id in docs and not (docs[doc_id].parser or "").startswith("augment")}
    parents = sorted((c for c in idx.store.chunks.values() if c.kind == "parent" and c.doc_id in sources and len(c.text.strip()) >= 30), key=lambda c: (c.doc_id, c.order))[: args.limit]
    all_parents = [c.text for c in idx.store.chunks.values() if c.kind == "parent"]

    def spread(kw: str) -> int:
        return sum(1 for t in all_parents if kw in t)

    out, dropped = [], Counter()
    for i, p in enumerate(parents):
        style, style_desc = STYLES[i % len(STYLES)]
        md = idx.store.docs[p.doc_id]
        item, reason = None, ""
        for _attempt in range(2):
            try:
                raw = llm.complete(PROMPT.format(title=md.title or "", crumb=p.breadcrumb_text or "-", text=p.text[:1500], style_desc=style_desc), max_tokens=300)
            except Exception as exc:
                reason = f"模型调用失败：{exc}"[:60]
                continue
            cand = parse(raw)
            if cand is None:
                reason = "输出不是合法 JSON"
                continue
            kws = [k for k in cand["answer_keywords"] if k in p.text]
            if len(kws) < 1:
                reason = "关键词不在原文里"
                continue
            # 评测判定"命中任一关键词即相关"：常见词（"人力资源部""每月"）会让错误段落也算命中，只保留有区分度的
            kws = [k for k in kws if spread(k) <= 2]
            if not kws:
                reason = "关键词太常见，区分不了段落"
                continue
            if lcs_len(cand["question"], p.text) > 8:
                reason = "问题照抄原文"
                continue
            item = {"question": cand["question"], "answer_keywords": kws, "relevant_sources": [docs[p.doc_id].file_name], "category": f"合成·{style}",
                    "synthetic": True, "source_chunk_id": p.chunk_id}
            break
        if item:
            out.append(item)
            print(f"✓ [{style}] {item['question']}  ←  {p.breadcrumb_text}  {item['answer_keywords']}")
        else:
            dropped[reason] += 1
            print(f"✗ [{style}] {p.breadcrumb_text}：{reason}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n写出 {len(out)} 道题 → {args.out}；丢弃 {sum(dropped.values())} 道：{dict(dropped)}")


if __name__ == "__main__":
    main()
