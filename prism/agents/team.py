"""多智能体协作：规划者 → 研究员（并行检索，证据贴黑板）→ 分析师 → 审校（证据不足 → 改写重查）→ 撰稿人。

- 规划者：把问题拆成 1~N 个可独立检索的子问题（参考该用户的长期记忆）；
- 研究员：每个子问题一个，并行检索知识库，把证据写上 LanceDB 黑板；
- 分析师：按子问题从黑板语义取回证据，逐条整理关键事实并标注 [n]；
- 审校：判断证据是否足以回答；不够就给出改写后的检索问句，研究员再查一轮（Agentic RAG 的"证据不够？改写重查"）；
- 撰稿人：先结论后细节、每条结论带 [n]，证据不足的部分明说。
没有配置 LLM 时整条流水线仍可运行：规划 = 原问题、分析 / 撰稿 = 抽取式、审校 = 规则。
run() 是生成器：每一步产出一个事件，前端用 SSE 实时展示。
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..config import AgentsConfig
from ..generation.answer import parse_citations
from .board import Blackboard

logger = logging.getLogger(__name__)

# (query, top_k) → [{"chunkId", "docId", "label", "text", "score"}]
SearchFn = Callable[[str, int], list[dict[str, Any]]]

ROLES = [
    {"key": "planner", "name": "规划者", "desc": "拆解问题、分派子任务"},
    {"key": "researcher", "name": "研究员", "desc": "并行检索知识库，把证据贴上黑板"},
    {"key": "analyst", "name": "分析师", "desc": "按子任务从黑板取回证据，整理关键事实"},
    {"key": "reviewer", "name": "审校", "desc": "核对证据是否充分，不够就改写重查"},
    {"key": "writer", "name": "撰稿人", "desc": "带引用写出最终回答"},
]

PLANNER_PROMPT = """你是规划者。把用户问题拆成 1~{n} 个可以独立在知识库里检索的子问题；本身就是单一问题时只输出原问题。
{memory}只输出 JSON：{{"subtasks": ["子问题1", "子问题2"], "reason": "一句话说明拆分思路"}}

用户问题：{question}"""

ANALYST_SYSTEM = """你是分析师。根据黑板上的证据，逐个子问题整理关键事实：每条事实后用 [n] 标注证据编号，数字 / 日期 / 编号必须与证据一致。
证据里没有的信息写"证据不足"，不要编造，也不要用常识补全。用简洁的中文 Markdown。"""

REVIEWER_PROMPT = """你是审校。判断下面的分析是否已经足以完整回答用户问题。
只输出 JSON：{{"sufficient": true 或 false, "missing": ["还缺的信息"], "followups": ["为补齐缺口应该再检索的问句，最多 2 条"], "comment": "一句话结论"}}

用户问题：{question}
子问题：{subtasks}
分析：
{analysis}"""

WRITER_SYSTEM = """你是撰稿人。基于分析与证据回答用户问题：先给直接结论，再给必要的条件 / 例外 / 步骤；
每条结论后用 [n] 标注证据编号（n 是证据序号）；证据不足的部分明确写"根据现有资料无法确定"，不要编造。用简洁的中文 Markdown。"""


def _json(raw: str, kind: str = "object") -> Any:
    m = re.search(r"\{.*\}" if kind == "object" else r"\[.*\]", raw or "", re.S)
    return json.loads(m.group(0)) if m else None


class AgentTeam:
    def __init__(self, board: Blackboard, search: SearchFn, *, llm=None, fast_llm=None, cfg: AgentsConfig | None = None,
                 memory_recall: Callable[[str], list[str]] | None = None) -> None:
        self.board = board
        self.search = search
        self.llm = llm
        self.fast_llm = fast_llm or llm
        self.cfg = cfg or AgentsConfig()
        self.memory_recall = memory_recall

    # ----------------------------------------------------------------------------------
    def run(self, question: str, *, user_id: str = "default", kb_id: str = "", run_id: str | None = None) -> Iterator[dict[str, Any]]:
        question = (question or "").strip()
        run_id = run_id or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()

        def ev(kind: str, agent: str, content: str = "", **data: Any) -> dict[str, Any]:
            return {"type": kind, "agent": agent, "content": content, "data": data, "ms": round((time.perf_counter() - t0) * 1000)}

        similar = self.board.similar_runs(question, exclude_run=run_id) if question else []
        self.board.post(run_id, "user", "question", question, {"kbId": kb_id, "userId": user_id})
        yield ev("run", "system", question, runId=run_id, roles=ROLES, similarRuns=similar, offline=self.llm is None)

        memories: list[str] = []
        if self.memory_recall and self.cfg.use_memory:
            try:
                memories = self.memory_recall(question)
            except Exception as exc:   # 记忆不可用不影响协作
                logger.info("记忆召回失败：%s", exc)
        if memories:
            yield ev("memory", "planner", "参考该用户的长期记忆", memories=memories)

        subtasks, reason = self._plan(question, memories)
        self.board.post(run_id, "planner", "plan", "\n".join(subtasks), {"reason": reason})
        yield ev("plan", "planner", reason, subtasks=subtasks)

        queries = list(subtasks)
        evidence: list[dict[str, Any]] = []
        analysis = ""
        verdict: dict[str, Any] = {"sufficient": False}
        for round_no in range(1, self.cfg.max_review_rounds + 2):
            for q, hits in self._research(queries):
                if hits:
                    self.board.post_many(run_id, "researcher", "evidence", [(f"{h['label']}\n{h['text']}", {**h, "query": q}) for h in hits])
                yield ev("evidence", "researcher", q, round=round_no, items=[{"label": h["label"], "chunkId": h["chunkId"], "score": round(float(h.get("score", 0.0)), 4)} for h in hits])

            evidence, per_task = self._gather(run_id, subtasks + [q for q in queries if q not in subtasks])
            yield ev("recall", "analyst", f"从黑板按子任务取回 {len(evidence)} 条证据", perTask=per_task)

            analysis = self._analyze(question, subtasks, evidence)
            self.board.post(run_id, "analyst", "analysis", analysis)
            yield ev("analysis", "analyst", analysis, round=round_no)

            verdict = self._review(question, subtasks, analysis, evidence, per_task)
            self.board.post(run_id, "reviewer", "review", json.dumps(verdict, ensure_ascii=False))
            yield ev("review", "reviewer", verdict.get("comment", ""), round=round_no, **{k: v for k, v in verdict.items() if k != "comment"})
            followups = [f for f in verdict.get("followups") or [] if f and f not in queries][:2]
            if verdict.get("sufficient") or round_no > self.cfg.max_review_rounds or not followups:
                break
            queries = followups
            yield ev("requery", "reviewer", "证据不足，改写后重新检索", queries=queries)

        answer = self._write(question, analysis, evidence, memories, sufficient=bool(verdict.get("sufficient")))
        self.board.post(run_id, "writer", "answer", answer, [e["chunkId"] for e in evidence])
        yield ev("final", "writer", answer, sources=evidence, citations=parse_citations(answer, len(evidence)), sufficient=bool(verdict.get("sufficient")))

    # ----------------------------------------------------------------------------------
    def _plan(self, question: str, memories: list[str]) -> tuple[list[str], str]:
        if self.fast_llm is None or not question:
            return [question], "未配置模型：按原问题检索"
        hint = ("关于用户（只用来理解提问背景，不要写进子问题）：" + "；".join(memories) + "\n") if memories else ""
        try:
            data = _json(self.fast_llm.complete(PLANNER_PROMPT.format(n=self.cfg.max_subtasks, memory=hint, question=question), max_tokens=400))
            subtasks = [str(s).strip() for s in (data or {}).get("subtasks") or [] if str(s).strip()][: self.cfg.max_subtasks]
            if subtasks:
                return subtasks, str((data or {}).get("reason") or "")
        except Exception as exc:
            logger.info("规划失败，按原问题检索：%s", exc)
        return [question], "规划失败：按原问题检索"

    def _research(self, queries: list[str]) -> list[tuple[str, list[dict[str, Any]]]]:
        def one(q: str) -> list[dict[str, Any]]:
            try:
                return self.search(q, self.cfg.research_top_k)
            except Exception as exc:
                logger.warning("研究员检索失败（%s）：%s", q, exc)
                return []

        if len(queries) == 1:
            return [(queries[0], one(queries[0]))]
        with ThreadPoolExecutor(max_workers=min(4, len(queries))) as pool:
            return list(zip(queries, pool.map(one, queries)))

    def _gather(self, run_id: str, tasks: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """按子任务在黑板上做语义检索取回证据，去重后统一编号 [1..n]。"""
        evidence: list[dict[str, Any]] = []
        index: dict[str, int] = {}
        per_task = []
        for task in tasks:
            got = self.board.recall(run_id, task, kinds=["evidence"], k=self.cfg.research_top_k)
            nums = []
            for row in got:
                refs = row["refs"] if isinstance(row["refs"], dict) else {}
                cid = refs.get("chunkId") or row["id"]
                if cid not in index and len(evidence) < 12:
                    index[cid] = len(evidence) + 1
                    evidence.append({"n": index[cid], "chunkId": cid, "docId": refs.get("docId", ""), "label": refs.get("label", ""), "text": refs.get("text", row["content"]), "task": task})
                if cid in index:
                    nums.append(index[cid])
            per_task.append({"task": task, "evidence": sorted(set(nums)), "best": max((r["similarity"] for r in got), default=0.0)})
        return evidence, per_task

    @staticmethod
    def _evidence_text(evidence: list[dict[str, Any]], limit: int = 700) -> str:
        return "\n\n".join(f"[{e['n']}] 来源：{e['label']}\n{e['text'][:limit]}" for e in evidence) or "（黑板上没有证据）"

    def _analyze(self, question: str, subtasks: list[str], evidence: list[dict[str, Any]]) -> str:
        if not evidence:
            return "证据不足：知识库里没有检索到与这些子问题相关的资料。"
        if self.llm is None:
            lines = []
            for task in subtasks:
                own = [e for e in evidence if e["task"] == task][:2] or evidence[:1]
                lines.append(f"- **{task}**：" + "；".join(f"{e['text'][:120]} [{e['n']}]" for e in own))
            return "\n".join(lines)
        prompt = f"用户问题：{question}\n子问题：\n" + "\n".join(f"- {t}" for t in subtasks) + f"\n\n<证据>\n{self._evidence_text(evidence)}\n</证据>"
        return self.llm.chat([{"role": "user", "content": prompt}], system=ANALYST_SYSTEM, max_tokens=900, thinking=False).text.strip()

    def _review(self, question: str, subtasks: list[str], analysis: str, evidence: list[dict[str, Any]], per_task: list[dict[str, Any]]) -> dict[str, Any]:
        empty = [p["task"] for p in per_task if not p["evidence"]]
        if self.fast_llm is None:
            ok = bool(evidence) and not empty
            return {"sufficient": ok, "missing": empty, "followups": [], "comment": "每个子问题都有证据" if ok else "有子问题没有找到证据"}
        try:
            data = _json(self.fast_llm.complete(REVIEWER_PROMPT.format(question=question, subtasks="；".join(subtasks), analysis=analysis[:3000]), max_tokens=300)) or {}
            return {"sufficient": bool(data.get("sufficient")), "missing": [str(x) for x in data.get("missing") or []],
                    "followups": [str(x).strip() for x in data.get("followups") or [] if str(x).strip()], "comment": str(data.get("comment") or "")}
        except Exception as exc:
            logger.info("审校失败，按规则判断：%s", exc)
            return {"sufficient": bool(evidence) and not empty, "missing": empty, "followups": [], "comment": "审校模型不可用，按规则判断"}

    def _write(self, question: str, analysis: str, evidence: list[dict[str, Any]], memories: list[str], *, sufficient: bool) -> str:
        if not evidence:
            return "根据现有资料无法回答：知识库里没有检索到相关内容。"
        if self.llm is None:
            head = "以下是多智能体协作整理出的要点（离线模式，未配置文本模型）：\n\n"
            return head + analysis
        hint = ("\n<用户背景>（只用于个性化称呼与表达方式，不作为事实依据）\n" + "\n".join(f"- {m}" for m in memories) + "\n</用户背景>") if memories else ""
        prompt = (f"<证据>\n{self._evidence_text(evidence)}\n</证据>\n\n<分析>\n{analysis}\n</分析>{hint}\n\n"
                  f"<问题>\n{question}\n</问题>\n" + ("" if sufficient else "\n注意：审校认为证据可能不充分，缺的部分要明确说明。"))
        return self.llm.chat([{"role": "user", "content": prompt}], system=WRITER_SYSTEM, max_tokens=1200, thinking=False).text.strip()
