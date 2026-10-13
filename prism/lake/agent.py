"""Agent 湖分析智能体：用自然语言问"湖"里的事——目录发现 → 规划 → 调工具 → 看结果反思（要不要再查）→ 汇总结论。

有文本模型时：先按问题在湖目录（列 / meta 键 / 指标口径 / 示例 SQL，见 catalog.py）里找相关条目喂给规划提示词，模型规划工具调用；
执行后把结果交回模型反思，不够就追加调用（ReAct，总步数不超过 agent_max_steps）；SQL 出错会把报错交给模型改一次；最后写结论。
没有模型时按关键词路由到内置分析，用模板写结论。
每次分析本身也写回 Agent 湖：一条 analysis 记录 + 每次工具调用一条 tool_call 记录（parent_id 指向 analysis），
所以"做过哪些分析、当时的结论是什么"同样可以检索、回放。
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from typing import Any

from ..config import LakeConfig
from . import analytics
from .catalog import LakeCatalog, check_filter, custom_entries, is_empty_result, learned_examples, rowcount
from .retest import RetestFn, run_retest
from .store import KIND_LABELS, AgentLake

logger = logging.getLogger(__name__)

TOOLS = {
    "catalog": "湖目录：按意图找相关的列（类型 / 说明 / 真实取值）、meta JSON 里的键与 SQL 写法、指标口径（带固定 SQL）、问题 → SQL 示例。写 SQL 前不确定列名 / meta 键 / 口径时先查。参数：{\"intent\": \"...\"}",
    "overview": "湖的概况：各类型记录数、会话 / 用户数、近 N 天每天的记录数。参数：{\"days\": 30}",
    "retrieval_quality": "问答链路体检：检索延迟 p50/p95、置信度分布、拒答率、反馈、被检索最多的文档、各检索策略对比。参数：{\"days\": null}",
    "knowledge_gaps": "知识缺口：没检索到 / 置信度低 / 拒答 / 被点踩、且还没被复测解决的问题；附带已解决数、每个缺口之后变更过的文档。参数：{\"limit\": 20, \"days\": null}",
    "retest_gaps": "复测缺口：把未解决的缺口问题拿到现在的知识库重新检索并生成回答，判断已解决 / 有改善 / 仍是缺口（会写回湖里）。问\"补上了吗 / 现在能答了吗\"时用。参数：{\"limit\": 10, \"include_weak\": false}",
    "routing": "路由情况：自动路由兜底触发次数与明细（从哪个库换到哪个库）、「答案在其它知识库」的问题之后再问是否已答对（已修复 / 仍在发生）、各知识库提问量。参数：{}",
    "kb_profiles": "各知识库当前的名称、描述、分类与文档清单（判断路由问题、给知识库描述建议前先看）。参数：{}",
    "topics": "把用户问题按语义聚类成主题：主题词、规模、缺口率、代表问题。参数：{\"k\": null}",
    "sql": "对表 lake 执行只读 DuckDB SQL（SELECT / WITH），JSON 列可用 json_extract_string(meta, '$.strategy')。参数：{\"query\": \"SELECT ...\"}",
    "search": "在湖里做语义 + 全文混合检索，找相关的问题 / 回答 / 记忆 / 文档；可叠加结构化过滤（LanceDB 先过滤再算相似度）。参数：{\"query\": \"...\", \"kinds\": [\"question\"], \"top_k\": 8, \"days\": 7, \"where\": \"kb_id = 'xxx' AND status <> 'ok'\"}（where 只能是列的比较 / AND / OR / LIKE / IN / IS NULL，不能有子查询和函数）",
}

_COMPASS = """决策罗盘（按问题的意图选工具）：
| 意图 | 典型问法 | 用什么 |
| 精确统计 / 排行 / 分组 / 按时间筛 | "每天多少提问""最慢的 10 次""各知识库缺口数" | sql（指标口径里有现成 SQL 就照抄） |
| 答不上 / 缺口 / 差评 / 该补什么语料 | "哪些问题没答上来" | knowledge_gaps（给"下一步建议"前先 routing + kb_profiles 看哪些已修过） |
| 补上了吗 / 解决了吗 / 现在能答吗 | "之前的缺口现在好了吗" | 先 retest_gaps 再 knowledge_gaps |
| 路由 / 选错库 / 知识库描述 | "是不是分错库了" | routing + kb_profiles |
| 主题 / 热点 / 分类 | "用户最关心什么" | topics |
| 找具体内容 / 模糊语义 | "有人问过 Kafka 吗""类似 X 的问题" | search（可加 kinds / days / where 结构化过滤） |
| 列名 / meta 键 / 口径不确定 | "meta 里有什么" | catalog |
| 整体体检 | "检索质量怎么样" | retrieval_quality / overview |"""

_PLAN_SYSTEM = """你是数据分析智能体，负责分析一张名为 lake 的 Agent 湖表（RAG 问答系统运行时沉淀的全部上下文：问题 / 检索日志 / 回答 / 反馈 / 工具调用 / 记忆 / 文档）。
可用工具：
{tools}

{compass}

{catalog}

表 lake 的全部列名：{columns}

根据用户的问题选择 1~{max_steps} 个工具调用，只输出 JSON，不要解释：
{{"steps": [{{"tool": "工具名", "args": {{...}}, "why": "一句话说明"}}]}}
SQL 规则：只能查 lake 表；先按 kind 过滤记录类型；问题与它的检索 / 回答靠 parent_id 关联（retrieval / answer 的 parent_id = question 的 id）；meta 是 JSON 字符串，用 json_extract_string(meta, '$.key')；
时间条件用 ts（Unix 秒，例如 ts >= epoch(now()) - 7*86400）或 day 列；统计类问题的 SQL 要带样本数（count）；不要 SELECT * 返回整行。
kind = 'doc_event' 是文档变更记录，kind = 'retest' 是复测记录，kind = 'analysis' / 'tool_call' 是以前的湖分析本身。"""

_REFLECT_SYSTEM = """你是数据分析智能体，正在分析 Agent 湖表 lake。下面是用户问题和已经执行过的工具及其结果。
判断：凭这些结果能否回答用户的问题？
- 能 → 只输出 {{"done": true}}
- 还缺某个数字 / 明细 / 对比 → 输出还需要的工具调用（最多 {remaining} 个）：{{"steps": [{{"tool": "工具名", "args": {{...}}, "why": "缺什么"}}]}}
规则：某个工具结果为空时，不要只改引号 / 大小写 / 同义词再试一次——要么换条件（放宽时间、去掉过滤），要么接受"没有数据"；
不要重复已经执行过的调用；不要为了"更全面"追加与问题无关的工具。
可用工具：
{tools}

{catalog}"""

_SUMMARY_SYSTEM = """你是数据分析师。根据工具返回的结果回答用户关于 Agent 湖的问题：
- 先给结论，再列关键数字（必须来自工具结果，不要编造）；统计类结论写明口径和样本数（例如"近 7 天 42 个问题中 6 个拒答，拒答率 14%"）；
- 相关性只能说"统计上相关"，不要说成因果；某个工具结果为空就直说没有数据，不要推测；
- 发现问题时给出可执行的建议（例如补充哪类语料、调整哪个检索参数）；
- 复测结果"答案在其它知识库"（wrong_kb / misrouted）表示提问时选错了知识库或路由不准，建议用自动路由 / 调整知识库描述，不要建议补语料；
- "能回答（检索分偏低）"表示问法笼统、重排分低但回答正常，不是缺口，不要建议补语料或调检索参数；
- 已经做过的修复要计入结论、不要重复建议：知识库描述已覆盖相关主题（看 kb_profiles）、问答已有自动路由兜底和选库提示（routing 里兜底已生效 / 之后再问已答对）；
- 引用每个问题的分数时，用该问题自己那一行的数值（retestScore / newScore），不要把别的问题的分数挪过来；
- 结果为空就直说没有数据；用简洁的中文 Markdown。"""

_RULES = [
    (re.compile(r"有哪些列|哪些字段|表结构|meta 里|元数据|口径|怎么算|指标"), "catalog"),
    (re.compile(r"路由|选错|误路由|知识库描述|画像|分错"), "routing"),
    (re.compile(r"路由|知识库描述|画像|有哪些知识库"), "kb_profiles"),
    (re.compile(r"复测|补上|补好|解决了|修好|现在能|还缺"), "retest_gaps"),
    (re.compile(r"缺口|没答|答不上|拒答|点踩|差评|失败|没回答|答错"), "knowledge_gaps"),
    (re.compile(r"主题|热点|话题|聚类|分类|关注|高频"), "topics"),
    (re.compile(r"延迟|耗时|慢|质量|置信|分数|命中|策略|文档.*(多|少)|引用"), "retrieval_quality"),
    (re.compile(r"趋势|每天|每日|多少|数量|统计|概况|总体|总共|分布"), "overview"),
]


def _brief(value: Any, limit: int = 3500) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…（已截断）"


def _parse_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


class LakeAnalyst:
    def __init__(self, lake: AgentLake, llm: Any | None, cfg: LakeConfig | None = None, *, version: int | str | None = None, branch: str | None = None,
                 retest_fn: RetestFn | None = None, kb_info_fn: Callable[[], list[dict[str, Any]]] | None = None) -> None:
        self.lake = lake
        self.llm = llm
        self.retest_fn = retest_fn          # 复测需要现在的检索 / 回答能力，由服务层注入；没有就不提供 retest_gaps
        self.kb_info_fn = kb_info_fn        # 知识库画像（名称 / 描述 / 文档）由服务层提供；核心库不读业务库
        self._kb_cache: list[dict[str, Any]] | None = None
        self.cfg = cfg or LakeConfig()
        self._ref = (version, branch)
        self.view = lake.view(version=version, branch=branch)
        self._df = None
        self._df_vec = None
        self._catalog: LakeCatalog | None = None

    # ------------------------------------------------------------------ 数据
    def df(self):
        if self._df is None:
            self._df = analytics.frame(self.lake.to_arrow(view=self.view))
        return self._df

    def df_vec(self):
        if self._df_vec is None:
            self._df_vec = analytics.frame(self.lake.to_arrow(view=self.view, with_vector=True))
        return self._df_vec

    def catalog(self) -> LakeCatalog:
        """这份快照的目录（列 / meta 键 / 指标口径 / 示例），向量用湖的 embedder。"""
        if self._catalog is None:
            try:
                labels = {"kb_id": self.kb_names()}
            except Exception:
                labels = {}
            self._catalog = LakeCatalog.build(self.df(), self._columns(), low_score=self.cfg.low_score, embedder=getattr(self.lake, "embedder", None),
                                              custom=custom_entries(self.lake, view=self.view), labels=labels)
        return self._catalog

    def discover(self, intent: str) -> dict[str, Any]:
        """按意图找目录项，并附上湖里以前分析成功的 SQL 作为示例。"""
        extra = learned_examples(self.lake, intent, top_k=self.cfg.catalog_examples, view=self.view) if self.cfg.catalog_examples > 0 else []
        return self.catalog().discover(intent, extra_examples=extra)

    def kb_profiles(self) -> list[dict[str, Any]]:
        if self._kb_cache is None:
            self._kb_cache = list(self.kb_info_fn()) if self.kb_info_fn else []
        return self._kb_cache

    def kb_names(self) -> dict[str, str]:
        return {k["kbId"]: k["name"] for k in self.kb_profiles()}

    # ------------------------------------------------------------------ 工具
    def call(self, tool: str, args: dict[str, Any]) -> Any:
        args = args or {}
        if tool == "catalog":
            found = self.discover(str(args.get("intent") or ""))
            return {k: v for k, v in found.items() if k != "intent"}
        if tool == "routing":
            return analytics.routing(self.df(), kb_names=self.kb_names(), low_score=self.cfg.low_score)
        if tool == "kb_profiles":
            if self.kb_info_fn is None:
                raise ValueError("当前环境拿不到知识库信息")
            return self.kb_profiles()
        if tool == "overview":
            return analytics.overview(self.df(), days=int(args.get("days") or 30))
        if tool == "retrieval_quality":
            r = analytics.retrieval_quality(self.df(), low_score=self.cfg.low_score, days=args.get("days"), gap_limit=10)
            return {k: v for k, v in r.items() if k != "gaps"} | {"gapSamples": [g["question"] for g in r["gaps"][:5]]}
        if tool == "knowledge_gaps":
            r = analytics.retrieval_quality(self.df(), low_score=self.cfg.low_score, days=args.get("days"), gap_limit=int(args.get("limit") or 20))
            names = self.kb_names()

            def view(g: dict[str, Any]) -> dict[str, Any]:
                return {"question": g["question"], "kb": names.get(g["kbId"], g["kbId"]), "reasons": g["reasons"], "score": g["score"],
                        "day": time.strftime("%Y-%m-%d", time.localtime(g["ts"])), "retest": (g.get("retest") or {}).get("status"),
                        "retestScore": (g.get("retest") or {}).get("score"), "docChangesAfter": g.get("docChangesAfter") or [],
                        "answerInKb": ((g.get("retest") or {}).get("altKb") or {}).get("name")}
            return {"gapCount": r["gapCount"], "gapRate": r["gapRate"], "questions": r["questions"], "historicalGapCount": r["historicalGapCount"],
                    "resolvedCount": r["resolvedCount"], "misroutedCount": r["misroutedCount"], "misrouted": [view(g) for g in r["misroutedGaps"][:10]], "lastRetest": time.strftime("%Y-%m-%d %H:%M", time.localtime(r["lastRetestTs"])) if r["lastRetestTs"] else None,
                    "gaps": [view(g) for g in r["gaps"]], "resolved": [view(g) for g in r["resolvedGaps"][:10]]}
        if tool == "retest_gaps":
            if self.retest_fn is None:
                raise ValueError("当前环境不支持复测（没有可用的检索服务）")
            out = run_retest(self.lake, self.df(), self.retest_fn, low_score=self.cfg.low_score, include_weak=bool(args.get("include_weak")),
                             limit=int(args.get("limit") or 10), agent="lake-analyst")
            self._df = self._df_vec = None            # 写入了复测记录：重新打开表（旧句柄停在打开时的版本），后续工具读到最新数据
            self.view = self.lake.view(version=self._ref[0], branch=self._ref[1])
            return {"tested": out["tested"], "counts": out["counts"],
                    "results": [{"question": x["question"], "status": x["statusLabel"], "oldScore": x["oldScore"], "newScore": x["newScore"], "refused": x["refused"],
                                 "answer": x["answer"][:160], "answerInKb": (x.get("altKb") or {}).get("name")} for x in out["results"]]}
        if tool == "topics":
            q = analytics.retrieval_quality(self.df(), low_score=self.cfg.low_score, gap_limit=10_000)
            scores = {}
            r = self.df()
            for row in r[r["kind"] == "retrieval"].itertuples():
                if row.score == row.score and row.score is not None:
                    scores[row.parent_id] = float(row.score)
            t = analytics.topics(self.df_vec(), k=args.get("k"), gaps={g["questionId"] for g in q["gaps"]}, scores=scores)
            t.pop("assignments", None)
            return t
        if tool == "sql":
            return analytics.run_sql({"lake": self.lake.to_arrow(view=self.view)}, str(args.get("query") or ""), limit=min(self.cfg.sql_row_limit, 200))
        if tool == "search":
            kinds = args.get("kinds") or None
            extra = []
            if args.get("days"):
                extra.append(f"ts >= {time.time() - float(args['days']) * 86400:.0f}")
            if args.get("where"):
                extra.append(check_filter(str(args["where"]), [c["name"] for c in self._columns()]))
            where = self.lake.where_clause(kinds=kinds, extra=" AND ".join(f"({x})" for x in extra) or None)
            hits = self.lake.search(str(args.get("query") or ""), top_k=int(args.get("top_k") or 8), where=where, view=self.view)
            names = self.kb_names()
            return [{"kind": h["kind"], "content": h["content"][:240], "day": h["day"], "kb": names.get(h.get("kb_id") or "", h.get("kb_id")), "status": h.get("status"),
                     "score": h["rrf"], "routes": h["routes"]} for h in hits]
        raise ValueError(f"未知工具：{tool}")

    # ------------------------------------------------------------------ 规划
    def _tool_doc(self) -> str:
        return "\n".join(f"- {k}: {v}" for k, v in TOOLS.items() if k != "retest_gaps" or self.retest_fn is not None)

    def plan(self, question: str, found: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], str]:
        if self.llm is not None:
            system = _PLAN_SYSTEM.format(tools=self._tool_doc(), compass=_COMPASS, catalog=LakeCatalog.prompt_doc(found) if found else "",
                                         columns=", ".join(c["name"] for c in self._columns()), max_steps=self.cfg.agent_max_steps)
            try:
                resp = self.llm.chat([{"role": "user", "content": question}], system=system, max_tokens=800, thinking=False)
                data = _parse_json(resp.text)
                steps = [s for s in (data or {}).get("steps", []) if isinstance(s, dict) and s.get("tool") in TOOLS]
                if steps:
                    return steps[: self.cfg.agent_max_steps], "llm"
            except Exception as exc:
                logger.warning("湖分析规划失败，改用规则：%s", exc)
        return self.rule_plan(question), "rule"

    def reflect(self, question: str, steps: list[dict[str, Any]], found: dict[str, Any] | None, remaining: int) -> list[dict[str, Any]]:
        """看着已有结果决定还要不要查：返回追加的调用（去掉与已执行重复的），够用或解析失败时返回空。"""
        if self.llm is None or remaining <= 0:
            return []
        system = _REFLECT_SYSTEM.format(remaining=remaining, tools=self._tool_doc(), catalog=LakeCatalog.prompt_doc(found) if found else "")
        body = "\n\n".join(f"### 已执行 {i + 1}：{s['tool']} {_brief(s['args'], 300)}\n" + (f"出错：{s['error']}" if s["error"] else ("结果为空" if s.get("empty") else _brief(s["result"], 1800)))
                             for i, s in enumerate(steps))
        try:
            data = _parse_json(self.llm.chat([{"role": "user", "content": f"用户问题：{question}\n\n已执行的工具：\n{body}"}], system=system, max_tokens=600, thinking=False).text)
        except Exception as exc:
            logger.warning("湖分析反思失败，直接汇总：%s", exc)
            return []
        if not data or data.get("done"):
            return []
        done = {(s["tool"], json.dumps(s["args"], sort_keys=True, ensure_ascii=False)) for s in steps}
        out = []
        for s in data.get("steps", []):
            if not isinstance(s, dict) or s.get("tool") not in TOOLS:
                continue
            key = (s["tool"], json.dumps(dict(s.get("args") or {}), sort_keys=True, ensure_ascii=False))
            if key in done:
                continue
            done.add(key)
            out.append(s)
        return out[:remaining]

    def rule_plan(self, question: str) -> list[dict[str, Any]]:
        steps = [{"tool": tool, "args": {}, "why": f"问题里提到了{pat.pattern.split('|')[0]}等"} for pat, tool in _RULES
                 if pat.search(question) and (tool != "retest_gaps" or self.retest_fn is not None)]
        if any(s["tool"] == "retest_gaps" for s in steps) and not any(s["tool"] == "knowledge_gaps" for s in steps):
            steps.insert(1, {"tool": "knowledge_gaps", "args": {}, "why": "复测之后看还剩哪些缺口"})
        # 没有模型也能算指标：问题命中了目录里某个指标口径的关键词（内置或人工维护的），就直接跑它的固定 SQL
        try:
            metrics = [m for m in self.catalog().discover(question, per_kind={"metric": 3})["metrics"] if m["relevance"] >= 1.0]
        except Exception as exc:
            logger.info("湖目录不可用，规则规划不带指标：%s", exc)
            metrics = []
        steps += [{"tool": "sql", "args": {"query": m["sql"]}, "why": f"指标口径：{m['name']} = {m['definition']}"} for m in metrics[:2]]
        if not steps:
            steps = [{"tool": "search", "args": {"query": question, "top_k": 8}, "why": "没有匹配到统计类意图，先在湖里检索相关记录"},
                     {"tool": "overview", "args": {}, "why": "附上湖的整体概况"}]
        return steps[: self.cfg.agent_max_steps]

    def _columns(self) -> list[dict[str, str]]:
        return [{"name": f.name, "type": str(f.type)} for f in (self.view.schema if self.view is not None else []) if f.name != "vector"]

    # ------------------------------------------------------------------ 执行
    def run(self, question: str, *, record: bool = True, user_id: str = "default") -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            raise ValueError("请输入要分析的问题")
        t0 = time.perf_counter()
        # 目录发现：有模型时先按问题挑出相关的列 / meta 键 / 口径 / 示例，规划和反思都带着它
        found = None
        if self.llm is not None:
            try:
                found = self.discover(question)
            except Exception as exc:
                logger.warning("湖目录发现失败，按全表列名规划：%s", exc)
        steps_plan, planner = self.plan(question, found)
        steps: list[dict[str, Any]] = []
        rounds = 0
        while steps_plan:
            rounds += 1
            for s in steps_plan:
                steps.append(self._execute(question, s))
            remaining = self.cfg.agent_max_steps - len(steps)
            steps_plan = self.reflect(question, steps, found, remaining) if self.cfg.agent_reflect and planner == "llm" else []
        answer = self._summarize(question, steps) if self.llm is not None else None
        summarizer = "llm" if answer else "template"
        answer = answer or self._template(steps)
        out = {"question": question, "planner": planner, "summarizer": summarizer, "rounds": rounds, "steps": steps, "answer": answer,
               "catalog": {k: v for k, v in found.items() if k != "intent"} if found else None, "elapsedMs": round((time.perf_counter() - t0) * 1000, 1)}
        if record:
            out["recordId"] = self._record(out, user_id)
        return out

    def _execute(self, question: str, s: dict[str, Any]) -> dict[str, Any]:
        tool, args = s["tool"], dict(s.get("args") or {})
        t1 = time.perf_counter()
        try:
            result, error = self.call(tool, args), None
        except Exception as exc:
            result, error = None, str(exc)
            if tool == "sql" and self.llm is not None:
                fixed = self._fix_sql(question, args.get("query", ""), error)
                if fixed:
                    args = {"query": fixed, "fixedFrom": args.get("query")}
                    try:
                        result, error = self.call(tool, {"query": fixed}), None
                    except Exception as exc2:
                        error = str(exc2)
        return {"tool": tool, "args": args, "why": s.get("why", ""), "result": result, "error": error, "empty": error is None and is_empty_result(tool, result),
                "rows": rowcount(result) if error is None else None, "ms": round((time.perf_counter() - t1) * 1000, 1)}

    def _fix_sql(self, question: str, sql: str, error: str) -> str | None:
        prompt = f"问题：{question}\n原 SQL：{sql}\n报错：{error}\n表 lake 的列：\n{analytics.schema_doc(self._columns())}\n请只输出修正后的一条 DuckDB SELECT 语句。"
        try:
            text = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=600, thinking=False).text
        except Exception:
            return None
        m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S)
        sql2 = (m.group(1) if m else text).strip().rstrip(";")
        return sql2 if sql2.lower().startswith(("select", "with")) else None

    def _summarize(self, question: str, steps: list[dict[str, Any]]) -> str | None:
        body = "\n\n".join(f"### 工具 {i + 1}：{s['tool']}（{s['why']}）\n参数：{_brief(s['args'], 400)}\n结果：{('出错：' + s['error']) if s['error'] else ('为空（没有数据）' if s.get('empty') else _brief(s['result']))}" for i, s in enumerate(steps))
        try:
            text = self.llm.chat([{"role": "user", "content": f"用户问题：{question}\n\n{body}"}], system=_SUMMARY_SYSTEM, max_tokens=1500, thinking=False).text.strip()
            return text or None
        except Exception as exc:
            logger.warning("湖分析汇总失败，改用模板：%s", exc)
            return None

    @staticmethod
    def _template(steps: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for s in steps:
            r, tool = s["result"], s["tool"]
            if s["error"]:
                lines.append(f"- **{tool}** 出错：{s['error']}")
            elif tool == "catalog":
                lines.append(f"**湖目录**：相关列 {len(r['columns'])} 个、meta 键 {len(r['metaKeys'])} 个、指标 {len(r['metrics'])} 个、示例 {len(r['examples'])} 条。")
                lines += [f"  - {c['doc']}" for c in r["columns"][:6]] + [f"  - {m['doc']}" for m in r["metaKeys"][:4]] + [f"  - {m['name']} = {m['definition']}" for m in r["metrics"][:3]]
            elif tool == "overview":
                kinds = "、".join(f"{KIND_LABELS.get(k, k)} {n}" for k, n in r["kinds"].items()) or "无"
                lines.append(f"**概况**：共 {r['rows']} 条记录（{kinds}），{r['sessions']} 个会话、{r['users']} 个用户。")
            elif tool == "retrieval_quality":
                lat, sc = r["latency"], r["score"]
                rate = "—" if r["refusalRate"] is None else f"{r['refusalRate']:.0%}"
                lines.append(f"**检索质量**：{r['questions']} 个问题，检索耗时 p50 {lat['p50'] or '—'} ms / p95 {lat['p95'] or '—'} ms，"
                             f"平均最高分 {sc['avg'] if sc['avg'] is not None else '—'}，低置信 {sc['lowCount']} 次，拒答率 {rate}，"
                             f"反馈 👍{r['feedback']['up']} / 👎{r['feedback']['down']}，知识缺口 {r['gapCount']} 个。")
                if r["topDocs"]:
                    lines.append("被检索最多的文档：" + "、".join(f"{d['fileName']}（{d['retrieved']} 次）" for d in r["topDocs"][:5]))
            elif tool == "knowledge_gaps":
                lines.append(f"**知识缺口**：{r['gapCount']} 个仍未解决（历史共 {r['historicalGapCount']} 个，复测已解决 {r['resolvedCount']} 个，答案在其它知识库 {r['misroutedCount']} 个）" + ("：" if r["gaps"] else "。"))
                lines += [f"  - {g['question']}（{'、'.join(g['reasons'])}{'；之后变更过：' + '、'.join(g['docChangesAfter']) if g['docChangesAfter'] else ''}）" for g in r["gaps"][:10]]
            elif tool == "retest_gaps":
                c = r["counts"]
                lines.append(f"**复测**：{r['tested']} 个问题，已解决 {c['resolved']}、有改善 {c['improved']}、能回答（检索分偏低）{c['answerable']}、仍是缺口 {c['still_gap']}、答案在其它知识库 {c['wrong_kb']}" + ("：" if r["results"] else "。"))
                lines += [f"  - {x['question']}：{x['status']}（{x['oldScore'] if x['oldScore'] is not None else '—'} → {x['newScore'] if x['newScore'] is not None else '—'}）" for x in r["results"][:10]]
            elif tool == "topics":
                lines.append(f"**问题主题**：{r['questions']} 个问题聚成 {r['k']} 类：")
                lines += [f"  - {c['label']}：{c['size']} 个，缺口率 {c['gapRate']:.0%}，例：{c['samples'][0]['question']}" for c in r["clusters"]]
            elif tool == "sql":
                label = s["why"].split(" = ")[0].replace("指标口径：", "") if s["why"].startswith("指标口径：") else "SQL"
                if not r["rowCount"]:
                    lines.append(f"**{label}**：没有符合条件的数据。")
                elif r["rowCount"] == 1:
                    lines.append(f"**{label}**：" + "，".join(f"{c} = {v}" for c, v in zip(r["columns"], r["rows"][0])) + (f"（口径：{s['why'].split(' = ', 1)[1]}）" if " = " in s["why"] and label != "SQL" else ""))
                else:
                    lines.append(f"**{label}**：返回 {r['rowCount']} 行（{', '.join(r['columns'])}）：")
                    lines += ["  - " + "，".join(f"{c} = {v}" for c, v in zip(r["columns"], row)) for row in r["rows"][:5]]
            elif tool == "search":
                lines.append(f"**检索**：找到 {len(r)} 条相关记录：")
                lines += [f"  - [{KIND_LABELS.get(h['kind'], h['kind'])}] {h['content'][:80]}" for h in r[:6]]
        lines.append("\n_（未配置文本模型：按关键词选择分析工具、用模板生成结论。配置模型后由模型规划并撰写分析。）_")
        return "\n".join(lines)

    def _record(self, out: dict[str, Any], user_id: str) -> str | None:
        rid = uuid.uuid4().hex
        records = [{"id": rid, "kind": "analysis", "agent": "lake-analyst", "user_id": user_id, "title": out["question"], "content": f"{out['question']}\n\n{out['answer']}",
                    "meta": {"planner": out["planner"], "summarizer": out["summarizer"], "rounds": out["rounds"], "tools": [s["tool"] for s in out["steps"]],
                             "catalogHits": {k: len(v) for k, v in (out.get("catalog") or {}).items() if isinstance(v, list)}}, "latency_ms": out["elapsedMs"]}]
        for i, s in enumerate(out["steps"]):
            records.append({"kind": "tool_call", "agent": "lake-analyst", "user_id": user_id, "parent_id": rid, "title": s["tool"],
                            "content": f"{s['tool']} {json.dumps(s['args'], ensure_ascii=False)}\n{_brief(s['result'], 1500) if s['error'] is None else s['error']}",
                            "meta": {"step": i + 1, "args": s["args"], "why": s["why"], "rows": s.get("rows")}, "status": "error" if s["error"] else "ok", "latency_ms": s["ms"]})
        try:
            self.lake.append(records)
            return rid
        except Exception as exc:
            logger.warning("湖分析记录写入失败：%s", exc)
            return None
