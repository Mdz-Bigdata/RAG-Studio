"""飞书文档智能体：用自然语言对飞书文档做增删改查与分析——规划 → 调工具 → 看结果反思（要不要再查 / 再写）→ 汇总。

工具由服务层注入（核心库不碰数据库和知识库）：查（list_docs / search / read_doc）、增（create_doc）、改（append_doc / replace_doc /
replace_text）、删（delete_doc）、同步（sync_doc，把飞书最新内容重新入库）。分析不是单独的工具：读到的内容交给模型在汇总时完成
（总结 / 对比 / 找矛盾 / 抽要点），检索走 RAG 知识库里已同步的飞书文档。

写操作（write=True 的工具）默认不直接执行：智能体只给出"待确认的操作"（含完整参数），前端确认后再调接口执行；
allow_write=True 时才直接写飞书。没有文本模型时按规则路由：带文档链接 → 读文档；问"有哪些文档" → 列出；其余 → 检索。
run() 是生成器：每一步产出一个事件，前端用 SSE 实时展示。

docs 是当前知识库里的文档：已接入的飞书文档（documentId）和上传的文档（docId，kind="kb"，用 read_kb_doc 读）。
规划和反思都看得到，用户说「这篇文章 / 这份文档」时就知道指哪篇；模型仍然一个工具都没规划、而问题明显是在问文档内容时，
只有一篇就直接读它，有多篇就检索——不会再反过来让用户"把文章发给我"。
把知识库文档的正文写进飞书时用 source_doc_id：服务端原样复制正文，不让模型逐字转抄（又慢又容易漏改）。
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    desc: str                                  # 给模型看的说明，含参数格式
    fn: Callable[[dict[str, Any]], Any]
    write: bool = False                        # 会修改飞书文档（需确认）


_PLAN_SYSTEM = """你是飞书文档智能体，能对用户的飞书文档做增删改查和分析。可用工具：
{tools}

规则：
- 文档用 document_id 指代；用户给的是链接时，原样把链接填进 document_id（工具会自己解析）；不知道是哪篇时先 list_docs 或 search；
- 修改 / 删除前先 read_doc 看清当前内容，改动只动用户要求的部分：局部改字用 replace_text，加内容用 append_doc，整篇重写才用 replace_doc；
- 写入的内容用 Markdown（标题 / 列表 / 表格 / 代码块），只写有依据的内容：依据来自 read_doc / search 的结果，不要编造数字和事实；
- 分析类问题（总结 / 对比 / 找矛盾 / 列待办）先把相关文档读到，结论在最后汇总时写，不需要专门的工具；
- 把知识库文档的正文写进飞书（新建 / 追加 / 替换）时，在 create_doc / append_doc / replace_doc 里用 source_doc_id 指定那篇文档，不要自己抄写正文；
  要写进去的是你的总结 / 分析时，先读到原文，等看到结果再把写好的 Markdown 填进 markdown；
- 本轮能直接确定的调用都列出来，依赖前一步结果的（例如要先读到内容才能写）留到看到结果之后再列。

只输出 JSON，不要解释：{{"steps": [{{"tool": "工具名", "args": {{...}}, "why": "一句话说明"}}]}}；不需要任何工具时输出 {{"steps": []}}。
{docs}"""

_REFLECT_SYSTEM = """你是飞书文档智能体。下面是用户的请求和已经执行过的工具及其结果。判断：
- 已经完成用户的请求（该查的查到了 / 该写的写了或已提交待确认）→ 只输出 {{"done": true}}
- 还需要继续 → 输出接下来的工具调用（最多 {remaining} 个）：{{"steps": [{{"tool": "工具名", "args": {{...}}, "why": "..."}}]}}
规则：不要重复已经执行过的相同调用；工具报错时换参数或换工具，不要原样重试；写入的内容只能来自已读到的文档 / 检索结果或用户给出的信息。
状态为 pending 的写操作已经提交给用户确认，视为已完成，不要再提交一次。
可用工具：
{tools}
{docs}"""

_SUMMARY_SYSTEM = """你是飞书文档智能体。根据工具结果回答用户：
- 先说结论 / 做了什么，再给细节；引用文档内容时注明出自哪篇文档（标题）；
- 分析类问题（总结 / 对比 / 找矛盾 / 列待办）基于读到的原文给出结构化结论，原文没有的不要编造；
- 写操作：已执行的写明改了哪篇文档、改了什么；待确认（pending）的列出将要做的改动，提醒用户在页面上确认后才会执行；
- 工具报错的如实说明原因（例如应用没有文档权限）并给出处理办法；用简洁的中文 Markdown。"""

_DOCS_RULE = """
当前知识库里的文档（共 {count} 篇）：
{lines}
用户说「这篇 / 这份 / 这个文档 / 文章 / 全文」又没给链接时，指的就是上面这些文档：只有一篇就直接读它
（飞书文档用 read_doc，知识库文档用 read_kb_doc），有多篇先 search 确认是哪篇。文档已经在知识库里，不要反过来让用户提供链接或粘贴正文。"""
# 问的是文档内容（指代、总结、要点、分析），而不是闲聊。
_DOC_QUESTION_RE = re.compile(r"这篇|这份|这个文档|该文档|本文|文章|全文|文档|核心|要点|重点|主旨|总结|概括|摘要|讲了什么|说了什么|内容|对比|矛盾|待办")
_MAX_DOC_LINES = 30

_URL_RE = re.compile(r"https?://[^\s，。)）]+/(?:docx|wiki)/[A-Za-z0-9]+[^\s，。)）]*")


def _brief(value: Any, limit: int = 5000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…（已截断）"


def _parse_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


class FeishuAgent:
    def __init__(self, tools: dict[str, Tool], llm: Any | None = None, *, max_steps: int = 6, allow_write: bool = False,
                 docs: list[dict[str, Any]] | None = None) -> None:
        self.tools = tools
        self.llm = llm
        self.max_steps = max(1, max_steps)
        self.allow_write = allow_write
        # 当前知识库的文档：飞书文档 [{"title", "documentId", "url"}]，上传的文档 [{"title", "docId", "kind": "kb"}]
        self.docs = [d for d in (docs or []) if isinstance(d, dict) and (d.get("documentId") or d.get("url") or d.get("docId"))]

    def _tool_text(self) -> str:
        return "\n".join(f"- {name}{'（写操作）' if t.write else ''}：{t.desc}" for name, t in self.tools.items())

    def _docs_text(self) -> str:
        if not self.docs:
            return ""
        lines = "\n".join(f"- 《{d.get('title') or '未命名'}》 知识库文档 doc_id={d['docId']}（用 read_kb_doc 读）" if d.get("docId")
                          else f"- 《{d.get('title') or '未命名'}》 飞书文档 document_id={d.get('documentId') or d.get('url')}" for d in self.docs[:_MAX_DOC_LINES])
        more = f"\n- ……另有 {len(self.docs) - _MAX_DOC_LINES} 篇，用 list_docs 查看" if len(self.docs) > _MAX_DOC_LINES else ""
        return _DOCS_RULE.format(count=len(self.docs), lines=lines + more)

    def _doc_fallback(self, question: str) -> list[dict[str, Any]]:
        """The model planned nothing, yet the question is about the connected documents: read or search them."""
        if not self.docs or not _DOC_QUESTION_RE.search(question):
            return []
        if len(self.docs) == 1:
            doc = self.docs[0]
            why = f"知识库里只有《{doc.get('title') or '未命名'}》这一篇，问的就是它"
            if doc.get("docId") and "read_kb_doc" in self.tools:
                return [{"tool": "read_kb_doc", "args": {"doc_id": doc["docId"]}, "why": why}]
            if not doc.get("docId") and "read_doc" in self.tools:
                return [{"tool": "read_doc", "args": {"document_id": doc.get("documentId") or doc.get("url")}, "why": why}]
        if "search" in self.tools:
            return [{"tool": "search", "args": {"query": question, "top_k": 6}, "why": "在已接入的飞书文档里检索相关内容"}]
        return []

    # ----------------------------------------------------------------------------------
    def run(self, question: str, *, run_id: str | None = None) -> Iterator[dict[str, Any]]:
        question = (question or "").strip()
        run_id = run_id or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()

        def ev(kind: str, content: str = "", **data: Any) -> dict[str, Any]:
            return {"type": kind, "agent": "feishu", "content": content, "data": data, "ms": round((time.perf_counter() - t0) * 1000)}

        yield ev("run", question, runId=run_id, offline=self.llm is None, allowWrite=self.allow_write, tools=[{"name": n, "write": t.write} for n, t in self.tools.items()])
        steps, reason = self._plan(question)
        if not steps:
            fallback = self._doc_fallback(question)
            if fallback:
                steps, reason = fallback, "问的是已接入文档的内容：直接读取 / 检索文档"
        yield ev("plan", reason, steps=steps)

        executed: list[dict[str, Any]] = []
        seen: set[str] = set()
        while steps and len(executed) < self.max_steps:
            before = len(executed)
            for step in steps:
                if len(executed) >= self.max_steps:
                    break
                key = json.dumps([step.get("tool"), step.get("args")], ensure_ascii=False, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                record = self._execute(step)
                executed.append(record)
                yield ev("pending" if record["status"] == "pending" else "tool", record.get("why") or "", **record)
            steps = []
            if self.llm is None or len(executed) >= self.max_steps or len(executed) == before:
                break                                # 这一轮全是重复调用：模型在原地打转，直接汇总
            steps = self._reflect(question, executed)
            if steps:
                yield ev("reflect", "结果还不够，继续调用工具", steps=steps)

        answer = self._summarize(question, executed)
        actions = [{k: r[k] for k in ("id", "tool", "args", "status", "error") if k in r} for r in executed if r.get("write")]
        yield ev("final", answer, actions=actions, steps=len(executed))

    # ----------------------------------------------------------------------------------
    def _execute(self, step: dict[str, Any]) -> dict[str, Any]:
        name = str(step.get("tool") or "")
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        record: dict[str, Any] = {"id": uuid.uuid4().hex[:8], "tool": name, "args": args, "why": str(step.get("why") or "")}
        tool = self.tools.get(name)
        if tool is None:
            return {**record, "status": "error", "error": f"未知工具：{name}"}
        record["write"] = tool.write
        if tool.write and not self.allow_write:
            return {**record, "status": "pending", "result": "已提交给用户确认，确认后才会执行"}
        t = time.perf_counter()
        try:
            result = tool.fn(args)
            return {**record, "status": "ok", "result": result, "elapsedMs": round((time.perf_counter() - t) * 1000)}
        except Exception as exc:
            logger.info("飞书工具 %s 失败：%s", name, exc)
            return {**record, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:600], "elapsedMs": round((time.perf_counter() - t) * 1000)}

    def _plan(self, question: str) -> tuple[list[dict[str, Any]], str]:
        if self.llm is None:
            return self._rule_plan(question), "未配置文本模型：按规则选择工具（只能查询，不能生成要写入的内容）"
        try:
            raw = self.llm.chat([{"role": "user", "content": f"用户请求：{question}"}], system=_PLAN_SYSTEM.format(tools=self._tool_text(), docs=self._docs_text()), max_tokens=4000, thinking=False).text
            data = _parse_json(raw) or {}
            steps = [s for s in data.get("steps") or [] if isinstance(s, dict) and s.get("tool")]
            return steps[: self.max_steps], "模型规划" if steps else "模型判断不需要调用工具"
        except Exception as exc:
            logger.info("飞书智能体规划失败，按规则：%s", exc)
            return self._rule_plan(question), f"规划失败（{exc}），按规则选择工具"

    def _rule_plan(self, question: str) -> list[dict[str, Any]]:
        urls = _URL_RE.findall(question)
        if urls and "read_doc" in self.tools:
            return [{"tool": "read_doc", "args": {"document_id": u}, "why": "问题里带了文档链接"} for u in urls[: self.max_steps]]
        if re.search(r"有哪些|列出|清单|列表|多少篇|所有文档", question) and "list_docs" in self.tools:
            return [{"tool": "list_docs", "args": {}, "why": "列出已接入的飞书文档"}]
        if "search" in self.tools:
            return [{"tool": "search", "args": {"query": question, "top_k": 6}, "why": "在已同步的飞书文档里检索"}]
        return []

    def _reflect(self, question: str, executed: list[dict[str, Any]]) -> list[dict[str, Any]]:
        remaining = self.max_steps - len(executed)
        if remaining <= 0:
            return []
        try:
            raw = self.llm.chat([{"role": "user", "content": f"用户请求：{question}\n\n已执行：\n{self._history(executed)}"}],
                                system=_REFLECT_SYSTEM.format(remaining=remaining, tools=self._tool_text(), docs=self._docs_text()), max_tokens=4000, thinking=False).text
        except Exception as exc:
            logger.info("飞书智能体反思失败：%s", exc)
            return []
        data = _parse_json(raw) or {}
        if data.get("done"):
            return []
        return [s for s in data.get("steps") or [] if isinstance(s, dict) and s.get("tool")][:remaining]

    @staticmethod
    def _history(executed: list[dict[str, Any]], limit: int = 5000) -> str:
        parts = []
        for i, r in enumerate(executed, 1):
            body = r.get("error") if r["status"] == "error" else r.get("result")
            parts.append(f"[{i}] {r['tool']} {json.dumps(r.get('args') or {}, ensure_ascii=False)[:600]} → {r['status']}\n{_brief(body, limit)}")
        return "\n\n".join(parts) or "（还没有执行任何工具）"

    def _summarize(self, question: str, executed: list[dict[str, Any]]) -> str:
        if self.llm is not None:
            try:
                prompt = f"用户请求：{question}\n\n工具结果：\n{self._history(executed, 8000)}"
                return self.llm.chat([{"role": "user", "content": prompt}], system=_SUMMARY_SYSTEM, max_tokens=2000, thinking=False).text.strip()
            except Exception as exc:
                logger.info("飞书智能体汇总失败，用模板：%s", exc)
        return self._template_summary(executed)

    @staticmethod
    def _template_summary(executed: list[dict[str, Any]]) -> str:
        if not executed:
            return "没有调用任何工具：请换个问法，或给出飞书文档链接。"
        out = []
        for r in executed:
            if r["status"] == "error":
                out.append(f"- **{r['tool']}** 失败：{r.get('error')}")
            elif r["status"] == "pending":
                out.append(f"- **{r['tool']}** 待确认：{json.dumps(r.get('args'), ensure_ascii=False)[:200]}")
            elif r["tool"] == "search":
                hits = r.get("result") or []
                out.append(f"**检索到 {len(hits)} 条相关内容：**\n" + "\n".join(f"- 《{h.get('title')}》{(h.get('text') or '')[:160]}" for h in hits[:6]))
            elif r["tool"] in ("read_doc", "read_kb_doc"):
                doc = r.get("result") or {}
                out.append(f"**《{doc.get('title')}》**（{doc.get('chars', 0)} 字）\n\n{(doc.get('markdown') or '')[:1200]}")
            elif r["tool"] == "list_docs":
                docs = (r.get("result") or {}).get("docs") or []
                out.append(f"**共 {len(docs)} 篇飞书文档：**\n" + "\n".join(f"- {d.get('title')}（{d.get('documentId')}）" for d in docs[:30]))
            else:
                out.append(f"- **{r['tool']}**：{_brief(r.get('result'), 300)}")
        return "\n\n".join(out)
