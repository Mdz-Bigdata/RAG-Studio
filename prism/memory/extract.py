"""从一轮对话里抽取值得长期记住的用户信息：LLM（辅助模型，JSON 输出）或规则兜底。

只记"关于用户"的稳定信息：身份 / 部门、偏好、明确要求记住的事、正在进行的任务；
不记一次性的问题本身（"年假可以跨年吗"），也不把知识库里的事实当成用户记忆。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = """从下面这轮对话里提取值得长期记住的"关于用户"的信息，用于以后个性化回答。
只提取：用户的身份 / 部门 / 职级、偏好与习惯（例如希望回答简短、先给结论）、用户明确要求记住的事、用户正在进行的任务或计划。
不要提取：一次性的提问内容本身、知识库资料里的事实、助手说的话、寒暄。
每条写成一句第三人称陈述（"用户在财务部"），importance 0~1（明确要求记住的 0.9，身份 0.8，偏好 0.7，任务 0.6）。
kind 取 profile / preference / fact / task 之一。只输出 JSON 数组，没有就输出 []，格式：
[{{"content": "用户在财务部", "kind": "profile", "importance": 0.8}}]

用户：{user}
助手：{assistant}"""


@dataclass
class MemoryCandidate:
    content: str
    kind: str = "fact"
    importance: float = 0.5


_RULES: list[tuple[re.Pattern[str], str, float, str]] = [
    (re.compile(r"(?:请|帮我)?记住[，,：:\s]*(.{2,60}?)(?:[。！!？?]|$)"), "fact", 0.9, "用户要求记住：{0}"),
    (re.compile(r"我(?:的名字)?(?:叫|是)([一-龥A-Za-z]{2,10}?)(?:[，,。！!]|$)"), "profile", 0.8, "用户叫{0}"),
    (re.compile(r"(?:我|^|[，,。；;])(?:在|是)([一-龥A-Za-z]{1,12}?(?:部门|部|组|中心|团队|事业部))"), "profile", 0.8, "用户在{0}"),
    (re.compile(r"我(?:是|作为)(?:一名|一个)?((?:新|老)?(?:员工|经理|主管|实习生|工程师|销售|运维|财务|HR|产品经理))"), "profile", 0.7, "用户是{0}"),
    (re.compile(r"我(?:比较|更)?(?:喜欢|偏好|习惯|希望|倾向于?)(.{2,40}?)(?:[。！!？?]|$)"), "preference", 0.7, "用户偏好：{0}"),
    (re.compile(r"(?:以后|下次|今后)(?:回答|回复)?(?:请|都)?(.{2,40}?)(?:[。！!？?]|$)"), "preference", 0.75, "用户希望以后{0}"),
    (re.compile(r"我(?:正在|在做|负责|准备)(.{2,40}?)(?:[。！!？?]|$)"), "task", 0.6, "用户正在{0}"),
]


def extract_by_rules(user_text: str) -> list[MemoryCandidate]:
    out: list[MemoryCandidate] = []
    for pattern, kind, importance, template in _RULES:
        for m in pattern.finditer(user_text or ""):
            value = m.group(1).strip(" ，,：:")
            if len(value) >= 2:
                out.append(MemoryCandidate(template.format(value), kind, importance))
    seen: set[str] = set()
    return [c for c in out if not (c.content in seen or seen.add(c.content))]


def extract_with_llm(llm, user_text: str, assistant_text: str = "") -> list[MemoryCandidate]:
    raw = llm.complete(EXTRACT_PROMPT.format(user=user_text[:1500], assistant=(assistant_text or "")[:800]), max_tokens=400)
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        return []
    out = []
    for item in json.loads(m.group(0)):
        if not isinstance(item, dict):
            continue
        # 模型偶尔不按字段名来（实测 deepseek-flash 写成 "text"）：常见同义键都认
        content = next((str(item[k]).strip() for k in ("content", "text", "memory", "fact", "statement") if str(item.get(k) or "").strip()), "")
        if content:
            kind = str(item.get("kind") or item.get("type") or "fact")
            out.append(MemoryCandidate(content.rstrip("。"), kind if kind in ("profile", "preference", "fact", "task") else "fact", float(item.get("importance") or 0.5)))
    return out


def extract_memories(user_text: str, assistant_text: str = "", *, llm=None, mode: str = "llm") -> list[MemoryCandidate]:
    """mode=llm 且有模型时用 LLM，失败或无模型时退回规则；mode=rule 只用规则；off 不抽取。"""
    if mode == "off" or not (user_text or "").strip():
        return []
    if mode == "llm" and llm is not None:
        try:
            return extract_with_llm(llm, user_text, assistant_text)
        except Exception as exc:
            logger.info("LLM 记忆抽取失败，改用规则：%s", exc)
    return extract_by_rules(user_text)
