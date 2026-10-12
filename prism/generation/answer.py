"""回答生成：检索结果 → 上下文拼装 → LLM（流式 / 非流式）→ 引用解析 / 拒答。"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

from ..config import GenerationConfig
from ..schemas import Answer, Context, RetrievalResult
from .citecheck import strip_citations
from .llm import BaseLLM
from .prompts import ANSWER_SYSTEM, MULTIMODAL_SYSTEM, REFUSAL_TEXT, build_answer_prompt

_CITE = re.compile(r"\[(\d+)\]")
_IMAGE_CITE = re.compile(r"\[图\s*(\d+)\]")


class AnswerGenerator:
    def __init__(self, cfg: GenerationConfig, llm: BaseLLM | None) -> None:
        self.cfg = cfg
        self.llm = llm

    def build_messages(
        self,
        result: RetrievalResult,
        *,
        history: list[dict[str, str]] | None = None,
        image_b64: str | None = None,
        extra_context: str | None = None,
        images: list[Any] | None = None,
        image_payloads: list[str] | None = None,
        query_image_note: str | None = None,
        reference_notes: str | None = None,
        memories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """images：检索到的图片（ImageHit，[图n] 顺序，描述文本进提示词）；
        image_payloads：其中前 N 张的 data URL，交给看得了图的模型；image_b64：用户上传的查询图片。"""
        vision = self.llm is not None and getattr(self.llm, "supports_vision", False)
        payloads = list(image_payloads or []) if vision else []
        query_image = image_b64 if vision else None
        attached = (["用户图片"] if query_image else []) + [f"[图{i + 1}]" for i in range(len(payloads))]
        prompt = build_answer_prompt(result.query, result.contexts, extra_context=extra_context, images=images, attached=attached if (payloads or (query_image and images)) else None, query_image_note=query_image_note if not query_image else None, reference_notes=reference_notes, memories=memories)
        msgs: list[dict[str, Any]] = []
        for m in (history or [])[-8:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                # 上一轮回答里的 [n] / [图n] 指的是上一轮的资料：留着模型会把它们原样带进本轮，造成"串号"
                content = strip_citations(str(m["content"])) if m["role"] == "assistant" else str(m["content"])
                msgs.append({"role": m["role"], "content": content})
        user = self.llm.user_message(prompt, query_image, payloads) if self.llm is not None else {"role": "user", "content": prompt}
        msgs.append(user)
        return msgs

    @staticmethod
    def system_prompt(multimodal: bool) -> str:
        return ANSWER_SYSTEM + (MULTIMODAL_SYSTEM if multimodal else "")

    def thinking_for(self, result: RetrievalResult) -> bool:
        """回答前要不要让推理模型先思考（generation.llm.thinking）。auto：普通问答不思考——资料已经检索、精排好，
        思考只让首字晚 2.5~4.5 秒（deepseek-v4-pro 实测，回答同样正确并带引用）；多跳 / Agentic 这类要跨资料推理的模式才思考。
        另外两类也思考：问题要按日期判断（"去年没休完的年假今年还能休吗""1 月攒的调休现在还能用吗"）、
        或问的是历史规定（检索放行了已失效文档）。不思考时模型先写出"能。"再推算出"已过期，不能再用"，
        回答自相矛盾；思考后回答评测里这类题全部答对（多花 5~15 秒，只影响这类问题）。"""
        mode = self.cfg.llm.thinking
        if mode != "auto":
            return mode == "on"
        strategy = result.trace.strategy or ""
        if strategy.startswith(("multihop", "agentic")):         # 带图提问时是 multihop+image
            return True
        return bool(needs_date_reasoning(result.query) or any(n.startswith("时效") for n in result.trace.notes))

    def answer(self, result: RetrievalResult, *, history: list[dict[str, str]] | None = None, image_b64: str | None = None, extra_context: str | None = None, images: list[Any] | None = None, image_payloads: list[str] | None = None, query_image_note: str | None = None, reference_notes: str | None = None, memories: list[str] | None = None) -> Answer:
        contexts = result.contexts
        if not contexts and not extra_context and not images and not image_b64 and self.cfg.refuse_when_no_context:
            return Answer(query=result.query, text=REFUSAL_TEXT, refused=True, contexts=[], trace=result.trace)

        if self.llm is None:
            return Answer(query=result.query, text=extractive_answer(contexts, images), refused=False, citations=list(range(1, min(3, len(contexts)) + 1)), contexts=contexts, trace=result.trace, model="extractive")

        msgs = self.build_messages(result, history=history, image_b64=image_b64, extra_context=extra_context, images=images, image_payloads=image_payloads, query_image_note=query_image_note, reference_notes=reference_notes, memories=memories)
        resp = self.llm.chat(msgs, system=self.system_prompt(bool(images or image_b64)), thinking=self.thinking_for(result))
        text = resp.text.strip()
        refused = resp.refused or looks_like_refusal(text)
        return Answer(query=result.query, text=text, refused=refused, citations=parse_citations(text, len(contexts)), contexts=contexts, trace=result.trace, model=resp.model, usage=resp.usage)

    def stream(self, result: RetrievalResult, *, history: list[dict[str, str]] | None = None, image_b64: str | None = None, extra_context: str | None = None, images: list[Any] | None = None, image_payloads: list[str] | None = None, query_image_note: str | None = None, reference_notes: str | None = None, memories: list[str] | None = None) -> Iterator[str]:
        """流式增量文本；无 LLM 时分片输出抽取式回答（离线体验模式）。"""
        contexts = result.contexts
        if not contexts and not extra_context and not images and not image_b64 and self.cfg.refuse_when_no_context:
            yield REFUSAL_TEXT
            return
        if self.llm is None:
            text = extractive_answer(contexts, images)
            for i in range(0, len(text), 12):
                yield text[i:i + 12]
            return
        msgs = self.build_messages(result, history=history, image_b64=image_b64, extra_context=extra_context, images=images, image_payloads=image_payloads, query_image_note=query_image_note, reference_notes=reference_notes, memories=memories)
        yield from self.llm.stream(msgs, system=self.system_prompt(bool(images or image_b64)), thinking=self.thinking_for(result))


# 要结合当前日期推算的问题：有时间锚点（相对时间、具体年月、"现在还能"），而不只是问期限多长
_DATE_REASONING = re.compile(
    r"去年|前年|今年|明年|上个?月|下个?月|本月|这个月|上周|下周|上季度|下季度|昨天|前天|明天|"
    r"(?:19|20)\d{2}\s*年\s*\d{1,2}\s*月|(?:现在|目前|如今)(?:还|仍)(?:能|可以|有效|来得及)"
)


def needs_date_reasoning(query: str) -> str | None:
    """问题需要结合当前日期 / 期限推算才能回答 → 返回命中的说法；否则 None。"""
    m = _DATE_REASONING.search(query or "")
    return m.group(0) if m else None


def parse_citations(text: str, n_contexts: int) -> list[int]:
    return sorted({int(n) for n in _CITE.findall(text) if 1 <= int(n) <= n_contexts})


def parse_image_citations(text: str, n_images: int) -> list[int]:
    return sorted({int(n) for n in _IMAGE_CITE.findall(text) if 1 <= int(n) <= n_images})


def extractive_answer(contexts: list[Context], images: list[Any] | None = None) -> str:
    if not contexts and not images:
        return REFUSAL_TEXT
    if not contexts:
        return "\n".join(["### 离线检索模式（未配置 LLM）", "", "没有检索到相关的文字资料；以下是检索到的相关图片：", ""] + _image_lines(images or [])).strip()
    lines = ["### 离线检索模式（未配置 LLM）", "", "以下是检索链路（双路召回 → RRF 融合 → 精排 → Small-to-Big）给出的最相关资料，配置并激活文本模型后即可获得生成式回答：", ""]
    # 上下文按 Lost-in-the-Middle 排列，这里按分数取前 3 并保留原编号，保证 [n] 与引用面板一致
    ranked = sorted(enumerate(contexts, start=1), key=lambda x: x[1].score, reverse=True)[:3]
    # 只展示与最优资料同一量级的条目：弱相关内容凑数只会误导读者
    best = ranked[0][1].score
    ranked = [x for x in ranked if x[1].weaved or x[1].score >= 0.5 * best] or ranked[:1]
    for i, ctx in ranked:
        lines.append(f"**[{i}] {ctx.citation_label()}**（score={ctx.score:.3f}）")
        lines.append("")
        lines.append(ctx.text_for_llm().strip()[:600])
        lines.append("")
    if images:
        lines += ["**相关图片**", ""] + _image_lines(images)
    return "\n".join(lines).strip()


def _image_lines(images: list[Any]) -> list[str]:
    return [f"- [图{i}] {hit.record.label()}：{hit.record.brief(120)}" for i, hit in enumerate(images, start=1)]


def looks_like_refusal(text: str) -> bool:
    return any(k in text[:80] for k in ("无法回答", "没有足够信息", "未找到相关", "资料中没有"))
