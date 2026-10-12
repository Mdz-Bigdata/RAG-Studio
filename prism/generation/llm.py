"""LLM 客户端抽象。

- AnthropicLLM（默认）：官方 `anthropic` SDK，Claude Opus 5.5，自适应思考 + effort 控制深度，流式接收避免超时。
  凭据默认从环境解析（ANTHROPIC_API_KEY / `ant auth login`）；模型配置中心保存的 key 通过 api_key 显式注入。
- OpenAICompatibleLLM：OpenAI 兼容端点（Qwen / DashScope / DeepSeek / SiliconFlow / vLLM / Ollama）——
  MinerU-Popo 的推理服务也是这种形态，方便复用同一套本地 vLLM。
- FakeLLM：测试用，按关键词返回固定文本。

统一接口：
    chat(messages, system, max_tokens) -> LLMResponse      非流式
    stream(messages, system, max_tokens) -> Iterator[str]  流式增量文本
    user_message(text, image_b64) -> dict                  构造（可带图片的）用户消息，屏蔽各家内容块差异
"""

from __future__ import annotations

import base64
import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ..config import LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    stop_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    refused: bool = False


class BaseLLM(ABC):
    name: str = "base"
    model: str = ""
    supports_vision: bool = False

    # thinking：None = 模型默认；False = 本次关闭思考（端点支持时）；True = 允许思考
    @abstractmethod
    def chat(self, messages: list[dict[str, Any]], *, system: str | None = None, max_tokens: int | None = None, thinking: bool | None = None) -> LLMResponse: ...

    def stream(self, messages: list[dict[str, Any]], *, system: str | None = None, max_tokens: int | None = None, thinking: bool | None = None) -> Iterator[str]:
        """默认实现：非流式一次性返回。子类覆盖为真正的增量流。"""
        yield self.chat(messages, system=system, max_tokens=max_tokens, thinking=thinking).text

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> str:
        return self.chat([{"role": "user", "content": prompt}], system=system, max_tokens=max_tokens).text

    def user_message(self, text: str, image_b64: str | None = None, images: list[str] | None = None) -> dict[str, Any]:
        """image_b64：用户随问题上传的图片；images：检索到、需要让模型"亲眼看"的图片（按 [图n] 顺序）。"""
        return {"role": "user", "content": text}


def _all_images(image_b64: str | None, images: list[str] | None) -> list[str]:
    return ([image_b64] if image_b64 else []) + list(images or [])


def _split_data_url(image_b64: str) -> tuple[str, str]:
    """`data:image/png;base64,...` → (media_type, raw_base64)；裸 base64 默认按 png 处理。"""
    m = re.match(r"^data:(image/[\w.+-]+);base64,(.*)$", image_b64.strip(), re.S)
    if m:
        return m.group(1), m.group(2).replace("\n", "")
    raw = image_b64.strip().replace("\n", "")
    try:
        head = base64.b64decode(raw[:32] + "=" * (-len(raw[:32]) % 4), validate=False)
    except Exception:
        head = b""
    if head.startswith(b"\xff\xd8"):
        return "image/jpeg", raw
    if head.startswith(b"RIFF"):
        return "image/webp", raw
    if head.startswith(b"GIF8"):
        return "image/gif", raw
    return "image/png", raw


class AnthropicLLM(BaseLLM):
    name = "anthropic"
    supports_vision = True

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        model: str | None = None,
        effort: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        server_side_fallback: bool = True,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("需要 anthropic SDK：pip install 'rag-agent-studio[llm]'") from exc
        self._anthropic = anthropic
        kwargs: dict[str, Any] = {}
        if api_key or cfg.api_key:
            kwargs["api_key"] = api_key or cfg.api_key
        if base_url or cfg.base_url:
            kwargs["base_url"] = base_url or cfg.base_url
        self.client = anthropic.Anthropic(**kwargs)
        self.model = model or cfg.model
        self.effort = effort or cfg.effort
        self.max_tokens = cfg.max_tokens
        # 服务端 refusal fallback：安全分类器拒答时在同一请求内换模型重跑（按类别自动路由，无需维护模型列表）
        self.server_side_fallback = server_side_fallback

    def user_message(self, text: str, image_b64: str | None = None, images: list[str] | None = None) -> dict[str, Any]:
        pics = _all_images(image_b64, images)
        if not pics:
            return {"role": "user", "content": text}
        blocks: list[dict[str, Any]] = []
        for pic in pics:
            media_type, data = _split_data_url(pic)
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
        return {"role": "user", "content": blocks + [{"type": "text", "text": text}]}

    def _kwargs(self, messages: list[dict[str, Any]], system: str | None, max_tokens: int | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
        )
        if system:
            kwargs["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        return kwargs

    def _open_stream(self, kwargs: dict[str, Any]):
        if self.server_side_fallback:
            return self.client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        return self.client.messages.stream(**kwargs)

    def chat(self, messages: list[dict[str, Any]], *, system: str | None = None, max_tokens: int | None = None, thinking: bool | None = None) -> LLMResponse:
        # thinking 不用：Claude 是自适应思考，深度由 effort 控制
        kwargs = self._kwargs(messages, system, max_tokens)
        try:
            with self._open_stream(kwargs) as stream:
                msg = stream.get_final_message()
        except self._anthropic.RateLimitError as exc:
            raise RuntimeError(f"Anthropic 限流：{exc.message}") from exc
        except self._anthropic.APIStatusError as exc:
            raise RuntimeError(f"Anthropic API 错误 {exc.status_code}：{exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise RuntimeError("无法连接 Anthropic API") from exc

        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        refused = msg.stop_reason == "refusal"
        if msg.stop_reason == "max_tokens":
            logger.warning("回答被 max_tokens 截断（%d）", kwargs["max_tokens"])
        usage = {}
        if getattr(msg, "usage", None) is not None:
            usage = {
                "input_tokens": msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
                "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", None),
            }
        return LLMResponse(text=text, model=msg.model, stop_reason=msg.stop_reason, usage=usage, refused=refused)

    def stream(self, messages: list[dict[str, Any]], *, system: str | None = None, max_tokens: int | None = None, thinking: bool | None = None) -> Iterator[str]:
        kwargs = self._kwargs(messages, system, max_tokens)
        try:
            with self._open_stream(kwargs) as stream:
                for text in stream.text_stream:
                    if text:
                        yield text
        except self._anthropic.RateLimitError as exc:
            raise RuntimeError(f"Anthropic 限流：{exc.message}") from exc
        except self._anthropic.APIStatusError as exc:
            raise RuntimeError(f"Anthropic API 错误 {exc.status_code}：{exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise RuntimeError("无法连接 Anthropic API") from exc


class OpenAICompatibleLLM(BaseLLM):
    name = "openai_compatible"
    supports_vision = True  # 取决于模型；多模态模型（qwen-vl / gpt-4o 等）支持 image_url

    # 推理模型（DeepSeek / Qwen3 等）的思考过程也计入 max_tokens：给改写 / HyDE 这类小任务 120~400 token，
    # 预算会全部耗在思考上，返回空字符串（实测 deepseek-v4-pro：finish_reason=length、content=''）。
    _THINKING_OFF = {"thinking": {"type": "disabled"}}

    def __init__(self, cfg: LLMConfig, *, model: str | None = None, api_key: str | None = None, base_url: str | None = None, aux: bool = False) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("需要 openai 包：pip install 'rag-agent-studio[llm]'") from exc
        key = api_key or cfg.api_key or os.environ.get(cfg.api_key_env) or "EMPTY"
        self.client = OpenAI(base_url=base_url or cfg.base_url, api_key=key)
        self.model = model or cfg.model
        self.max_tokens = cfg.max_tokens
        self.temperature = cfg.temperature
        # 辅助模型（查询改写 / HyDE / 精排 / 压缩 / text-to-SQL / 图片描述）关闭思考：又快又不会被截断成空。
        # 端点不认识该参数时（OpenAI 官方等）自动去掉，之后不再发送。
        self.disable_thinking = aux
        self._thinking_param_ok = True

    def _create(self, msgs: list[dict[str, Any]], max_tokens: int, thinking: bool | None = None, **kw: Any):
        off = self.disable_thinking if thinking is None else not thinking   # 单次调用的 thinking 优先于实例默认
        extra = self._THINKING_OFF if (off and self._thinking_param_ok) else None
        try:
            return self.client.chat.completions.create(model=self.model, messages=msgs, max_tokens=max_tokens, temperature=self.temperature, extra_body=extra, **kw)
        except Exception as exc:
            if extra is not None and getattr(exc, "status_code", None) in (400, 422) and "thinking" in str(exc).lower():
                self._thinking_param_ok = False
                logger.info("端点不支持 thinking 参数，已去掉：%s", exc)
                return self.client.chat.completions.create(model=self.model, messages=msgs, max_tokens=max_tokens, temperature=self.temperature, **kw)
            raise

    def user_message(self, text: str, image_b64: str | None = None, images: list[str] | None = None) -> dict[str, Any]:
        pics = _all_images(image_b64, images)
        if not pics:
            return {"role": "user", "content": text}
        urls = [p if p.startswith("data:") else f"data:image/png;base64,{p}" for p in pics]
        return {"role": "user", "content": [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": u}} for u in urls]}

    def chat(self, messages: list[dict[str, Any]], *, system: str | None = None, max_tokens: int | None = None, thinking: bool | None = None) -> LLMResponse:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        budget = max_tokens or self.max_tokens
        resp = self._create(msgs, budget, thinking)
        choice = resp.choices[0]
        if choice.finish_reason == "length" and not (choice.message.content or "").strip() and _reasoned(resp):
            # 思考把预算用完了还没开始回答：加上思考预算重试一次（关不掉思考的端点的兜底）
            logger.info("推理模型在 %d token 内没有给出回答，放宽预算重试", budget)
            resp = self._create(msgs, budget + 2048, thinking)
            choice = resp.choices[0]
        usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens} if resp.usage else {}
        return LLMResponse(text=choice.message.content or "", model=resp.model, stop_reason=choice.finish_reason, usage=usage)

    def stream(self, messages: list[dict[str, Any]], *, system: str | None = None, max_tokens: int | None = None, thinking: bool | None = None) -> Iterator[str]:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        resp = self._create(msgs, max_tokens or self.max_tokens, thinking, stream=True)
        for chunk in resp:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def _reasoned(resp: Any) -> bool:
    """响应里是否有思考内容（reasoning_content 字段，或 usage 里的 reasoning_tokens）。"""
    msg = resp.choices[0].message
    if getattr(msg, "reasoning_content", None) or (getattr(msg, "model_extra", None) or {}).get("reasoning_content"):
        return True
    details = getattr(resp.usage, "completion_tokens_details", None) if resp.usage is not None else None
    return bool(getattr(details, "reasoning_tokens", 0))


class FakeLLM(BaseLLM):
    """测试 / 离线演示：`responder(prompt) -> str`，默认原样回显最后一行。"""

    name = "fake"
    model = "fake"

    def __init__(self, responder: Callable[[str], str] | None = None, *, vision: bool = False) -> None:
        self.responder = responder or (lambda p: p.strip().splitlines()[-1] if p.strip() else "")
        self.calls: list[str] = []
        self.supports_vision = vision
        self.image_counts: list[int] = []   # 每次调用收到的图片数（测试断言用）
        self.thinking: list[bool | None] = []   # 每次调用的 thinking 参数（测试断言用）

    def user_message(self, text: str, image_b64: str | None = None, images: list[str] | None = None) -> dict[str, Any]:
        pics = _all_images(image_b64, images)
        if not pics:
            return {"role": "user", "content": text}
        return {"role": "user", "content": [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": u}} for u in pics]}

    def chat(self, messages: list[dict[str, Any]], *, system: str | None = None, max_tokens: int | None = None, thinking: bool | None = None) -> LLMResponse:
        self.thinking.append(thinking)
        prompt = messages[-1]["content"] if messages else ""
        if not isinstance(prompt, str):
            self.image_counts.append(sum(1 for p in prompt if isinstance(p, dict) and p.get("type") in ("image", "image_url")))
            prompt = " ".join(p.get("text", "") for p in prompt if isinstance(p, dict))
        else:
            self.image_counts.append(0)
        self.calls.append(prompt)
        return LLMResponse(text=self.responder(prompt), model=self.model, stop_reason="end_turn")

    def stream(self, messages: list[dict[str, Any]], *, system: str | None = None, max_tokens: int | None = None, thinking: bool | None = None) -> Iterator[str]:
        text = self.chat(messages, system=system, max_tokens=max_tokens, thinking=thinking).text
        for i in range(0, len(text), 8):
            yield text[i:i + 8]


def build_llm(cfg: LLMConfig, *, aux: bool = False, api_key: str | None = None, base_url: str | None = None) -> BaseLLM | None:
    """aux=True 时构建辅助任务模型（查询改写 / HyDE 等），默认用更低 effort。"""
    if cfg.provider == "none":
        return None
    if cfg.provider == "anthropic":
        return AnthropicLLM(cfg, model=(cfg.aux_model if aux and cfg.aux_model else None), effort=(cfg.aux_effort if aux else None), api_key=api_key, base_url=base_url)
    if cfg.provider == "openai_compatible":
        return OpenAICompatibleLLM(cfg, model=(cfg.aux_model if aux and cfg.aux_model else None), api_key=api_key, base_url=base_url, aux=aux)
    raise ValueError(f"未知 LLM provider: {cfg.provider}")
