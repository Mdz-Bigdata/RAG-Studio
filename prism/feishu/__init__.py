"""飞书文档接入：开放平台客户端、文档块 ⇄ Markdown / HTML、文档智能体（增删改查与分析）。"""

from .agent import FeishuAgent, Tool
from .client import DocRef, FeishuClient, FeishuError, parse_ref
from .convert import blocks_to_markdown, markdown_to_blocks
from .render import blocks_to_html

__all__ = ["DocRef", "FeishuAgent", "FeishuClient", "FeishuError", "Tool", "blocks_to_html", "blocks_to_markdown", "markdown_to_blocks", "parse_ref"]
