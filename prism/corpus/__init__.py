"""语料优化与语料增强：质量体检与治理、覆盖缺口驱动的 FAQ 增强、关键词 / 别名扩展、自动元数据抽取。"""

from .aliases import AliasDict, discover_aliases
from .augment import build_faq_markdown, draft_faq
from .health import CorpusHealthChecker, exclusion_key, fingerprint
from .keywords import extract_keywords
from .metadata import apply_metadata, extract_metadata

__all__ = ["AliasDict", "discover_aliases", "build_faq_markdown", "draft_faq", "CorpusHealthChecker", "exclusion_key", "fingerprint", "extract_keywords", "apply_metadata", "extract_metadata"]
