"""结构感知切片器（默认策略）。

流程：
1. 维护标题栈 → 每个块得到面包屑 [文档标题, H1, H2, ...]
2. 同一标题路径下的连续块组成一个 section
3. section 内贪心累积成 parent（≤ parent_max_tokens）：
   - 表格 / 代码 / 公式是原子块，不拆；超长表格按行拆并重复表头
   - 超长段落按策略拆：semantic（相邻句向量距离）或 recursive（段落→句→字符）
   - 过小 parent 与相邻 parent 合并
4. 每个 parent 再切成带重叠的 child（句子窗口），child 继承面包屑与页码

面包屑同时写进 chunk.embed_text()，检索时即使块里没有关键词，标题也能补上语义信号。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...config import ChunkingConfig
from ...schemas import ATOMIC_BLOCK_TYPES, Block, BlockType, Chunk, ParsedDocument
from .parent_child import build_child_texts
from .recursive import recursive_split
from .semantic import semantic_split
from .tokenizer import count_tokens, split_sentences


@dataclass
class _Section:
    breadcrumbs: list[str]
    blocks: list[Block] = field(default_factory=list)


@dataclass
class _Parent:
    breadcrumbs: list[str]
    text: str
    block_types: list[str]
    page_start: int | None
    page_end: int | None
    tokens: int
    atomic: bool = False


class StructuralChunker:
    def __init__(self, cfg: ChunkingConfig, embedder=None) -> None:
        self.cfg = cfg
        self.embedder = embedder
        if cfg.strategy == "semantic" and embedder is None:
            raise ValueError("semantic 切块需要传入 embedder")

    # ----------------------------------------------------------------------------------
    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        sections = self._adopt_orphan_titles(self._build_sections(doc))
        parents: list[_Parent] = []
        for sec in sections:
            parents.extend(self._section_to_parents(sec))
        parents = self._merge_small(parents)
        return self._materialize(doc, parents)

    @staticmethod
    def _adopt_orphan_titles(sections: list[_Section]) -> list[_Section]:
        """只有标题、没有正文的节：
        - 后面紧跟着它的子节 → 标题已经在子节的面包屑里，不用管；
        - 否则它是"孤儿标题"（封面的主标题 / 副标题、被版面模型误判成标题的短行）——直接丢会让这行字
          从索引里消失，所以把它并进后一个有正文的节的开头；文档末尾的孤儿标题并进前一个节的结尾。
        """
        out: list[_Section] = []
        pending: list[str] = []
        for i, sec in enumerate(sections):
            if sec.blocks:
                if pending:
                    lead = Block(type=BlockType.TEXT, text="\n".join(pending), page=sec.blocks[0].page, meta={"orphan_titles": True})
                    sec = _Section(breadcrumbs=sec.breadcrumbs, blocks=[lead] + sec.blocks)
                    pending = []
                out.append(sec)
                continue
            nxt = sections[i + 1] if i + 1 < len(sections) else None
            is_ancestor = nxt is not None and len(nxt.breadcrumbs) > len(sec.breadcrumbs) and nxt.breadcrumbs[: len(sec.breadcrumbs)] == sec.breadcrumbs
            if not is_ancestor:
                pending.append(sec.breadcrumbs[-1])
        if pending:
            tail = Block(type=BlockType.TEXT, text="\n".join(pending), meta={"orphan_titles": True})
            if out:
                last = out[-1]
                tail.page = last.blocks[-1].page
                out[-1] = _Section(breadcrumbs=last.breadcrumbs, blocks=last.blocks + [tail])
            else:
                out.append(_Section(breadcrumbs=sections[-1].breadcrumbs[:1] or ["文档"], blocks=[tail]))
        return out

    # ----------------------------------------------------------------------------------
    def _build_sections(self, doc: ParsedDocument) -> list[_Section]:
        title = doc.title
        stack: list[tuple[int, str]] = []
        sections: list[_Section] = []
        cur: _Section | None = None
        for b in doc.blocks:
            if b.type == BlockType.TITLE:
                level = b.level or 1
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, b.text.strip()))
                # 文档标题通常就是 H1，避免面包屑出现 "手册 > 手册 > 章节"
                cur = _Section(breadcrumbs=_dedupe_consecutive([title] + [t for _, t in stack]))
                sections.append(cur)
                continue
            if not b.is_content():
                continue
            if cur is None:
                cur = _Section(breadcrumbs=[title])
                sections.append(cur)
            cur.blocks.append(b)
        return [s for s in sections if s.blocks or len(s.breadcrumbs) > 1]

    def _section_to_parents(self, sec: _Section) -> list[_Parent]:
        cfg = self.cfg
        parents: list[_Parent] = []
        buf: list[Block] = []
        buf_tokens = 0

        def flush() -> None:
            nonlocal buf, buf_tokens
            if buf:
                parents.append(self._make_parent(sec.breadcrumbs, buf))
            buf, buf_tokens = [], 0

        if not sec.blocks:
            return []  # 纯标题节已在 _adopt_orphan_titles 里处理（并入相邻节或由子节面包屑承载）

        for b in sec.blocks:
            bt = count_tokens(b.text)
            atomic = cfg.respect_atomic_blocks and b.type in ATOMIC_BLOCK_TYPES
            if atomic:
                if bt > cfg.parent_max_tokens:
                    flush()
                    for piece in self._split_atomic(b):
                        parents.append(_Parent(sec.breadcrumbs, piece, [b.type.value], b.page, b.page, count_tokens(piece), atomic=True))
                    continue
                if buf and buf_tokens + bt > cfg.parent_max_tokens:
                    flush()
                buf.append(b)
                buf_tokens += bt
                continue
            if bt > cfg.parent_max_tokens:
                flush()
                for piece in self._split_long_text(b.text):
                    parents.append(_Parent(sec.breadcrumbs, piece, [b.type.value], b.page, b.page, count_tokens(piece)))
                continue
            if buf and buf_tokens + bt > cfg.parent_max_tokens:
                flush()
            buf.append(b)
            buf_tokens += bt
        flush()
        return parents

    def _make_parent(self, breadcrumbs: list[str], blocks: list[Block]) -> _Parent:
        text = "\n\n".join(b.text for b in blocks)
        pages = [b.page for b in blocks if b.page is not None]
        return _Parent(
            breadcrumbs=breadcrumbs,
            text=text,
            block_types=sorted({b.type.value for b in blocks}),
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            tokens=count_tokens(text),
            atomic=len(blocks) == 1 and blocks[0].type in ATOMIC_BLOCK_TYPES,
        )

    def _split_atomic(self, b: Block) -> list[str]:
        if b.type == BlockType.TABLE and b.text.lstrip().startswith("|"):
            return build_child_texts(b.text, self.cfg.table_row_split_tokens, 0.0, [BlockType.TABLE.value])
        if b.type == BlockType.CODE:
            return build_child_texts(b.text, self.cfg.parent_max_tokens, 0.0, [BlockType.CODE.value])
        return recursive_split(b.text, self.cfg.parent_max_tokens)

    def _split_long_text(self, text: str) -> list[str]:
        cfg = self.cfg
        if cfg.strategy == "semantic":
            sentences = split_sentences(text, max_tokens=cfg.parent_max_tokens)
            groups = semantic_split(
                sentences,
                self.embedder,
                percentile=cfg.semantic_breakpoint_percentile,
                min_sentences=cfg.semantic_min_sentences,
                max_tokens=cfg.parent_max_tokens,
            )
            return ["".join(g) if _mostly_cjk(g) else " ".join(g) for g in groups if g]
        return recursive_split(text, cfg.parent_max_tokens)

    def _merge_small(self, parents: list[_Parent]) -> list[_Parent]:
        """同一面包屑下 < parent_min_tokens 的小块并入前一个块（不超过上限、不动原子块）。"""
        out: list[_Parent] = []
        for p in parents:
            if (
                out
                and not p.atomic
                and not out[-1].atomic
                and p.tokens < self.cfg.parent_min_tokens
                and out[-1].breadcrumbs == p.breadcrumbs
                and out[-1].tokens + p.tokens <= self.cfg.parent_max_tokens
            ):
                prev = out[-1]
                prev.text = prev.text + "\n\n" + p.text
                prev.tokens += p.tokens
                prev.block_types = sorted(set(prev.block_types) | set(p.block_types))
                prev.page_end = p.page_end if p.page_end is not None else prev.page_end
                continue
            out.append(p)
        return out

    def _materialize(self, doc: ParsedDocument, parents: list[_Parent]) -> list[Chunk]:
        cfg = self.cfg
        chunks: list[Chunk] = []
        order = 0
        base_meta = {
            "source": doc.metadata.source,
            "title": doc.title,
            "version": doc.metadata.version,
            "valid_from": doc.metadata.valid_from.isoformat() if doc.metadata.valid_from else None,
            "valid_to": doc.metadata.valid_to.isoformat() if doc.metadata.valid_to else None,
            "department": doc.metadata.department,
            "tags": list(doc.metadata.tags),
        }
        for pi, p in enumerate(parents):
            pid = f"{doc.doc_id}:p{pi:04d}"
            parent = Chunk(
                chunk_id=pid,
                doc_id=doc.doc_id,
                kind="parent",
                text=p.text,
                order=order,
                breadcrumbs=p.breadcrumbs,
                page_start=p.page_start,
                page_end=p.page_end,
                token_count=p.tokens,
                block_types=p.block_types,
                metadata=dict(base_meta),
            )
            order += 1
            child_texts = build_child_texts(p.text, cfg.child_tokens, cfg.child_overlap_ratio, p.block_types)
            for ci, ct in enumerate(child_texts):
                cid = f"{pid}:c{ci:03d}"
                parent.child_ids.append(cid)
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        doc_id=doc.doc_id,
                        kind="child",
                        text=ct,
                        parent_id=pid,
                        order=order,
                        breadcrumbs=p.breadcrumbs,
                        page_start=p.page_start,
                        page_end=p.page_end,
                        token_count=count_tokens(ct),
                        block_types=p.block_types,
                        metadata=dict(base_meta),
                    )
                )
                order += 1
            chunks.insert(len(chunks) - len(child_texts), parent)
        return chunks


def _dedupe_consecutive(items: list[str]) -> list[str]:
    out: list[str] = []
    for x in items:
        if not out or out[-1] != x:
            out.append(x)
    return out


def _mostly_cjk(sentences: list[str]) -> bool:
    text = "".join(sentences)
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return cjk >= len(text) * 0.3
