"""检索范围：时效 / 权限 / 元数据过滤的结果，按文档记。

过滤条件几乎都是文档级的（这份文档过没过期、这个人能不能看、属于哪个部门）。原来的做法是展开成"可见段落 id 的集合"：
每次查询都遍历全库段落，再把整个集合交给向量库 / BM25——千万级段落时光展开就要好几秒，下推给向量库的 IN 列表也放不下。

ChunkScope 只记文档：允许哪些（include）或排除哪些（exclude），构造时取两者中小的那份。
- 兼容 set 的只读用法（`cid in scope` / `bool(scope)`）：旧代码照常工作——段落 id 以 doc_id 开头，判断是 O(1)；
- 能下推的后端（LanceDB）用 sql()：变成 doc_id IN / NOT IN 的预过滤，先过滤再算距离，不会被不可见的段落挤掉名额；
- len() / 迭代要展开成段落 id：只给不认识 ChunkScope 的老后端用（numpy / qdrant / chroma，本来就只适合小库）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

PUSHDOWN_LIMIT = 20000     # 文档列表超过这么长就不拼进 SQL（解析慢），改为多取再按 in 过滤


def doc_prefix(chunk_id: str) -> str:
    """段落 id 的文档前缀：<doc_id>:p0001:c002 → <doc_id>。"""
    return chunk_id.split(":", 1)[0]


class ChunkScope:
    __slots__ = ("include", "exclude", "_doc_of", "_expand", "_ids")

    def __init__(self, *, include: frozenset[str] | None = None, exclude: frozenset[str] = frozenset(),
                 doc_of: Callable[[str], str | None] = doc_prefix, expand: Callable[["ChunkScope"], Iterable[str]] | None = None) -> None:
        self.include = include
        self.exclude = exclude if include is None else frozenset()
        self._doc_of = doc_of
        self._expand = expand
        self._ids: frozenset[str] | None = None

    @classmethod
    def of_docs(cls, keep: Iterable[str], universe: Iterable[str], **kw) -> "ChunkScope":
        """keep：允许的文档；universe：全部已知文档。哪份小记哪份（排除模式下，不在 universe 里的文档算允许）。"""
        keep = frozenset(keep)
        drop = frozenset(d for d in universe if d not in keep)
        if not drop:
            return cls(**kw)
        return cls(include=keep, **kw) if len(keep) <= len(drop) else cls(exclude=drop, **kw)

    # --- 判断 ---------------------------------------------------------------------------
    @property
    def unrestricted(self) -> bool:
        return self.include is None and not self.exclude

    @property
    def empty(self) -> bool:
        return self.include is not None and not self.include

    def allows_doc(self, doc_id: str | None) -> bool:
        if self.include is not None:
            return doc_id in self.include
        return doc_id not in self.exclude

    def __contains__(self, chunk_id: object) -> bool:
        return isinstance(chunk_id, str) and self.allows_doc(self._doc_of(chunk_id))

    def __bool__(self) -> bool:
        return not self.empty

    def restrict(self, doc_ids: Iterable[str]) -> "ChunkScope":
        """再收窄到这些文档（自查询识别出的元数据条件）。"""
        want = frozenset(doc_ids)
        keep = want & self.include if self.include is not None else want - self.exclude
        return ChunkScope(include=keep, doc_of=self._doc_of, expand=self._expand)

    def docs_among(self, doc_ids: Iterable[str]) -> list[str]:
        return [d for d in doc_ids if self.allows_doc(d)]

    # --- 下推 ---------------------------------------------------------------------------
    @property
    def pushable(self) -> bool:
        return len(self.include if self.include is not None else self.exclude) <= PUSHDOWN_LIMIT

    def sql(self, column: str = "doc_id") -> str | None:
        """LanceDB / SQL 的 where 条件；不限制时返回 None。"""
        if self.unrestricted:
            return None
        if self.include is not None:
            return f"{column} IN ({_quote(self.include)})" if self.include else "false"
        return f"{column} NOT IN ({_quote(self.exclude)})"

    # --- 展开（老后端）------------------------------------------------------------------
    def chunk_ids(self) -> frozenset[str]:
        if self._ids is None:
            if self._expand is None:
                raise TypeError("这个检索范围不能展开成段落 id（没有提供 expand）")
            self._ids = frozenset(self._expand(self))
        return self._ids

    def __len__(self) -> int:
        return len(self.chunk_ids())

    def __iter__(self) -> Iterator[str]:
        return iter(self.chunk_ids())

    def __repr__(self) -> str:
        if self.unrestricted:
            return "ChunkScope(all)"
        return f"ChunkScope(include={len(self.include)} docs)" if self.include is not None else f"ChunkScope(exclude={len(self.exclude)} docs)"


def _quote(values: Iterable[str]) -> str:
    return ",".join("'" + v.replace("'", "''") + "'" for v in sorted(values))


def is_scope(allowed: object) -> bool:
    return isinstance(allowed, ChunkScope)
