"""文档元数据的列式索引：时效 / 权限 / 元数据条件的过滤与取值统计，用 numpy 一次算完。

亿级段落的知识库约有几百万篇文档。逐篇拿 DocMetadata 判断：100 万篇实测过滤一遍 0.2 s、统计取值（自查询要用）1.5 s，
DocMetadata 对象本身就占 1.8 GB——500 万篇时每种新条件的第一次查询要等 8 秒，内存 9 GB。
这里每篇文档一行：
- 单值字段一列一个数组：生效 / 失效日期（序数）、权限级别、部门编码、文件类型编码、extra.year；
- 多值字段（标签、标题 / 版本号里的年份、允许访问的部门）用两条平铺数组：行号 + 取值编码；
- 增量维护：新增 / 修改追加一行，旧行标记删除；删除行占到 1/4 时整体压缩。

语义与逐篇判断完全一致（metadata_filter.matches / facets、DocMetadata.is_effective / is_visible_to、temporal 的历史年份），
tests/test_scale.py 里用随机元数据逐项对拍。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

import numpy as np

from ..schemas import ACCESS_ADMIN_LEVEL, DocMetadata

_FIELDS = ("department", "tags", "year", "file_format")


def _title_years(md: DocMetadata) -> set[int]:
    from ..retrieval.temporal import _YEAR

    return {int(y) for y in _YEAR.findall(" ".join(x for x in (md.version or "", md.title or "") if x))}


class _Vocab:
    """取值 → 编码。key 用 casefold（与 metadata_filter 的比较方式一致），display 保留第一次见到的写法。"""

    def __init__(self, fold: bool = True) -> None:
        self.code: dict[Any, int] = {}
        self.display: list[Any] = []
        self.fold = fold

    def key(self, v: Any) -> Any:
        return v.casefold() if self.fold and isinstance(v, str) else v

    def get(self, v: Any) -> int:
        k = self.key(v)
        c = self.code.get(k)
        if c is None:
            c = self.code[k] = len(self.display)
            self.display.append(v)
        return c

    def lookup(self, v: Any) -> int | None:
        return self.code.get(self.key(v))


class _Multi:
    """多值字段：平铺的 (行号, 编码)。"""

    def __init__(self) -> None:
        self.rows = np.zeros(0, np.int32)
        self.codes = np.zeros(0, np.int32)
        self._buf_rows: list[int] = []
        self._buf_codes: list[int] = []

    def add(self, row: int, codes: Iterable[int]) -> None:
        for c in codes:
            self._buf_rows.append(row)
            self._buf_codes.append(c)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self._buf_rows:
            self.rows = np.concatenate([self.rows, np.asarray(self._buf_rows, np.int32)])
            self.codes = np.concatenate([self.codes, np.asarray(self._buf_codes, np.int32)])
            self._buf_rows, self._buf_codes = [], []
        return self.rows, self.codes

    def rows_with(self, n: int, codes: Iterable[int]) -> np.ndarray:
        rows, cs = self.arrays()
        mask = np.zeros(n, bool)
        want = np.fromiter(codes, np.int32)
        if len(want):
            mask[rows[np.isin(cs, want)]] = True
        return mask

    def has_any(self, n: int) -> np.ndarray:
        rows, _ = self.arrays()
        mask = np.zeros(n, bool)
        mask[rows] = True
        return mask

    def counts(self, row_mask: np.ndarray, n_codes: int) -> np.ndarray:
        rows, cs = self.arrays()
        return np.bincount(cs[row_mask[rows]], minlength=n_codes)

    def keep(self, old_to_new: np.ndarray) -> None:
        rows, cs = self.arrays()
        new = old_to_new[rows]
        ok = new >= 0
        self.rows, self.codes = new[ok].astype(np.int32), cs[ok]


class DocIndex:
    _COLS = (("vf", np.int32), ("vt", np.int32), ("access", np.int16), ("dept", np.int32), ("fmt", np.int32), ("xyear", np.int16), ("alive", bool))

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.row: dict[str, int] = {}
        self.n = 0
        self._cap = 0
        for name, dt in self._COLS:
            setattr(self, name, np.zeros(0, dt))
        self.depts, self.fmts, self.tag_vocab, self.allowed_vocab = _Vocab(), _Vocab(), _Vocab(), _Vocab(fold=False)
        self.tags, self.tyears, self.allowed = _Multi(), _Multi(), _Multi()
        self.dead = 0

    @classmethod
    def build(cls, items: Iterable[tuple[str, DocMetadata]]) -> "DocIndex":
        idx = cls()
        for doc_id, md in items:
            idx.upsert(doc_id, md)
        return idx

    def __len__(self) -> int:
        return self.n - self.dead

    # --- 写 -------------------------------------------------------------------------------
    def _grow(self) -> None:
        if self.n < self._cap:
            return
        self._cap = max(1024, self._cap * 2)
        for name, dt in self._COLS:
            arr = getattr(self, name)
            new = np.zeros(self._cap, dt)
            new[: len(arr)] = arr
            setattr(self, name, new)

    def upsert(self, doc_id: str, md: DocMetadata) -> None:
        self.remove(doc_id)
        self._grow()
        r = self.n
        self.n += 1
        self.ids.append(doc_id)
        self.row[doc_id] = r
        self.vf[r] = md.valid_from.toordinal() if md.valid_from else 0
        self.vt[r] = md.valid_to.toordinal() if md.valid_to else 0
        self.access[r] = int(md.access_level or 0)
        dept = md.department.strip() if md.department and md.department.strip() else None
        self.dept[r] = self.depts.get(dept) if dept else -1
        fmt = md.file_format.strip().lower() if md.file_format and md.file_format.strip() else None
        self.fmt[r] = self.fmts.get(fmt) if fmt else -1
        extra = md.extra.get("year")
        self.xyear[r] = int(extra) if isinstance(extra, (int, str)) and str(extra).isdigit() else 0
        self.alive[r] = True
        tags = {self.tag_vocab.key(t.strip()): t.strip() for t in md.tags if isinstance(t, str) and t.strip()}
        self.tags.add(r, (self.tag_vocab.get(t) for t in tags.values()))
        self.tyears.add(r, _title_years(md))
        self.allowed.add(r, {self.allowed_vocab.get(d) for d in md.allowed_departments})

    def remove(self, doc_id: str) -> None:
        r = self.row.pop(doc_id, None)
        if r is None:
            return
        self.alive[r] = False
        self.dead += 1
        if self.dead > 1024 and self.dead * 4 > self.n:
            self._compact()

    def _compact(self) -> None:
        keep = np.flatnonzero(self.alive[: self.n])
        old_to_new = np.full(self.n, -1, np.int64)
        old_to_new[keep] = np.arange(len(keep))
        for name, _ in self._COLS:
            arr = getattr(self, name)
            setattr(self, name, arr[keep].copy())
        self._cap = self.n = len(keep)
        self.ids = [self.ids[i] for i in keep]
        self.row = {d: i for i, d in enumerate(self.ids)}
        for m in (self.tags, self.tyears, self.allowed):
            m.keep(old_to_new)
        self.dead = 0

    # --- 判断 -----------------------------------------------------------------------------
    def live(self) -> np.ndarray:
        return self.alive[: self.n].copy()

    def effective(self, on: date | None = None) -> np.ndarray:
        """DocMetadata.is_effective：生效日之前、失效日之后都不算。"""
        d = (on or date.today()).toordinal()
        vf, vt = self.vf[: self.n], self.vt[: self.n]
        return self.live() & ((vf == 0) | (d >= vf)) & ((vt == 0) | (d <= vt))

    def visible(self, user: dict[str, Any] | None) -> np.ndarray:
        """DocMetadata.is_visible_to：级别够，且（不限部门 / 属于允许的部门 / 最高级别）。"""
        mask = self.live()
        if user is None:
            return mask
        level = int(user.get("level", 0) or 0)
        mask &= self.access[: self.n] <= level
        if level < ACCESS_ADMIN_LEVEL:
            restricted = self.allowed.has_any(self.n)
            code = self.allowed_vocab.lookup(str(user.get("department") or ""))
            ok = self.allowed.rows_with(self.n, [code]) if code is not None else np.zeros(self.n, bool)
            mask &= ~restricted | ok
        return mask

    def _year_rows(self, year: int, today: date) -> np.ndarray:
        """metadata_filter.doc_years 里含这个年份的文档：标题 / 版本号、extra.year、生效区间（没写失效日的算到今年）。"""
        n = self.n
        hit = self.tyears.rows_with(n, [year]) | (self.xyear[:n] == year)
        vf, vt = self.vf[:n], self.vt[:n]
        vfy, vty = _years(vf), _years(vt)
        end = np.where(vt > 0, vty, np.maximum(vfy, today.year))
        hit |= (vf > 0) & (vfy <= year) & (year <= end)
        hit |= (vf == 0) & (vt > 0) & (vty == year)
        return hit

    def _has_year(self) -> np.ndarray:
        n = self.n
        return self.tyears.has_any(n) | (self.xyear[:n] > 0) | (self.vf[:n] > 0) | (self.vt[:n] > 0)

    def match(self, conditions: dict[str, list[Any]], *, include_missing: bool, today: date | None = None) -> np.ndarray:
        """metadata_filter.matches：字段之间"且"，同一字段多个取值"或"；include_missing：没填该属性算不算匹配。"""
        today = today or date.today()
        n = self.n
        mask = self.live()
        for field, want in conditions.items():
            if field not in _FIELDS or not want:
                continue
            if field == "department":
                have, hit = self.dept[:n] >= 0, np.isin(self.dept[:n], [c for c in map(self.depts.lookup, want) if c is not None])
            elif field == "file_format":
                have, hit = self.fmt[:n] >= 0, np.isin(self.fmt[:n], [c for c in map(self.fmts.lookup, want) if c is not None])
            elif field == "tags":
                have, hit = self.tags.has_any(n), self.tags.rows_with(n, [c for c in map(self.tag_vocab.lookup, want) if c is not None])
            else:
                have, hit = self._has_year(), np.zeros(n, bool)
                for y in want:
                    if str(y).isdigit():
                        hit |= self._year_rows(int(y), today)
            mask &= (hit | ~have) if include_missing else hit
        return mask

    def facets(self, rows: np.ndarray, today: date | None = None) -> dict[str, dict[Any, int]]:
        """metadata_filter.facets：每个属性在这批文档里有哪些值、各几篇，按文档数从多到少。
        唯一的差别在显示写法：大小写不同的同一取值（Cloud / cloud）逐篇统计显示子集里第一次见到的写法，这里显示整库第一次
        见到的写法——匹配本来就不分大小写，只影响给模型 / 界面看的列表。"""
        today = today or date.today()
        n = self.n
        rows = rows & self.live()
        out: dict[str, dict[Any, int]] = {}

        def ranked(pairs: Iterable[tuple[Any, Any, int]]) -> dict[Any, int]:   # (排序键, 显示值, 篇数)
            return {shown: c for key, shown, c in sorted(((k, s, c) for k, s, c in pairs if c), key=lambda x: (-x[2], str(x[0])))}

        def single(col: np.ndarray, vocab: _Vocab) -> dict[Any, int]:
            sel = col[:n][rows]
            counts = np.bincount(sel[sel >= 0], minlength=len(vocab.display))
            return ranked((vocab.key(vocab.display[c]), vocab.display[c], int(counts[c])) for c in range(len(vocab.display)))

        out["department"] = single(self.dept, self.depts)
        tag_counts = self.tags.counts(rows, len(self.tag_vocab.display))
        out["tags"] = ranked((self.tag_vocab.key(t), t, int(tag_counts[c])) for c, t in enumerate(self.tag_vocab.display))
        # 年份：一次算完，不按候选年份逐个扫。每篇文档的年份 = 标题 / 版本号年份 ∪ extra.year ∪ 生效区间 ∪（只有失效日期时）失效年。
        # 区间用差分数组计数（起点年 +1、终点年后 -1，前缀和）；散点先按 (行, 年) 去重，再去掉落在本行区间里的（区间已经数过）
        vf, vt = self.vf[:n], self.vt[:n]
        vfy, vty = _years(vf), _years(vt)
        end = np.where(vt > 0, vty, np.maximum(vfy, today.year))
        has_range = rows & (vf > 0) & (end >= vfy)
        pr, pc = self.tyears.arrays()
        sel = rows[pr]
        xr = np.flatnonzero(rows & (self.xyear[:n] > 0))
        vr = np.flatnonzero(rows & (vf == 0) & (vt > 0))
        prow = np.concatenate([pr[sel], xr, vr]).astype(np.int64)
        pyear = np.concatenate([pc[sel], self.xyear[:n][xr], vty[vr]]).astype(np.int64)
        years: dict[int, int] = {}
        if len(prow):
            key = np.unique(prow * 10000 + pyear)
            prow, pyear = key // 10000, key % 10000
            inside = has_range[prow] & (vfy[prow] <= pyear) & (pyear <= end[prow])
            for y, c in zip(*np.unique(pyear[~inside], return_counts=True)):
                years[int(y)] = years.get(int(y), 0) + int(c)
        lo_y, hi_y = vfy[has_range], end[has_range]
        if len(lo_y):
            base = int(lo_y.min())
            diff = np.zeros(int(hi_y.max()) - base + 2, np.int64)
            np.add.at(diff, lo_y - base, 1)
            np.add.at(diff, hi_y - base + 1, -1)
            for i, c in enumerate(np.cumsum(diff)[:-1]):
                if c:
                    years[base + i] = years.get(base + i, 0) + int(c)
        out["year"] = ranked((y, y, c) for y, c in years.items())
        out["file_format"] = single(self.fmt, self.fmts)
        return {f: out[f] for f in _FIELDS}

    def history_year(self, years: set[int], on: date | None = None) -> int | None:
        """temporal.history_intent 的年份判断：问题里的年份落在某份已失效文档的标题 / 版本号 / 生效区间里。"""
        n = self.n
        expired = self.live() & ~self.effective(on)
        if not expired.any():
            return None
        vf, vt = self.vf[:n], self.vt[:n]
        vfy, vty = _years(vf), _years(vt)
        for y in sorted(years):
            hit = self.tyears.rows_with(n, [y])
            hit |= (vf > 0) & (vt > 0) & (vfy <= y) & (y <= vty)
            hit |= (vf == 0) & (vt > 0) & (vty == y)
            if (hit & expired).any():
                return y
        return None

    # --- 取 id ----------------------------------------------------------------------------
    def ids_where(self, mask: np.ndarray) -> list[str]:
        return [self.ids[i] for i in np.flatnonzero(mask[: self.n])]

    def rows_of(self, doc_ids: Iterable[str]) -> np.ndarray:
        mask = np.zeros(self.n, bool)
        rows = [self.row[d] for d in doc_ids if d in self.row]
        mask[rows] = True
        return mask


def _years(ordinals: np.ndarray) -> np.ndarray:
    """日期序数 → 年份（0 → 0）。numpy 的 datetime64 换算比逐个 date.fromordinal 快两个数量级。"""
    epoch = date(1970, 1, 1).toordinal()
    out = (np.asarray(ordinals, np.int64) - epoch).astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64) + 1970
    return np.where(np.asarray(ordinals) > 0, out, 0)
