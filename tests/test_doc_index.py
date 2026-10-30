"""文档元数据列式索引（DocIndex）与逐篇判断逐项对拍：随机元数据上 effective / visible / match / facets / 历史年份完全一致。"""

from __future__ import annotations

import random
from datetime import date, timedelta

from prism.index.doc_index import DocIndex
from prism.retrieval.metadata_filter import facets, matches
from prism.retrieval.temporal import _doc_years
from prism.schemas import DocMetadata

TODAY = date(2026, 10, 30)
DEPTS = ["人事", "财务", "研发", None, "  ", "Sales"]
TAGS = ["制度", "云计算", "安全", "Cloud", "cloud"]


def random_docs(n: int, seed: int) -> dict[str, DocMetadata]:
    rng = random.Random(seed)
    out = {}
    for i in range(n):
        md = DocMetadata(title=rng.choice([f"文档{i}", f"员工手册 {rng.choice([2023, 2024, 2026])} 版", "K8s 手册", f"v{rng.randint(1, 3)}"]),
                         version=rng.choice([None, "2024", "1.2", "2025版"]), file_format=rng.choice(["md", "PDF", "", "  "]),
                         department=rng.choice(DEPTS), tags=rng.sample(TAGS, rng.randint(0, 3)), access_level=rng.choice([0, 0, 1, 2, 3]),
                         allowed_departments=rng.choice([[], [], ["人事"], ["财务", "研发"]]))
        if rng.random() < 0.4:
            md.valid_from = TODAY - timedelta(days=rng.randint(-200, 1500))
        if rng.random() < 0.4:
            md.valid_to = (md.valid_from or TODAY) + timedelta(days=rng.randint(-100, 800))
        if rng.random() < 0.2:
            md.extra["year"] = rng.choice([2022, "2023"])
        out[f"d{i:05d}"] = md
    return out


def _fold(f: dict) -> dict:
    """大小写不同的同一取值（Cloud / cloud）只是显示写法：逐篇统计显示子集里第一次见到的写法，列式索引显示整库第一次见到的写法。"""
    return {k: {(v.casefold() if isinstance(v, str) else v): n for v, n in vals.items()} for k, vals in f.items()}


def check(docs: dict[str, DocMetadata], di: DocIndex) -> None:
    ids = list(docs)
    for on in (TODAY, TODAY - timedelta(days=400), None):
        assert set(di.ids_where(di.effective(on))) == {d for d, md in docs.items() if md.is_effective(on)}
    for user in (None, {"level": 0}, {"level": 1, "department": "人事"}, {"level": 2, "department": "销售"}, {"level": 3}):
        assert set(di.ids_where(di.visible(user))) == {d for d, md in docs.items() if md.is_visible_to(user)}, user
    for cond in ({"department": ["人事"]}, {"department": ["sales"]}, {"tags": ["cloud"]}, {"tags": ["制度", "安全"]}, {"year": [2024]}, {"year": ["2026", 2023]},
                 {"file_format": ["pdf"]}, {"department": ["财务"], "year": [2025]}, {"tags": ["不存在"]}, {"year": [1999]}):
        for inc in (True, False):
            got = set(di.ids_where(di.match(cond, include_missing=inc, today=TODAY)))
            want = {d for d, md in docs.items() if matches(md, cond, include_missing=inc, today=TODAY)}
            assert got == want, (cond, inc, len(got ^ want))
    assert _fold(di.facets(di.live(), TODAY)) == _fold(facets(docs.values(), TODAY))
    half = di.rows_of(ids[::2])
    assert _fold(di.facets(half, TODAY)) == _fold(facets([docs[d] for d in ids[::2]], TODAY))
    for years in ({2024}, {2023, 2026}, {1999}, {2025}):
        want = None
        for md in docs.values():
            if not md.is_effective(TODAY) and {str(y) for y in years} & _doc_years(md):
                want = int(next(iter(sorted({str(y) for y in years} & _doc_years(md)))))
                break
        got = di.history_year(years, TODAY)
        assert (got is None) == (want is None), (years, got, want)


def test_parity_with_per_document_checks():
    docs = random_docs(600, seed=1)
    check(docs, DocIndex.build(docs.items()))


def test_incremental_updates_and_compaction():
    docs = random_docs(300, seed=2)
    di = DocIndex.build(docs.items())
    rng = random.Random(3)
    # 改元数据、删除、再新增；删得够多会触发压缩
    for d in rng.sample(list(docs), 150):
        docs[d] = docs[d].model_copy(update={"access_level": rng.choice([0, 2]), "tags": rng.sample(TAGS, 2), "valid_to": None})
        di.upsert(d, docs[d])
    for d in rng.sample(list(docs), 120):
        docs.pop(d)
        di.remove(d)
    extra = random_docs(2000, seed=4)
    for d, md in extra.items():
        docs["x" + d] = md
        di.upsert("x" + d, md)
    for d in list(docs)[::3][:1200]:
        docs.pop(d)
        di.remove(d)
    assert len(di) == len(docs)
    check(docs, di)
