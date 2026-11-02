"""实体关系表按实体名索引后，mentioned_in 与原来的逐条子串匹配结果完全一致，且不随关系数线性变慢。"""

from __future__ import annotations

import random
import time
from pathlib import Path

from prism.graph.unweaver import EntityRelation, EntityStore


def naive(store: EntityStore, text: str) -> list[EntityRelation]:
    low = text.lower()
    return [r for r in store.relations if len(r.entity) >= 2 and r.entity.lower() in low]


def rel(entity: str, chunk: str, doc: str = "d1", kind: str = "DEFINED_IN", target: str | None = None) -> EntityRelation:
    return EntityRelation(entity=entity, definition=f"{entity} 的定义", chunk_id=chunk, doc_id=doc, relation_type=kind, target=target)


def test_mentioned_in_matches_naive_scan(tmp_path: Path):
    rng = random.Random(1)
    vocab = ["年假", "Ingress", "SO-8817", "差旅报销", "试用期", "MySQL 1062", "K8s", "ab", "超长实体名" * 8, "通讯补贴标准（远程员工）"]
    store = EntityStore()
    rels = [rel(v, f"d{i % 7}:p{i:04d}", doc=f"d{i % 7}", kind="RELATED_TO" if i % 5 == 0 else "DEFINED_IN", target="X" if i % 5 == 0 else None)
            for i, v in enumerate(vocab * 40)]
    store.add(rels)
    assert len(store) == len({(r.entity.lower(), r.chunk_id, r.relation_type, r.target) for r in rels})   # 去重与原来一致
    texts = ["员工年假可以跨年吗？K8S Ingress 502 怎么排查，so-8817 已发货", "mysql 1062 主键冲突", "无关的句子 ab", "", "差旅报销与通讯补贴标准（远程员工）"]
    for t in texts:
        assert store.mentioned_in(t) == naive(store, t), t
    # 删除文档后索引同步
    store.remove_document("d3")
    assert all(r.doc_id != "d3" for r in store.relations)
    for t in texts:
        assert store.mentioned_in(t) == naive(store, t), t
    # 落盘再读回
    store.save(tmp_path)
    again = EntityStore.load(tmp_path)
    for t in texts:
        assert [r.model_dump() for r in again.mentioned_in(t)] == [r.model_dump() for r in naive(again, t)]
    assert again.neighbors("年假") == [r for r in again.relations if r.relation_type == "RELATED_TO" and r.entity == "年假"]
    rng.shuffle(texts)


def test_mentioned_in_does_not_scale_with_relation_count():
    text = "第三季度各区域销售额同比增长 15%，华东区 450 万；年假可以跨年使用，差旅报销上限每天 600 元。" * 6
    small, big = EntityStore(), EntityStore()
    small.add([rel(f"实体{i}", f"d:p{i}") for i in range(200)] + [rel("年假", "d:x")])
    big.add([rel(f"实体{i}", f"d:p{i}") for i in range(50000)] + [rel("年假", "d:x")])
    assert [r.entity for r in big.mentioned_in(text)] == ["年假"]

    def ms(store: EntityStore) -> float:
        t = time.perf_counter()
        for _ in range(20):
            store.mentioned_in(text)
        return (time.perf_counter() - t) * 1000 / 20

    assert ms(big) < 4 * ms(small) + 5          # 关系数多 250 倍，耗时不应线性增长（原来会）
