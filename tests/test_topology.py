"""实体织网结构拓扑：文档 → 章节 → 小节骨架、实体归属、关系与提及边、实体名归一。全部离线。"""

from __future__ import annotations

from prism.graph.topology import build_topology
from prism.graph.unweaver import EntityRelation, EntityStore
from prism.index.doc_store import ChunkStore
from prism.schemas import Chunk, DocMetadata


def _store() -> ChunkStore:
    store = ChunkStore()
    store.add_document("d1", DocMetadata(title="员工手册"))
    store.add_chunks([
        Chunk(chunk_id="d1:p0", doc_id="d1", kind="parent", order=0, breadcrumbs=["员工手册", "假期管理", "年假"], text="带薪年假每年 5 天，满 3 年增至 10 天，需在 OA 系统提交申请。"),
        Chunk(chunk_id="d1:p1", doc_id="d1", kind="parent", order=1, breadcrumbs=["员工手册", "假期管理", "调休"], text="调休需直属主管审批。"),
        Chunk(chunk_id="d1:p2", doc_id="d1", kind="parent", order=2, breadcrumbs=["员工手册", "报销"], text="出差前须在OA系统发起申请，未休的带薪年假不折现。"),
    ])
    return store


def _entities() -> EntityStore:
    es = EntityStore()
    es.add([
        EntityRelation(entity="带薪年假", definition="员工每年享有的带薪休假", chunk_id="d1:p0", doc_id="d1", source="llm"),
        EntityRelation(entity="OA 系统", definition="公司的办公审批系统", chunk_id="d1:p0", doc_id="d1", source="llm"),
        EntityRelation(entity="带薪年假", relation_type="RELATED_TO", definition="需要", target="OA系统", chunk_id="d1:p0", doc_id="d1", source="llm"),
    ])
    return es


def test_structure_without_entities():
    g = build_topology(_store(), EntityStore())
    ids = {n["id"] for n in g["nodes"]}
    assert {"doc:d1", "sec:d1:假期管理", "sec:d1:假期管理/年假", "sec:d1:假期管理/调休", "sec:d1:报销"} <= ids
    contains = {(e["source"], e["target"]) for e in g["edges"] if e["type"] == "contains"}
    assert ("doc:d1", "sec:d1:假期管理") in contains and ("sec:d1:假期管理", "sec:d1:假期管理/年假") in contains
    assert g["stats"]["entities"] == 0 and g["stats"]["sections"] == 4


def test_entities_relations_and_mentions():
    g = build_topology(_store(), _entities())
    ents = {n["id"]: n for n in g["nodes"] if n["type"] == "entity"}
    assert set(ents) == {"ent:带薪年假", "ent:oa系统"}                    # 「OA 系统」与「OA系统」归一成一个
    assert ents["ent:带薪年假"]["definitions"][0]["text"] == "员工每年享有的带薪休假"
    edges = {(e["source"], e["target"], e["type"], e["label"]) for e in g["edges"]}
    assert ("sec:d1:假期管理/年假", "ent:带薪年假", "defines", "") in edges
    assert ("ent:带薪年假", "ent:oa系统", "relation", "需要") in edges
    # 「报销」章节提到了两个实体但没有定义它们 → 提及边（UnWeaver 织入的候选位置）
    assert ("sec:d1:报销", "ent:带薪年假", "mentions", "") in edges
    assert ("sec:d1:报销", "ent:oa系统", "mentions", "") in edges
    assert g["stats"]["relations"] == 1


def test_entity_cap():
    es = EntityStore()
    es.add([EntityRelation(entity=f"术语{i}", definition="一个用于测试的定义", chunk_id="d1:p0", doc_id="d1") for i in range(30)])
    g = build_topology(_store(), es, max_entities=10)
    assert g["stats"]["entities"] == 10 and g["stats"]["totalEntities"] == 30 and g["stats"]["truncated"]


def test_entity_types_llm_vote_and_rule_fallback():
    from prism.graph.topology import infer_type
    from prism.graph.unweaver import normalize_type

    es = EntityStore()
    es.add([
        EntityRelation(entity="带薪年假", definition="员工每年享有的带薪休假", chunk_id="d1:p0", doc_id="d1", source="llm", entity_type="policy"),
        EntityRelation(entity="带薪年假", relation_type="RELATED_TO", definition="需要", target="OA系统", chunk_id="d1:p0", doc_id="d1", source="llm", entity_type="policy", target_type="system"),
        EntityRelation(entity="直属主管", definition="负责审批调休的上级", chunk_id="d1:p1", doc_id="d1"),     # 正则抽取，没有类型 → 规则推断
    ])
    g = build_topology(_store(), es)
    ents = {n["label"]: n for n in g["nodes"] if n["type"] == "entity"}
    assert (ents["带薪年假"]["entityType"], ents["带薪年假"]["entityTypeSource"]) == ("policy", "llm")
    assert (ents["OA系统"]["entityType"], ents["OA系统"]["entityTypeSource"]) == ("system", "llm")       # 作宾语时的 target_type
    assert (ents["直属主管"]["entityType"], ents["直属主管"]["entityTypeSource"]) == ("role", "rule")
    counts = {t["id"]: t["count"] for t in g["entityTypes"]}
    assert counts["policy"] == 1 and counts["system"] == 1 and counts["role"] == 1 and len(g["entityTypes"]) == 8

    assert infer_type("P95 延迟") == "metric" and infer_type("1299 元") == "metric"
    assert infer_type("iPhone 15 Pro Max") == "product" and infer_type("Ingress") == "system"
    assert infer_type("退货流程") == "process" and infer_type("差旅报销制度") == "policy"
    assert infer_type("幂等写入", "同一请求重复执行结果不变") == "concept" and infer_type("XYZ") == "other"
    assert normalize_type("制度政策") == "policy" and normalize_type("SYSTEM") == "system" and normalize_type("乱写") is None


def test_relation_groups():
    from prism.graph.topology import relation_group

    cases = {"包含": "structure", "包含于": "structure", "需要": "require", "先于": "require", "适用于": "constraint", "上限为": "constraint",
             "导致": "cause", "避免": "cause", "审批": "process", "可申请": "process", "替代": "replace", "降级到": "replace",
             "冲突时让位于": "replace", "搭载": "usage", "配置于": "usage", "涉及": "other", "": "other"}
    assert {k: relation_group(k) for k in cases} == cases
    g = build_topology(_store(), _entities())
    rel = [e for e in g["edges"] if e["type"] == "relation"]
    assert rel and rel[0]["group"] == "require"                      # 「需要」
    groups = {r["id"]: r for r in g["relationGroups"]}
    assert len(groups) == 8 and groups["require"]["count"] == 1 and groups["require"]["examples"] == ["需要"]


def test_llm_relation_category_parsed_and_preferred():
    import json as _json

    from prism.graph.unweaver import LLMEntityExtractor, normalize_group

    class FakeLLM:
        def complete(self, prompt, **_):
            return _json.dumps({
                "definitions": [{"entity": "带薪年假", "type": "policy", "definition": "员工每年享有的带薪休假"}],
                "relations": [
                    {"subject": "带薪年假", "relation": "冲突时让位于", "category": "constraint", "object": "国家规定"},   # 规则会判成 replace，LLM 说 constraint
                    {"subject": "带薪年假", "relation": "需要", "category": "依赖前提", "object": "OA系统"},          # 中文类别名也认
                    {"subject": "带薪年假", "relation": "涉及", "category": "乱写", "object": "直属主管"},            # 无效类别 → 规则兜底
                ],
                "types": {"国家规定": "policy", "OA系统": "system", "直属主管": "role"},
            }, ensure_ascii=False)

    rels = LLMEntityExtractor(FakeLLM()).extract("……", "d1:p0", "d1")
    groups = {r.target: r.relation_group for r in rels if r.relation_type == "RELATED_TO"}
    assert groups == {"国家规定": "constraint", "OA系统": "require", "直属主管": None}
    assert normalize_group("CAUSE") == "cause" and normalize_group("因果影响") == "cause" and normalize_group("x") is None

    es = EntityStore()
    es.add(rels)
    g = build_topology(_store(), es)
    by_label = {e["label"]: e for e in g["edges"] if e["type"] == "relation"}
    assert (by_label["冲突时让位于"]["group"], by_label["冲突时让位于"]["groupSource"]) == ("constraint", "llm")
    assert (by_label["需要"]["group"], by_label["需要"]["groupSource"]) == ("require", "llm")
    assert (by_label["涉及"]["group"], by_label["涉及"]["groupSource"]) == ("other", "rule")
    assert g["relationGroupSource"] == {"llm": 2, "rule": 1}


def test_self_referencing_relation_is_skipped():
    es = _entities()
    es.add([EntityRelation(entity="sshd", relation_type="RELATED_TO", definition="重启", target="sshd", chunk_id="d1:p0", doc_id="d1", source="llm", relation_group="process")])
    g = build_topology(_store(), es)                                   # 之前这里会 KeyError → 接口 500
    assert not any(e["type"] == "relation" and e["source"] == e["target"] for e in g["edges"])


def test_relation_edges_record_source_docs():
    es = _entities()
    es.add([EntityRelation(entity="带薪年假", relation_type="RELATED_TO", definition="需要", target="OA系统", chunk_id="d2:p0", doc_id="d2", source="llm")])
    g = build_topology(_store(), es)
    rel = [e for e in g["edges"] if e["type"] == "relation"]
    assert len(rel) == 1 and rel[0]["docIds"] == ["d1", "d2"]          # 同一三元组出现在两篇文档：按出现顺序记录
