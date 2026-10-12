"""UnWeaver 实体关系织布机 + GraphRAG-lite。

索引阶段：从块中挖掘"专有名词 → 定义"（正则）以及可选的 LLM 三元组 (实体, 关系, 实体)，存成实体图。
检索阶段：已召回的块提到了实体 E，但定义 E 的块没被召回 → 自动拉取定义块作为"编织节点"并入上下文，
防止语义碎裂带来的上下文缺失（"该费用不予退还"——哪个费用？定义在两页之前）。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from ..schemas import Chunk, Context, WeavedRelation

_DEFINE_PATTERNS = [
    # **RAG** 是指 / 是 / 定义为 / 指 …
    re.compile(r"\*\*([^\*\n]{2,24})\*\*[:：\s]*(?:是指|定义为|指的是|是|指|为)([^。\n]{5,200})"),
    # 「RAG」/“RAG” 是指 …
    re.compile(r"[「“\"]([^」”\"\n]{2,24})[」”\"][:：\s]*(?:是指|定义为|指的是|是|指)([^。\n]{5,200})"),
    # RAG（Retrieval-Augmented Generation）是 …
    re.compile(r"(?<![一-龥A-Za-z])([A-Za-z][A-Za-z0-9\-]{1,30})（([^）\n]{2,60})）[:：\s]*(?:是指|是|指)([^。\n]{5,200})"),
    # X is defined as / refers to …
    re.compile(r"\b([A-Z][A-Za-z0-9\-]{1,30}(?:\s[A-Z][A-Za-z0-9\-]{1,30}){0,3})\s+(?:is defined as|refers to|means)\s+([^.\n]{5,200})"),
]


class EntityRelation(BaseModel):
    entity: str
    relation_type: str = "DEFINED_IN"          # DEFINED_IN / RELATED_TO
    definition: str = ""
    target: str | None = None                  # RELATED_TO 的宾语
    chunk_id: str                              # 定义所在 parent chunk
    doc_id: str
    source: str = "regex"                      # regex / llm
    entity_type: str | None = None             # 实体类型（ENTITY_TYPES 的键），LLM 抽取时给出
    target_type: str | None = None             # RELATED_TO 宾语的类型
    relation_group: str | None = None          # RELATED_TO 的关系类别（RELATION_GROUPS 的键），LLM 抽取时给出


# 实体类型：固定集合，前端按它着色与筛选（键是稳定 id，值是中文名）
ENTITY_TYPES: dict[str, str] = {
    "policy": "制度政策",
    "process": "流程操作",
    "product": "产品服务",
    "system": "系统组件",
    "role": "组织角色",
    "metric": "指标数值",
    "concept": "概念术语",
    "other": "其他",
}
_TYPE_ALIASES = {v: k for k, v in ENTITY_TYPES.items()} | {"制度": "policy", "政策": "policy", "流程": "process", "产品": "product", "系统": "system", "组件": "system", "角色": "role", "组织": "role", "指标": "metric", "数值": "metric", "概念": "concept", "术语": "concept"}


# 关系类别：LLM 抽取时直接给出（关系短语本身是自由文本，"适用于""冲突时让位于"…，按类别筛选才可用）
RELATION_GROUPS: dict[str, str] = {
    "structure": "组成包含",
    "require": "依赖前提",
    "constraint": "适用约束",
    "cause": "因果影响",
    "process": "流程动作",
    "replace": "替代演进",
    "usage": "使用支撑",
    "other": "其他",
}
_GROUP_ALIASES = {v: k for k, v in RELATION_GROUPS.items()} | {"组成": "structure", "包含": "structure", "依赖": "require", "前提": "require", "约束": "constraint", "适用": "constraint", "因果": "cause", "影响": "cause", "流程": "process", "动作": "process", "替代": "replace", "演进": "replace", "使用": "usage", "支撑": "usage"}


def normalize_group(value: object) -> str | None:
    v = str(value or "").strip()
    if not v:
        return None
    if v.lower() in RELATION_GROUPS:
        return v.lower()
    return _GROUP_ALIASES.get(v)


def normalize_type(value: object) -> str | None:
    v = str(value or "").strip()
    if not v:
        return None
    if v.lower() in ENTITY_TYPES:
        return v.lower()
    return _TYPE_ALIASES.get(v)


class EntityStore:
    def __init__(self) -> None:
        self.relations: list[EntityRelation] = []

    def __len__(self) -> int:
        return len(self.relations)

    def add(self, relations: list[EntityRelation]) -> None:
        seen = {(r.entity.lower(), r.chunk_id, r.relation_type, r.target) for r in self.relations}
        for r in relations:
            key = (r.entity.lower(), r.chunk_id, r.relation_type, r.target)
            if key not in seen:
                self.relations.append(r)
                seen.add(key)

    def remove_document(self, doc_id: str) -> None:
        self.relations = [r for r in self.relations if r.doc_id != doc_id]

    def entities(self) -> list[str]:
        return sorted({r.entity for r in self.relations})

    def definitions(self) -> list[EntityRelation]:
        return [r for r in self.relations if r.relation_type == "DEFINED_IN"]

    def mentioned_in(self, text: str) -> list[EntityRelation]:
        low = text.lower()
        return [r for r in self.relations if len(r.entity) >= 2 and r.entity.lower() in low]

    def neighbors(self, entity: str) -> list[EntityRelation]:
        e = entity.lower()
        return [r for r in self.relations if r.relation_type == "RELATED_TO" and (r.entity.lower() == e or (r.target or "").lower() == e)]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "entities.json").write_text(json.dumps([r.model_dump() for r in self.relations], ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "EntityStore":
        store = cls()
        p = directory / "entities.json"
        if p.exists():
            store.relations = [EntityRelation.model_validate(x) for x in json.loads(p.read_text(encoding="utf-8"))]
        return store


def extract_definitions(text: str, chunk_id: str, doc_id: str) -> list[EntityRelation]:
    out: list[EntityRelation] = []
    for pat in _DEFINE_PATTERNS:
        for m in pat.finditer(text):
            groups = [g.strip() for g in m.groups()]
            entity = groups[0]
            definition = groups[-1]
            if len(groups) == 3:
                definition = f"（{groups[1]}）{definition}"
            if len(entity) < 2 or len(definition) < 5 or entity.lower() in _STOP_ENTITIES:
                continue
            out.append(EntityRelation(entity=entity, definition=definition[:400], chunk_id=chunk_id, doc_id=doc_id))
    return out


_STOP_ENTITIES = {"注意", "说明", "备注", "提示", "总结", "结论", "例如", "重要", "本文", "本章", "本节", "以上", "以下", "note", "tip", "warning"}

_LLM_PROMPT = """从下面的文本中抽取知识图谱：关键实体的定义、类型，以及实体之间的关系三元组。只输出 JSON：
{{"definitions": [{{"entity": "术语", "type": "类型", "definition": "定义"}}], "relations": [{{"subject": "实体A", "relation": "关系", "category": "关系类别", "object": "实体B"}}], "types": {{"实体B": "类型"}}}}
要求：
- 实体是文中反复出现、值得单独检索的名词：制度 / 政策 / 流程 / 产品 / 系统 / 组件 / 角色 / 指标 / 专有术语；不要抽"本文""用户""问题"这类泛词
- 实体用文本中的原词，2~20 个字；definition 用一句话概括它在文中的含义或规定（不超过 80 字）
- relation 用 2~6 个字的动词短语（如"包含""适用于""需要""审批""导致""替代"）；subject 与 object 都要是实体
- category 是这条关系的类别，只能取这些英文值之一：structure（组成 / 包含 / 隶属）、require（依赖 / 前提 / 需要）、constraint（适用范围 / 规定 / 限制 / 权益）、cause（因果 / 影响 / 导致 / 避免）、process（流程中的动作：审批 / 申请 / 处理 / 负责）、replace（替代 / 取代 / 升级降级 / 演进）、usage（使用 / 采用 / 搭载 / 基于 / 配置）、other
- type 只能取这些英文值之一：policy（制度 / 政策 / 规定）、process（流程 / 操作 / 步骤）、product（产品 / 服务 / 商品）、system（系统 / 工具 / 组件 / 平台）、role（组织 / 部门 / 角色 / 人员）、metric（指标 / 数值 / 金额 / 时长 / 阈值）、concept（概念 / 术语 / 方法）、other
- relations 里出现、但 definitions 里没有的实体，在 types 里给出类型
- 最多 8 条定义、8 条关系；没有就给空数组

文本：
{text}"""


class LLMEntityExtractor:
    def __init__(self, llm) -> None:
        self.llm = llm

    def extract(self, text: str, chunk_id: str, doc_id: str) -> list[EntityRelation]:
        try:
            raw = self.llm.complete(_LLM_PROMPT.format(text=text[:3000]), max_tokens=800)
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0)) if m else {}
        except Exception:
            return []
        out: list[EntityRelation] = []
        types: dict[str, str] = {}
        raw_types = data.get("types") or {}
        if isinstance(raw_types, dict):
            for k, v in raw_types.items():
                t = normalize_type(v)
                if t:
                    types[str(k).strip()] = t
        for d in data.get("definitions", []) or []:
            e, defn = str(d.get("entity", "")).strip(), str(d.get("definition", "")).strip()
            t = normalize_type(d.get("type"))
            if t:
                types.setdefault(e, t)
            if len(e) >= 2 and len(defn) >= 5:
                out.append(EntityRelation(entity=e, definition=defn[:400], chunk_id=chunk_id, doc_id=doc_id, source="llm", entity_type=t))
        for r in data.get("relations", []) or []:
            s, rel, o = str(r.get("subject", "")).strip(), str(r.get("relation", "")).strip(), str(r.get("object", "")).strip()
            if s and rel and o:
                out.append(EntityRelation(entity=s, relation_type="RELATED_TO", definition=rel, target=o, chunk_id=chunk_id, doc_id=doc_id, source="llm", entity_type=types.get(s), target_type=types.get(o), relation_group=normalize_group(r.get("category"))))
        return out


def weave_contexts(contexts: list[Context], store: EntityStore, get_chunk, *, max_weaved: int = 3, allow: Callable[[Chunk], bool] | None = None) -> tuple[list[Context], list[WeavedRelation]]:
    """对已检索的上下文做实体织入。get_chunk(chunk_id) -> Chunk | None。
    allow(chunk)：织入块也必须通过检索时的元数据过滤（过期 / 权限 / 文档筛选）——实体图是全库的，
    不过滤的话已废止的旧版制度、当前用户无权查看的文档会绕过过滤混进上下文。"""
    if not contexts or len(store) == 0:
        return contexts, []
    present = {c.chunk.chunk_id for c in contexts}
    weaved: list[Context] = []
    relations: list[WeavedRelation] = []
    used_entities: set[str] = set()
    for ctx in contexts:
        for rel in store.mentioned_in(ctx.chunk.text):
            if rel.relation_type != "DEFINED_IN" or rel.chunk_id in present or rel.entity.lower() in used_entities:
                continue
            chunk: Chunk | None = get_chunk(rel.chunk_id)
            if chunk is None or chunk.chunk_id in present or (allow is not None and not allow(chunk)):
                continue
            present.add(rel.chunk_id)
            used_entities.add(rel.entity.lower())
            weaved.append(Context(chunk=chunk, score=0.5, weaved=True, weaved_entity=rel.entity))
            relations.append(WeavedRelation(entity=rel.entity, relation_type=rel.relation_type, definition=rel.definition, source_chunk_id=rel.chunk_id, triggered_by=ctx.chunk.chunk_id))
            if len(weaved) >= max_weaved:
                return contexts + weaved, relations
    return contexts + weaved, relations
