"""知识库结构拓扑：文档 → 章节 → 小节 → 实体，以及实体之间的关系。

结构骨架来自切片的标题路径（breadcrumbs），所以只要有文档就有拓扑；实体与关系来自 UnWeaver
（正则定义 + 可选的 LLM 三元组）。实体还会连到"提到它但不是定义它"的章节——这正是检索时
UnWeaver 会把定义块织入上下文的那些位置。
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .unweaver import ENTITY_TYPES, RELATION_GROUPS, EntityStore

NodeT = dict[str, Any]


def _key(name: str) -> str:
    """实体归一：忽略大小写与空白（「OA 系统」与「OA系统」是同一个实体）。"""
    return re.sub(r"\s+", "", name).lower()


# 规则兜底：没有 LLM 类型（正则定义 / 旧索引）时按名称、再按定义推断。顺序即优先级。
_TYPE_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("metric", re.compile(r"^[\d.]+\s*[a-zA-Z%‰万千元天个次年月日时分秒]*$|[\d.]+\s*(?:天|元|%|ms|秒|小时|分钟|万|千|GB|MB|KB|k|K|倍|次|个月|年)|(?:率|延迟|时长|时延|额度|阈值|上限|下限|指标|售价|价格|金额|数量|库存|QPS|TPS|P\d\d)$", re.I)),
    ("role", re.compile(r"(?:员工|主管|经理|总监|客服|管理员|用户|客户|部门|团队|小组|人员|负责人|审批人|工程师|专员|HR|人力资源部|财务部|行政部|巡检人|审核人|顾问|供应商|领导|老板|同事)$", re.I)),
    ("product", re.compile(r"iPhone|iPad|MacBook|AirPods|Watch|(?:产品|套餐|课程|商品|型号|会员|订阅|机型|配件|教程)$|\b(?:Pro|Max|Air|Plus|mini)\b", re.I)),
    ("system", re.compile(r"(?:系统|平台|工具|服务器|集群|数据库|组件|模块|网关|容器|镜像|缓存|队列|接口|框架|引擎|插件|客户端|后台|控制台|SDK|API)$|\b(?:K8s|Kubernetes|Ingress|Nginx|Pod|Docker|Redis|MySQL|Kafka|Prometheus|Grafana|sshd|xz|liblzma|OA|ERP|CRM|LLM|Agent|Claude|GPT|DeepSeek|WebSocket|FastAPI|LanceDB|Qdrant)\b", re.I)),
    ("process", re.compile(r"(?:流程|申请|审批|报销|退货|退款|换货|处理|操作|步骤|升级|降级|回滚|排查|巡检|复盘|登记|提交|发起|审核|签收|入职|离职|调岗|打卡|调休|加班|出差|部署|发布|迁移|备份|恢复|重启|扩容)$", re.I)),
    ("policy", re.compile(r"(?:制度|政策|规定|规范|条款|原则|铁律|守则|准则|标准|办法|细则|协议|合同|条件|规则|保修|三包|补贴|津贴|福利|年假|病假|事假|婚假|产假|假期|休假)", re.I)),
]


# 关系类别兜底：LLM 抽取时会直接给出类别；没有类别的关系（旧索引 / 未开 LLM 抽取）按关键词归类。顺序即优先级。
_RELATION_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("replace", re.compile(r"替代|取代|降级|升级|迁移|固化为|编译成|路由|超时走|转为|改为|让位于|演进|继承")),
    ("cause", re.compile(r"导致|触发|影响|引起|造成|避免|防止|降低|减少|提升|提高|增加|弥补|缓解|加剧|阻止")),
    ("process", re.compile(r"审批|审核|申请|提交|负责|处理|打款|验收|发布|检查|记录|上传|验证|备案|签收|办理|执行|识别|重建|产出|展示|通知|评估")),
    ("require", re.compile(r"需要|需报|依赖|必须|附上|先于|前提|要求|先跑通|先做到|确保|须")),
    ("constraint", re.compile(r"适用|规定|限制|上限|下限|截止|生效|废止|不符合|无需|不享受|享有|不予|不应|承担|定价|只计算|不做|不为|不承载|保持|明确|有效期|禁止|允许")),
    ("structure", re.compile(r"包含|属于|组成|拆成|下设|涵盖|具有|对应|统一于|分为|构成|隶属")),
    ("usage", re.compile(r"使用|采用|基于|依据|支持|搭载|配置|复用|配合|协同|同步|维护|集成|调用|运行于|部署|放在|表达|方便")),
]


def relation_group(label: str) -> str:
    for g, pat in _RELATION_RULES:
        if pat.search(label or ""):
            return g
    return "other"


def infer_type(name: str, definition: str = "") -> str:
    for t, pat in _TYPE_RULES:
        if pat.search(name):
            return t
    for t, pat in _TYPE_RULES[1:]:                     # 定义里出现数字太常见，不用它判指标
        if definition and pat.search(definition[:40]):
            return t
    return "concept" if definition else "other"


def build_topology(store, entities: EntityStore, *, section_depth: int = 2, max_entities: int = 200, max_mentions: int = 4) -> dict[str, Any]:
    """store: ChunkStore（docs: doc_id → DocMetadata，chunks: chunk_id → Chunk）。"""
    nodes: dict[str, NodeT] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    def node(nid: str, **attrs: Any) -> NodeT:
        if nid not in nodes:
            nodes[nid] = {"id": nid, **attrs}
        return nodes[nid]

    def edge(src: str, dst: str, etype: str, label: str = "") -> None:
        if src == dst:
            return
        key = (src, dst, etype if etype != "relation" else f"relation:{label}")
        if key not in edges:
            edges[key] = {"source": src, "target": dst, "type": etype, "label": label}

    # ---- 结构骨架：文档 → 各级标题（breadcrumbs[0] 是文档标题）
    anchor: dict[str, str] = {}                        # parent chunk_id → 所属最深一级章节节点（或文档节点）
    parents = [c for c in store.chunks.values() if c.kind == "parent"]
    chunk_count: dict[str, int] = defaultdict(int)
    for doc_id, md in store.docs.items():
        node(f"doc:{doc_id}", type="doc", label=md.title or doc_id, docId=doc_id)
    for p in sorted(parents, key=lambda c: (c.doc_id, c.order)):
        doc_node = f"doc:{p.doc_id}"
        if doc_node not in nodes:
            node(doc_node, type="doc", label=p.doc_id, docId=p.doc_id)
        path = [b for b in p.breadcrumbs[1:1 + section_depth] if b and b.strip()]
        prev = doc_node
        for level in range(len(path)):
            nid = f"sec:{p.doc_id}:{'/'.join(path[:level + 1])}"
            node(nid, type="section", label=path[level], docId=p.doc_id, level=level + 1, path=" > ".join(path[:level + 1]), firstChunkId=p.chunk_id)
            edge(prev, nid, "contains")
            prev = nid
        anchor[p.chunk_id] = prev
        chunk_count[prev] += 1

    # ---- 实体：定义所在章节 → 实体；实体 → 实体（三元组关系）
    by_entity: dict[str, list] = defaultdict(list)
    for r in entities.relations:
        by_entity[_key(r.entity)].append(r)
        if r.relation_type == "RELATED_TO" and r.target:
            by_entity.setdefault(_key(r.target), [])
    # 实体太多时按出现次数保留前 max_entities 个，图才看得清
    rank = sorted(by_entity, key=lambda k: (-len(by_entity[k]), k))
    keep = set(rank[:max_entities])
    names: dict[str, str] = {}
    for r in entities.relations:
        names.setdefault(_key(r.entity), r.entity.strip())
        if r.target:
            names.setdefault(_key(r.target), r.target.strip())

    # 实体类型：LLM 给出的类型按出现次数投票（作主语 / 定义时的 entity_type，作宾语时的 target_type）
    votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in entities.relations:
        if r.entity_type in ENTITY_TYPES:
            votes[_key(r.entity)][r.entity_type] += 1
        if r.target and r.target_type in ENTITY_TYPES:
            votes[_key(r.target)][r.target_type] += 1

    def type_of(key: str, defs: list[dict]) -> tuple[str, str]:
        v = votes.get(key)
        if v:
            return max(v.items(), key=lambda kv: (kv[1], kv[0] != "other"))[0], "llm"
        return infer_type(names.get(key, key), defs[0]["text"] if defs else ""), "rule"

    for key in keep:
        rels = by_entity[key]
        defs = [{"text": r.definition, "chunkId": r.chunk_id, "docId": r.doc_id, "source": r.source} for r in rels if r.relation_type == "DEFINED_IN"]
        etype, etype_src = type_of(key, defs)
        node(f"ent:{key}", type="entity", label=names.get(key, key), definitions=defs[:5], sources=sorted({r.source for r in rels}), entityType=etype, entityTypeSource=etype_src)
        for r in rels:
            home = anchor.get(r.chunk_id) or f"doc:{r.doc_id}"
            if home in nodes:
                edge(home, f"ent:{key}", "defines" if r.relation_type == "DEFINED_IN" else "mentions")
            if r.relation_type == "RELATED_TO" and r.target:
                tkey = _key(r.target)
                if tkey in keep:
                    ttype, ttype_src = type_of(tkey, [])
                    node(f"ent:{tkey}", type="entity", label=names.get(tkey, r.target.strip()), definitions=[], sources=[r.source], entityType=ttype, entityTypeSource=ttype_src)
                    edge(f"ent:{key}", f"ent:{tkey}", "relation", r.definition)
                    rel_edge = edges.get((f"ent:{key}", f"ent:{tkey}", f"relation:{r.definition}"))
                    if rel_edge is None:                      # 自指关系（"sshd 重启 sshd"）不画边
                        continue
                    # 关系是从哪些文档抽出来的（同一三元组可能在多篇文档里出现），前端按文档给边着色用
                    if r.doc_id not in rel_edge.setdefault("docIds", []):
                        rel_edge["docIds"].append(r.doc_id)
                    if r.relation_group in RELATION_GROUPS and rel_edge.get("groupSource") != "llm":
                        rel_edge["group"], rel_edge["groupSource"] = r.relation_group, "llm"
                    elif "group" not in rel_edge:
                        rel_edge["group"], rel_edge["groupSource"] = relation_group(r.definition), "rule"

    # ---- 提及：其他章节的正文出现了这个实体（UnWeaver 织入的候选位置）
    if keep:
        texts = [(anchor.get(p.chunk_id), re.sub(r"\s+", "", p.text).lower()) for p in parents]
        for key in keep:
            if len(key) < 2:
                continue
            linked = {e["source"] for e in edges.values() if e["target"] == f"ent:{key}"}
            hits = 0
            for home, text in texts:
                if home and home not in linked and key in text:
                    edge(home, f"ent:{key}", "mentions")
                    linked.add(home)
                    hits += 1
                    if hits >= max_mentions:
                        break

    degree: dict[str, int] = defaultdict(int)
    for e in edges.values():
        degree[e["source"]] += 1
        degree[e["target"]] += 1
    for nid, n in nodes.items():
        n["degree"] = degree[nid]
        if n["type"] != "entity":
            n["chunks"] = chunk_count.get(nid, 0)

    counts: dict[str, int] = defaultdict(int)
    for n in nodes.values():
        counts[n["type"]] += 1
    ecounts: dict[str, int] = defaultdict(int)
    for e in edges.values():
        ecounts[e["type"]] += 1
    type_counts: dict[str, int] = defaultdict(int)
    for n in nodes.values():
        if n["type"] == "entity":
            type_counts[n["entityType"]] += 1
    group_labels: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in edges.values():
        if e["type"] == "relation":
            group_labels[e["group"]][e["label"]] += 1
    group_sources: dict[str, int] = defaultdict(int)
    for e in edges.values():
        if e["type"] == "relation":
            group_sources[e["groupSource"]] += 1
    relation_groups = [
        {"id": g, "label": label, "count": sum(group_labels[g].values()), "examples": [k for k, _ in sorted(group_labels[g].items(), key=lambda kv: -kv[1])[:4]]}
        for g, label in RELATION_GROUPS.items()
    ]
    return {
        "relationGroups": relation_groups,
        "relationGroupSource": dict(group_sources),
        "entityTypes": [{"id": k, "label": v, "count": type_counts.get(k, 0)} for k, v in ENTITY_TYPES.items()],
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
        "stats": {"docs": counts["doc"], "sections": counts["section"], "entities": counts["entity"], "relations": ecounts["relation"], "mentions": ecounts["mentions"], "defines": ecounts["defines"], "totalEntities": len(by_entity), "truncated": len(by_entity) > max_entities},
    }
