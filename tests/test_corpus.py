"""语料优化与增强：别名词典、关键词扩展、自动元数据、质量体检、FAQ 草稿。全部离线（哈希向量 + BM25）。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from prism.config import Settings
from prism.corpus import AliasDict, CorpusHealthChecker, build_faq_markdown, discover_aliases, draft_faq, exclusion_key, extract_metadata
from prism.corpus.metadata import apply_metadata
from prism.ingest import IngestPipeline
from prism.retrieval.retriever import HybridRetriever
from prism.schemas import Chunk, Context, DocMetadata

ROOT = Path(__file__).resolve().parents[1]


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt: str, **_) -> str:
        self.prompts.append(prompt)
        return self.reply


def _settings(tmp_path: Path) -> Settings:
    s = Settings.lite(index_dir=tmp_path / "index")
    s.chunking.parent_min_tokens = 0          # 每个小节单独成块，便于断言
    s.retrieval.graph.unweaver_enabled = False
    return s


def _ingest(pipeline: IngestPipeline, doc_id: str, markdown: str, **meta) -> list[Chunk]:
    doc = pipeline.parse_markdown_text(markdown, doc_id=doc_id, source=f"{doc_id}.md", metadata=DocMetadata(source=f"{doc_id}.md", **meta))
    return pipeline.ingest_document(doc)


# --- 别名 -------------------------------------------------------------------------------------
def test_alias_discovery_and_expansion():
    text = "根据服务等级协议（SLA），核心系统可用性不低于 99.95%。Kubernetes（K8s）集群由运维部维护。员工带薪年休假（以下简称“年假”）按工龄计算。"
    found = {tuple(x["terms"]) for x in discover_aliases([text])}
    assert ("服务等级协议", "SLA") in found and ("Kubernetes", "K8s") in found and ("员工带薪年休假", "年假") in found

    d = AliasDict([["K8s", "Kubernetes"], ["SLA", "服务等级协议"], ["x"]])
    assert len(d) == 2                                             # 单词组丢弃
    assert d.expansions("k8s 集群怎么扩容") == ["Kubernetes"]       # 大小写不敏感
    assert d.expansions("please translate this") == []             # 缩写按词边界匹配
    assert discover_aliases([text], existing=d) and all(tuple(x["terms"]) != ("服务等级协议", "SLA") for x in discover_aliases([text], existing=d))


def test_alias_expands_bm25_query_and_document(tmp_path):
    settings = _settings(tmp_path)
    pipe = IngestPipeline(settings, aliases=AliasDict([["K8s", "Kubernetes"]]))
    chunks = _ingest(pipe, "ops", "# 运维手册\n\n## 扩容\n\nKubernetes 集群扩容时先检查节点资源，再调整副本数。\n")
    assert any("K8s" in c.aliases for c in chunks if c.kind == "child")      # 文档扩展
    retriever = HybridRetriever(settings, pipe.index)
    assert retriever.retrieve_sparse("K8s 扩容")                              # 文档扩展后 BM25 能命中

    plain = IngestPipeline(_settings(tmp_path / "plain"))
    _ingest(plain, "ops", "# 运维手册\n\n## 扩容\n\nKubernetes 集群扩容时先检查节点资源，再调整副本数。\n")
    r2 = HybridRetriever(plain.settings, plain.index)
    assert not r2.retrieve_sparse("K8s")
    plain.index.aliases = AliasDict([["K8s", "Kubernetes"]])                  # 只做查询扩展，不重建索引也能命中
    assert r2.retrieve_sparse("K8s")


# --- 关键词与剔除 -------------------------------------------------------------------------------
def test_keywords_and_exclusion_in_pipeline(tmp_path):
    settings = _settings(tmp_path)
    settings.corpus.keywords = True
    noise = "第一章 总则 ........ 3\n第二章 假期 ........ 5\n第三章 差旅 ........ 9"
    md = f"# 员工手册\n\n## 目录\n\n{noise}\n\n## 报销\n\n差旅报销需在出差结束后 30 日内提交，错误码 E-1062 表示发票重复。\n"
    pipe = IngestPipeline(settings, excluded={exclusion_key("hb", noise)})
    chunks = _ingest(pipe, "hb", md)
    assert not any("........" in c.text for c in chunks)                     # 剔除的段落不入库
    other = _ingest(pipe, "other", md)                                       # 别的文档里内容相同的段落不受影响
    assert any("........" in c.text for c in other)
    parent = next(c for c in chunks if c.kind == "parent")
    assert "E-1062" in parent.keywords and any("报销" in k or "发票" in k for k in parent.keywords)
    child = next(c for c in chunks if c.kind == "child")
    assert child.keywords == parent.keywords and "关键词：" in child.embed_text()


# --- 自动元数据 ---------------------------------------------------------------------------------
def test_extract_metadata_rules_from_real_doc():
    text = (ROOT / "data" / "raw" / "员工手册_2024版.md").read_text(encoding="utf-8").split("---", 2)[2]   # 去掉 front matter，只看正文
    found = extract_metadata(text, file_stem="员工手册_2024版")
    assert found["title"]["value"] == "员工手册（2024 版）"
    assert found["valid_from"]["value"] == "2024-01-01" and found["valid_to"]["value"] == "2025-12-31"
    assert "2025 年 12 月 31 日废止" in found["valid_to"]["evidence"]
    meta = DocMetadata(title="员工手册_2024版", version="人工填写")
    applied = apply_metadata(meta, found, file_stem="员工手册_2024版")
    assert "title" in applied and "version" not in applied and meta.version == "人工填写"   # 只填空字段
    assert meta.valid_to == date(2025, 12, 31) and not meta.is_effective(date(2026, 6, 1))


def test_extract_metadata_access_and_llm_evidence():
    text = "密级：机密\n\n# 数据中心巡检规范\n\n本规范由运维部负责解释。巡检每日两次。\n"
    llm = FakeLLM('{"title": null, "version": "3.0", "valid_from": "2027-01-01", "department": "运维部", "tags": ["巡检", "数据中心"]}')
    found = extract_metadata(text, llm)
    assert found["access_level"]["value"] == 2
    assert found["department"]["value"] == "运维部" and found["department"]["source"] == "llm"
    assert "version" not in found and "valid_from" not in found          # 原文里找不到依据的值丢弃
    assert found["tags"]["value"] == ["巡检", "数据中心"]
    assert "access_level" not in extract_metadata("# 保密制度\n\n员工不得泄露公司机密信息。\n")   # 正文提到"机密"不算密级


# --- 质量体检 ---------------------------------------------------------------------------------
def test_health_finds_duplicates_conflicts_and_noise(tmp_path):
    settings = _settings(tmp_path)
    pipe = IngestPipeline(settings)
    shared = "出差住宿标准：一线城市每晚不超过 600 元，其他城市每晚不超过 400 元，超标部分由个人承担，需部门负责人审批。"
    long_para = ("报销单需附发票原件与行程单，财务部在收到单据后十个工作日内完成审核并打款，逾期未提交视为放弃报销。"
                 "跨部门出差的费用由发起部门承担，项目出差计入项目成本，培训出差计入培训预算，年度结束前统一清算。")
    garbled = "设备额定功率为 220V，接口锟斤拷标准见附表，锟斤拷型号以铭牌为准，其余参数请咨询厂商技术支持。"
    _ingest(pipe, "old", f"# 差旅管理办法（2025 版）\n\n## 住宿\n\n{shared}\n\n## 报销\n\n{long_para}\n", valid_from=date(2025, 1, 1), version="2025")
    _ingest(pipe, "new", f"# 差旅管理办法（2026 版）\n\n## 住宿\n\n{shared}\n\n## 报销\n\n{long_para.replace('十个', '七个')}\n\n## 附录\n\n{garbled}\n", valid_from=date(2026, 1, 1), version="2026")
    _ingest(pipe, "toc", "# 目录\n\n## 目录\n\n第一章 总则 ........ 3\n第二章 假期 ........ 5\n第三章 差旅 ........ 9\n")

    r = CorpusHealthChecker(pipe.index).run(on_date=date(2026, 6, 1))
    by_issue = {i: [c for c in r["chunks"] if i in c["issues"]] for i in r["issueCounts"]}
    dup = by_issue["duplicate"]
    assert len(dup) == 1 and dup[0]["docId"] == "old" and dup[0]["duplicateOf"]["docId"] == "new"   # 保留新版
    assert by_issue["near_duplicate"] and by_issue["near_duplicate"][0]["docId"] == "old"
    assert by_issue["garbled"] and by_issue["low_info"]
    assert len(r["conflicts"]) == 1 and r["conflicts"][0]["keepDocId"] == "new"
    old_row = next(d for d in r["docs"] if d["docId"] == "old")
    assert any(i["type"] == "version_conflict" for i in old_row["issues"])
    assert r["score"] < 90 and r["recommendations"]

    # 剔除：连同 child 一起移出检索
    target = dup[0]["chunkId"]
    before = len(pipe.index.bm25.ids)
    removed = pipe.index.remove_chunks({target})
    assert target in removed and len(pipe.index.bm25.ids) < before and pipe.index.store.get(target) is None


# --- FAQ 增强 ---------------------------------------------------------------------------------
def _ctx(text: str, crumb: str = "员工手册 > 年假") -> Context:
    return Context(chunk=Chunk(chunk_id="c1", doc_id="d1", kind="parent", text=text, breadcrumbs=crumb.split(" > ")), score=1.0)


def test_draft_faq_llm_extractive_and_gap():
    ctx = [_ctx("当年未休完的年假可以跨年使用，最晚在次年 3 月 31 日前休完。逾期作废。")]
    llm = FakeLLM("问题：年假能攒到明年吗\n答案：可以。当年未休完的年假可跨年使用，但须在次年 3 月 31 日前休完，逾期作废。\n依据：1")
    item = draft_faq("年假能攒到明年吗", ctx, llm)
    assert item["status"] == "pending" and item["method"] == "llm" and "3 月 31 日" in item["answer"] and item["sources"][0]["chunkId"] == "c1"

    gap = draft_faq("年终奖怎么发", ctx, FakeLLM("NO_ANSWER"))
    assert gap["status"] == "gap" and not gap["answer"]
    assert draft_faq("年终奖怎么发", [], None)["status"] == "gap"

    ext = draft_faq("年假能攒到明年吗", ctx, None)
    assert ext["method"] == "extractive" and ext["answer"].startswith("当年未休完")

    md = build_faq_markdown([item], "员工制度")
    assert md.startswith("# 增强语料 · FAQ（员工制度）") and "## 年假能攒到明年吗" in md and "来源：员工手册 > 年假" in md
