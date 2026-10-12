"""难负例安全性：标签、同源切片、教师降级和问题划分。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from prism.config import Settings
from prism.eval import GoldenExample
from prism.index.doc_store import ChunkStore
from prism.retrieval import HybridRetriever
from prism.retrieval.rerank import NoopReranker
from prism.schemas import Chunk
from prism.training import HardNegativeMiner, MiningConfig, export_training


def _chunk(cid, text, *, parent=None, doc="other"):
    return Chunk(chunk_id=cid, doc_id=doc, kind="child", text=text, parent_id=parent,
                 breadcrumbs=["测试来源"], metadata={"source": f"{doc}.md"})


class Teacher:
    name = "api"
    model_name = "test-teacher"

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def rerank(self, query, candidates, top_k):
        self.calls.append((query, candidates, top_k))
        return sorted(((cid, self.scores[cid]) for cid, _ in candidates), key=lambda x: -x[1])


def _retriever(chunks, teacher=None):
    store = ChunkStore()
    store.add_chunks(chunks)
    cfg = Settings.lite().retrieval
    ranking = [(c.chunk_id, 1 - i * 0.01) for i, c in enumerate(chunks) if c.kind == "child"]
    idx = SimpleNamespace(store=store, embedder=SimpleNamespace(name="hash"),
                          signature=lambda: {"embedder": "hash", "model": None})
    return SimpleNamespace(index=idx, cfg=cfg, reranker=teacher or NoopReranker(),
                           _allowed_ids=lambda *args: None,
                           retrieve_dense=lambda q, n: ranking[:n],
                           retrieve_sparse=lambda q, n: list(reversed(ranking))[:n])


def _config(**kw):
    return MiningConfig(candidate_pool=10, rank_max=10, **kw)


def test_known_positives_siblings_duplicates_and_refusals_are_excluded():
    pos = _chunk("p:c1", "年假最晚次年三月使用", parent="p", doc="policy")
    sibling = _chunk("p:c2", "其他年假补充规则", parent="p", doc="policy")
    copy = _chunk("copy", pos.text, doc="copy")
    known = _chunk("p2:c1", "年假三月的另一段说明", parent="p2", doc="policy")
    neg1 = _chunk("n1", "病假需要医院诊断证明", parent="np1")
    neg2 = _chunk("n2", "交通补贴申请需当月完成", parent="np2", doc="traffic")
    retriever = _retriever([pos, sibling, copy, known, neg1, neg2])
    examples = [GoldenExample(question="年假能跨年吗？", relevant_chunk_ids=[pos.chunk_id]),
                GoldenExample(question=" 年假能跨年吗 ", relevant_chunk_ids=[known.chunk_id]),
                GoldenExample(question="食堂开门吗", expected_refusal=True),
                GoldenExample(question="未知", relevant_doc_ids=["policy"])]
    report = HardNegativeMiner(retriever, _config(exclude_same_document=False)).run(examples)
    assert report.stats["queries"] == 3
    assert report.stats["items"] == 1
    item = report.items[0]
    assert set(item["positive_chunk_ids"]) == {"p:c1", "p2:c1"}
    assert set(item["negative_chunk_ids"]) == {"n1", "n2"}
    assert item["requires_review"] and item["shortage"] == 2
    assert item["pos"][0].startswith("[来源：")
    assert item["rerank_pos"][0].startswith("测试来源\n")
    assert {s["reason"] for s in report.skipped} == {"expected_refusal", "precise_labels_required"}


def test_parent_answer_excludes_children_that_do_not_contain_answer():
    parent = Chunk(chunk_id="p", doc_id="policy", text="年假三月截止，其他说明", kind="parent")
    pos = _chunk("pc1", "年假三月截止", parent="p", doc="policy")
    sibling = _chunk("pc2", "其他说明", parent="p", doc="policy")
    negative = _chunk("n", "物流补贴规则")
    report = HardNegativeMiner(_retriever([parent, pos, sibling, negative]),
                               _config(exclude_same_document=False)).run([
        GoldenExample(question="何时截止", answer_keywords=["三月"]),
    ])
    assert report.items[0]["positive_chunk_ids"] == ["pc1"]
    assert report.items[0]["negative_chunk_ids"] == ["n"]


def test_parent_labels_score_all_children_before_selecting_positives():
    children = [_chunk(f"p:c{i}", f"其他假期补充条款第{i}项", parent="p", doc="policy") for i in range(4)]
    answer = _chunk("p:c4", "年假三月截止", parent="p", doc="policy")
    negative = _chunk("n", "交通补贴申请期限")
    teacher = Teacher({**{c.chunk_id: 0.2 for c in children}, "p:c4": 0.9, "n": 0.3})
    retriever = _retriever(children + [answer, negative], teacher)
    report = HardNegativeMiner(retriever, _config()).run([
        GoldenExample(question="年假截止", relevant_chunk_ids=["p"]),
    ])
    assert report.items[0]["positive_chunk_ids"] == ["p:c4"]
    assert report.items[0]["requires_review"]  # parent 标签尚未确认到 child
    assert len(teacher.calls[0][1]) == 6
    report = HardNegativeMiner(retriever, _config()).run([
        GoldenExample(question="年假截止", relevant_chunk_ids=["p"], answer_keywords=["三月"]),
    ])
    assert report.items[0]["positive_chunk_ids"] == ["p:c4"]
    assert not report.items[0]["requires_review"]


def test_keyword_scope_and_explicit_labels_take_precedence():
    p = _chunk("p", "年假三月截止", doc="policy")
    elsewhere = _chunk("else", "年假三月旧版本规则", doc="old")
    neg = _chunk("n", "交通补贴申请期限", doc="traffic")
    report = HardNegativeMiner(_retriever([p, elsewhere, neg]), _config()).run([
        GoldenExample(question="年假", answer_keywords=["三月"], relevant_sources=["policy.md"]),
    ])
    assert report.items[0]["positive_chunk_ids"] == ["p"]
    report = HardNegativeMiner(_retriever([p, elsewhere, neg]), _config()).run([
        GoldenExample(question="年假", relevant_chunk_ids=["missing"], answer_keywords=["三月"]),
    ])
    assert report.skipped[0]["reason"] == "no_positive_in_active_index"


def test_teacher_filters_near_positive_and_keeps_score_aligned():
    chunks = [_chunk("p", "年假规则", doc="policy"),
              _chunk("suspect", "其他假期的相似条款"),
              _chunk("hard", "病假证明申请要求", doc="sick"),
              _chunk("easy", "仓库库位盘点流程", doc="warehouse")]
    teacher = Teacher({"p": 0.9, "suspect": 0.86, "hard": 0.7, "easy": 0.2})
    report = HardNegativeMiner(_retriever(chunks, teacher), _config()).run([
        GoldenExample(question="年假", relevant_chunk_ids=["p"]),
    ])
    item = report.items[0]
    assert item["negative_chunk_ids"] == ["hard", "easy"]
    assert item["pos_scores"] == [0.9] and item["neg_scores"] == [0.7, 0.2]
    assert not item["requires_review"]
    assert teacher.calls[0][2] == len(teacher.calls[0][1]) == 4


def test_suspected_positives_are_reported_and_easy_negatives_filtered(tmp_path):
    chunks = [_chunk("p", "年假规则", doc="policy"),
              _chunk("suspect", "其他假期的相似条款"),
              _chunk("hard", "病假证明申请要求", doc="sick"),
              _chunk("easy", "仓库库位盘点流程", doc="warehouse")]
    teacher = Teacher({"p": 0.9, "suspect": 0.86, "hard": 0.7, "easy": 0.2})
    label = GoldenExample(question="年假", relevant_chunk_ids=["p"])
    report = HardNegativeMiner(_retriever(chunks, teacher), _config(min_negative_score=0.3)).run([label])
    item = report.items[0]
    assert item["negative_chunk_ids"] == ["hard"] and item["easy_filtered"] == 1
    assert [(s["chunk_id"], s["score"]) for s in item["suspected_positives"]] == [("suspect", 0.86)]
    assert "相似条款" in item["suspected_positives"][0]["text"]
    assert report.stats["suspected_positives"] == 1 and report.stats["easy_filtered"] == 1
    assert any("疑似漏标" in w for w in report.warnings)
    # 教师分随训练集导出，供蒸馏使用；重排数据仍是二元标签
    export_training(report, tmp_path / "out")
    rows = [json.loads(line) for name in ("train.jsonl", "dev.jsonl")
            for line in (tmp_path / "out" / name).read_text().splitlines()]
    assert rows == [{"query": "年假", "pos": item["pos"], "neg": item["neg"], "pos_scores": [0.9], "neg_scores": [0.7]}]
    # 只剩疑似正例和简单负例时不凑数，跳过并保留线索
    report = HardNegativeMiner(_retriever(chunks, teacher), _config(min_negative_score=0.75)).run([label])
    assert not report.items
    assert report.skipped[0]["reason"] == "no_safe_negatives" and report.skipped[0]["easy_filtered"] == 2
    assert report.skipped[0]["suspected_positives"][0]["chunk_id"] == "suspect"


class SyntheticTeacher(Teacher):
    """改写文本没有固定 ID，按正文关键字给分。"""

    def rerank(self, query, candidates, top_k):
        self.calls.append((query, candidates, top_k))
        return [(cid, self.scores.get(cid, next((v for k, v in self.scores.items() if k in text), 0.0)))
                for cid, text in candidates]


def test_synthetic_negatives_are_filtered_and_held_for_review(tmp_path):
    from prism.generation.llm import FakeLLM

    pos = _chunk("p", "年假最晚可以延到次年 3 月 31 日使用", doc="policy")
    pos.hypothetical_questions = ["年假能跨年吗"]
    chunks = [pos, _chunk("hard", "病假证明申请要求", doc="sick")]
    passages = ["调休最晚可以延到次年 3 月 31 日使用",       # 含答案关键词
                "病假证明申请要求。",                         # 照抄已挖到的负例
                "年假当年没休完的部分可以顺延到第二年",        # 教师判定仍在回答问题
                "调休假应当在加班后的两个月内安排休完",        # 保留
                "仓库盘点流程"]                              # 超出请求数量，被截断
    llm = FakeLLM(lambda prompt: "```json\n" + json.dumps(passages, ensure_ascii=False) + "\n```")
    teacher = SyntheticTeacher({"p": 0.9, "hard": 0.6, "顺延": 0.85, "调休假": 0.5, "仓库": 0.01})
    label = GoldenExample(question="年假可以跨年吗", relevant_chunk_ids=["p"], answer_keywords=["3 月 31 日"])
    config = MiningConfig(candidate_pool=10, rank_max=10, synthetic_negatives=4, min_negative_score=0.1)
    with pytest.raises(ValidationError):
        MiningConfig(synthetic_negatives=5)
    report = HardNegativeMiner(_retriever(chunks, teacher), config, llm=llm).run([label])
    item = report.items[0]
    assert "年假可以跨年吗" in llm.calls[0] and pos.text in llm.calls[0]
    assert "不要出现：3 月 31 日" in llm.calls[0]
    assert [s["body"] for s in item["synthetic_neg"]] == ["调休假应当在加班后的两个月内安排休完"]
    assert item["synthetic_neg"][0]["score"] == 0.5 and item["synthetic_neg"][0]["source_chunk_id"] == "p"
    # 正例的假设性问题不能带进改写负例的训练文本
    assert "年假能跨年吗" in item["pos"][0] and "年假能跨年吗" not in item["synthetic_neg"][0]["text"]
    assert item["synthetic_neg"][0]["text"].startswith("[来源：测试来源]")
    assert item["synthetic_rejected"] == {"contains_answer_keyword": 1, "near_copy": 1, "teacher_too_close": 1}
    assert item["negative_chunk_ids"] == ["hard"] and not item["requires_review"]
    assert report.stats["synthetic_negatives"] == 1 and report.stats["synthetic_rejected"] == 3
    assert report.models["teacherFiltered"] == 1 and report.models["syntheticModel"] == "fake"

    def rows(name, **kw):
        counts = export_training(report, tmp_path / name, **kw)
        files = {f: [json.loads(line) for line in (tmp_path / name / f).read_text().splitlines()] for f in counts}
        return files["train.jsonl"] + files["dev.jsonl"], files["reranker.train.jsonl"] + files["reranker.dev.jsonl"]

    held, held_pairs = rows("held")
    assert held[0]["neg"] == item["neg"] and len(held_pairs) == 2
    merged, merged_pairs = rows("merged", include_synthetic=True)
    assert merged[0]["neg"] == item["neg"] + [item["synthetic_neg"][0]["text"]]
    assert merged[0]["neg_scores"] == [0.6, 0.5]
    assert {"query": "年假可以跨年吗", "passage": item["synthetic_neg"][0]["rerank_text"], "label": 0} in merged_pairs
    assert json.loads((tmp_path / "merged" / "mining-report.json").read_text())["export"]["include_synthetic"]


def test_synthetic_only_items_wait_for_review_and_failures_are_reported(tmp_path):
    from prism.generation.llm import FakeLLM

    chunks = [_chunk("p", "年假最晚可以延到次年三月使用", doc="policy"), _chunk("easy", "仓库盘点流程", doc="warehouse")]
    teacher = SyntheticTeacher({"p": 0.9, "easy": 0.01, "调休假": 0.5})
    label = GoldenExample(question="年假可以跨年吗", relevant_chunk_ids=["p"])
    config = MiningConfig(candidate_pool=10, rank_max=10, synthetic_negatives=1, min_negative_score=0.1)
    llm = FakeLLM(lambda prompt: json.dumps(["调休假应当在加班后的两个月内安排休完"], ensure_ascii=False))
    report = HardNegativeMiner(_retriever(chunks, teacher), config, llm=llm).run([label])
    assert report.items[0]["neg"] == [] and len(report.items[0]["synthetic_neg"]) == 1
    assert not sum(export_training(report, tmp_path / "held").values())
    counts = export_training(report, tmp_path / "merged", include_synthetic=True)
    assert counts["train.jsonl"] + counts["dev.jsonl"] == 1
    assert counts["reranker.train.jsonl"] + counts["reranker.dev.jsonl"] == 2
    # 输出不是 JSON、没有文本模型：不产生改写负例，原有流程不受影响
    broken = HardNegativeMiner(_retriever(chunks, teacher), config, llm=FakeLLM(lambda prompt: "抱歉，我做不到")).run([label])
    assert broken.skipped[0]["reason"] == "no_safe_negatives" and any("生成失败" in w for w in broken.warnings)
    missing = HardNegativeMiner(_retriever(chunks, teacher), config).run([label])
    assert not missing.items and any("没有可用的文本模型" in w for w in missing.warnings)
    # 正例的末级标题是它自己的主题，不能套在改写文本上；没有答案关键词时提示词里不出现这一条
    chunks[0].breadcrumbs = ["员工手册", "假期管理", "年假"]
    report = HardNegativeMiner(_retriever(chunks, teacher), config, llm=llm).run([label])
    assert report.items[0]["synthetic_neg"][0]["text"].startswith("[来源：员工手册 > 假期管理]\n")
    assert report.items[0]["synthetic_neg"][0]["rerank_text"].startswith("员工手册 > 假期管理\n")
    assert "不要出现" not in llm.calls[-1]
    # 没有教师时改写负例不打分，整组仍待复核
    unscored = HardNegativeMiner(_retriever(chunks), config.model_copy(update={"min_negative_score": 0.0}), llm=llm).run([label])
    assert unscored.items[0]["requires_review"] and unscored.items[0]["synthetic_neg"][0]["score"] is None


def test_relative_margin_and_low_positive_are_applied_on_same_teacher_scale():
    chunks = [_chunk("p", "年假规则", doc="policy"),
              _chunk("near", "病假证明申请要求", doc="sick"),
              _chunk("far", "仓库库位盘点流程", doc="warehouse")]
    teacher = Teacher({"p": 0.8, "near": 0.6, "far": 0.3})
    retriever = _retriever(chunks, teacher)
    label = GoldenExample(question="年假", relevant_chunk_ids=["p"])
    report = HardNegativeMiner(retriever, _config(absolute_margin=0, relative_margin=0.5)).run([label])
    assert report.items[0]["negative_chunk_ids"] == ["far"]
    report = HardNegativeMiner(retriever, _config(min_positive_score=0.9)).run([label])
    assert not report.items and report.skipped[0]["reason"] == "low_positive_score"


@pytest.mark.parametrize("failure", ["nan", "partial", "fallback", "error"])
def test_teacher_failure_never_marks_candidates_verified(failure):
    chunks = [_chunk("p", "年假规则", doc="policy"), _chunk("n", "交通补贴申请期限")]
    teacher = Teacher({"p": 0.9, "n": float("nan") if failure == "nan" else 0.3})
    if failure == "partial":
        teacher.rerank = lambda *args: [("p", 0.9)]
    if failure == "fallback":
        teacher.status = lambda: {"state": "failed"}
    if failure == "error":
        def fail(*args):
            raise RuntimeError("test scorer unavailable")
        teacher.rerank = fail
    report = HardNegativeMiner(_retriever(chunks, teacher), _config()).run([
        GoldenExample(question="年假", relevant_chunk_ids=["p"]),
    ])
    assert report.items[0]["requires_review"]
    assert report.items[0]["pos_scores"] == []
    assert any("教师" in w for w in report.warnings)


def test_background_teacher_load_cannot_verify_a_fallback_batch():
    chunks = [_chunk("p", "年假规则", doc="policy"), _chunk("n", "交通补贴申请期限")]
    teacher = Teacher({"p": 0.9, "n": 0.3})
    teacher.name = "cross_encoder"
    teacher.fallback = NoopReranker()
    states = iter([{"state": "loading"}, {"state": "ready"}])
    teacher.status = lambda: next(states)
    teacher.pop_note = lambda: None  # 并发查询已消费了 fallback note
    report = HardNegativeMiner(_retriever(chunks, teacher), _config()).run([
        GoldenExample(question="年假", relevant_chunk_ids=["p"]),
    ])
    assert report.items[0]["requires_review"]
    assert report.models["teacherFiltered"] == 0


def test_heldout_and_stable_query_split(tmp_path):
    chunks = [_chunk("p", "年假规则", doc="policy"), _chunk("n", "交通补贴申请期限")]
    miner = HardNegativeMiner(_retriever(chunks), _config())
    ex = GoldenExample(question="年假能跨年吗？", relevant_chunk_ids=["p"])
    held = [GoldenExample(question=" 年假能跨年吗 ")]
    assert miner.run([ex], held_out=held).skipped[0]["reason"] == "held_out_question"
    first = miner.run([ex])
    second = miner.run([ex.model_copy(update={"question": "年假能跨年吗"})])
    assert first.items[0]["split"] == second.items[0]["split"]
    counts = export_training(first, tmp_path / "unreviewed")
    assert not sum(counts.values())
    counts = export_training(first, tmp_path / "reviewed", allow_unverified=True)
    assert counts["train.jsonl"] + counts["dev.jsonl"] == 1
    manifest = json.loads((tmp_path / "reviewed" / "mining-report.json").read_text())
    assert manifest["export"]["allow_unverified"]
    with pytest.raises(FileExistsError):
        export_training(first, tmp_path / "reviewed")


def test_negative_parent_diversity_and_document_limit():
    chunks = [_chunk("p", "年假规则", doc="policy"),
              _chunk("n1", "病假证明需要提前提交", parent="n", doc="sick"),
              _chunk("n2", "病假医疗文件的审核流程", parent="n", doc="sick"),
              _chunk("n3", "病假薪资计算的说明", parent="n3", doc="sick"),
              _chunk("n4", "公司其他福利制度", parent="n4", doc="benefits")]
    report = HardNegativeMiner(_retriever(chunks), _config(max_per_doc=1)).run([
        GoldenExample(question="年假", relevant_chunk_ids=["p"]),
    ])
    assert len(report.items[0]["neg"]) == 2
    assert report.stats["shortages"] == 2


def test_real_index_mining_does_not_run_generation(built_index):
    retriever = HybridRetriever(built_index.settings, built_index, reranker=NoopReranker())
    report = HardNegativeMiner(retriever, MiningConfig(use_teacher=False)).run([
        GoldenExample(question="年假能跨年使用吗", answer_keywords=["次年 3 月 31 日"]),
    ])
    assert len(report.items) == 1 and report.items[0]["neg"]
    assert report.items[0]["requires_review"]


def test_report_and_manifest_never_export_retrieval_credentials(tmp_path):
    retriever = _retriever([_chunk("p", "年假规则", doc="policy"), _chunk("n", "交通补贴申请期限")])
    retriever.cfg.rerank.api_key = "synthetic-key-should-never-be-exported"
    report = HardNegativeMiner(retriever, _config()).run([
        GoldenExample(question="年假", relevant_chunk_ids=["p"]),
    ])
    assert "synthetic-key" not in json.dumps(report.to_dict())
    export_training(report, tmp_path / "exports")
    assert "synthetic-key" not in (tmp_path / "exports" / "mining-report.json").read_text()


def test_cli_exports_audit_but_holds_unreviewed_training_rows(tmp_path, built_index, monkeypatch):
    from typer.testing import CliRunner

    import prism.cli as cli

    monkeypatch.setattr(cli, "_settings", lambda *args: built_index.settings)
    monkeypatch.setattr(cli, "_load_index", lambda *args: built_index)
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(GoldenExample(question="年假能跨年使用吗", answer_keywords=["次年 3 月 31 日"]).model_dump_json())
    output = tmp_path / "output"
    result = CliRunner().invoke(cli.app, ["mine-negatives", str(annotations), "--output", str(output), "--no-teacher", "--lite"])
    assert result.exit_code == 0, result.output
    assert json.loads((output / "mining-report.json").read_text())["stats"]["items"] == 1
    assert not (output / "train.jsonl").read_text() and not (output / "dev.jsonl").read_text()
    result = CliRunner().invoke(cli.app, ["mine-negatives", str(annotations), "--output", str(output)])
    assert result.exit_code == 1 and "已存在" in result.output


@pytest.mark.parametrize("kw", [{"absolute_margin": float("nan")}, {"rank_min": 101},
                               {"min_negative_score": 0.9},
                               {"dev_fraction": 1}, {"max_per_doc": 0}, {"unknown": 1}])
def test_config_validation(kw):
    with pytest.raises(ValidationError):
        MiningConfig(**kw)
