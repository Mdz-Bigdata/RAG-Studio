"""Fine-tuning input safety and dependency-free planning, without model downloads."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from prism.training.finetune import FinetuneConfig, _complete_no_duplicates_sampler, main, prepare_dataset, run_finetuning


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def config(tmp_path):
    return FinetuneConfig(
        task="embedding",
        train=write_rows(tmp_path / "train.jsonl", [{"query": "退款怎么办", "pos": ["退款流程"], "neg": ["退款额度", "年假流程"]}]),
        dev=write_rows(tmp_path / "dev.jsonl", [{"query": "年假怎么申请", "pos": ["年假申请表"], "neg": ["病假申请表"]}]),
        model="missing-test-model", output=tmp_path / "output",
    )


def test_dry_run_never_imports_training_dependencies_or_writes(config, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("dry-run imported optional dependencies")
    monkeypatch.setattr("prism.training.finetune._load_training_dependencies", unexpected)
    result = run_finetuning(config, dry_run=True)
    assert result["status"] == "dry_run"
    assert result["train"]["training_examples"] == 2
    assert len(result["train"]["sha256"]) == 64
    assert not config.output.exists()


def test_cli_dry_run_works_with_stdlib_only(config):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-S", "-m", "prism.training.finetune", "--task", "embedding", "--train", str(config.train),
         "--dev", str(config.dev), "--model", config.model, "--output", str(config.output), "--dry-run"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "dry_run"
    assert not config.output.exists()


def test_all_positives_and_variable_negative_counts_are_retained(tmp_path):
    path = write_rows(tmp_path / "train.jsonl", [
        {"query": "A", "pos": ["p1", "p2"], "neg": ["n1", "n2"], "metadata": {"source": "ignored"}},
        {"query": "B", "pos": ["p3"], "neg": ["n3"]},
    ])
    examples, summary = prepare_dataset(path, "embedding")
    assert len(examples) == 5
    assert {(row["positive"], row["negative"]) for row in examples if row["query"] == "A"} == {
        ("p1", "n1"), ("p1", "n2"), ("p2", "n1"), ("p2", "n2")}
    assert all(list(row) == ["query", "positive", "negative"] for row in examples)
    assert summary["positive_pairs"] == 3


def test_distillation_aligns_teacher_scores_with_deduplicated_texts(tmp_path):
    path = write_rows(tmp_path / "train.jsonl", [
        {"query": "A", "pos": ["p1", " P1 ", "p2"], "neg": ["n1"],
         "pos_scores": [0.9, 0.1, 0.8], "neg_scores": [0.3]},
    ])
    examples, summary = prepare_dataset(path, "embedding", distill=True)
    assert {(row["positive"], row["negative"]): row["label"] for row in examples} == {
        ("p1", "n1"): [0.9, 0.3], ("p2", "n1"): [0.8, 0.3]}
    assert summary["rows_with_teacher_scores"] == 1
    assert "label" not in prepare_dataset(path, "embedding")[0][0]


@pytest.mark.parametrize("row, message", [
    ({"query": "q", "pos": ["p"], "neg": ["n"]}, "requires pos_scores"),
    ({"query": "q", "pos": ["p"], "neg": ["n"], "pos_scores": [4.2], "neg_scores": [0.2]}, "0..1"),
])
def test_distillation_rejects_missing_or_unscaled_teacher_scores(tmp_path, row, message):
    with pytest.raises(ValueError, match=message):
        prepare_dataset(write_rows(tmp_path / "train.jsonl", [row]), "embedding", distill=True)


def test_distillation_plan_is_embedding_only_and_checks_both_splits(config, tmp_path):
    scored = replace(
        config, distill_weight=0.5,
        train=write_rows(tmp_path / "scored.jsonl", [{"query": "退款怎么办", "pos": ["退款流程"], "neg": ["退款额度"],
                                                     "pos_scores": [0.9], "neg_scores": [0.4]}]))
    with pytest.raises(ValueError, match="requires pos_scores"):
        run_finetuning(scored, dry_run=True)  # dev 没有教师分
    dev = write_rows(tmp_path / "scored-dev.jsonl", [{"query": "年假怎么申请", "pos": ["年假申请表"], "neg": ["病假申请表"],
                                                      "pos_scores": [0.8], "neg_scores": [0.2]}])
    plan = run_finetuning(replace(scored, dev=dev), dry_run=True)
    assert "0.5 * teacher KL divergence" in plan["loss"] and plan["parameters"]["distill_weight"] == 0.5
    with pytest.raises(ValueError, match="embedding training"):
        run_finetuning(replace(config, task="reranker", distill_weight=0.5), dry_run=True)


def test_normalized_duplicates_share_canonical_text(tmp_path):
    path = write_rows(tmp_path / "train.jsonl", [
        {"query": " Ｑ ", "pos": ["P"], "neg": ["N"]},
        {"query": "q", "pos": ["p", "p2"], "neg": ["n"]},
    ])
    examples, _ = prepare_dataset(path, "embedding")
    assert len(examples) == 2
    assert len({row["query"] for row in examples}) == 1
    assert len({row["negative"] for row in examples}) == 1


@pytest.mark.parametrize("rows, message", [
    ([{"query": "q", "pos": ["p"], "neg": [" P "]}], "conflicting"),
    ([{"query": "q", "pos": ["p"], "neg": ["n"]}, {"query": "Q", "pos": ["n"], "neg": ["p2"]}], "conflicting"),
    ([{"query": "q", "pos": [], "neg": ["n"]}], "non-empty list"),
    ([{"query": "q", "pos": "p", "neg": ["n"]}], "non-empty list"),
    ([{"query": "q", "pos": [42], "neg": ["n"]}], "non-empty string"),
    ([{"query": " ", "pos": ["p"], "neg": ["n"]}], "non-empty string"),
    ([{"query": "q", "pos": ["p"], "neg": ["n"], "pos_scores": [0.8]}], "align"),
    ([{"query": "q", "pos": ["p"], "neg": ["n"], "pos_scores": [True], "neg_scores": [0.2]}], "finite"),
])
def test_embedding_rejects_invalid_or_conflicting_data(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        prepare_dataset(write_rows(tmp_path / "bad.jsonl", rows), "embedding")


@pytest.mark.parametrize("contents", ["", "\n", "[]\n", "not JSON\n", '{"query":NaN}\n'])
def test_invalid_jsonl_is_rejected(tmp_path, contents):
    path = tmp_path / "bad.jsonl"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        prepare_dataset(path, "embedding")


@pytest.mark.parametrize("label", [True, False, 0.5, 2, "1", None])
def test_reranker_rejects_invalid_binary_labels(tmp_path, label):
    with pytest.raises(ValueError, match="numeric 0 or 1"):
        prepare_dataset(write_rows(tmp_path / "bad.jsonl", [{"query": "q", "passage": "p", "label": label}]), "reranker")


def test_reranker_pair_conflicts_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="conflicting"):
        prepare_dataset(write_rows(tmp_path / "bad.jsonl", [
            {"query": "q", "passage": "p", "label": 1}, {"query": "Q", "passage": " P ", "label": 0},
        ]), "reranker")


def test_reranker_deduplicates_pairs_and_keeps_only_inputs(tmp_path):
    examples, summary = prepare_dataset(write_rows(tmp_path / "pairs.jsonl", [
        {"query": "q", "passage": "p", "label": 1, "source": "ignored"},
        {"query": "Q", "passage": " P ", "label": 1},
        {"query": "q", "passage": "n", "label": 0},
    ]), "reranker")
    assert len(examples) == 2
    assert all(list(row) == ["query", "passage", "label"] for row in examples)
    assert summary["queries"] == 1


def test_reranker_requires_both_label_classes(tmp_path):
    with pytest.raises(ValueError, match="both positive"):
        prepare_dataset(write_rows(tmp_path / "pairs.jsonl", [{"query": "q", "passage": "p", "label": 1}]), "reranker")


def test_normalized_train_dev_query_overlap_is_rejected(config):
    write_rows(config.dev, [{"query": " 退款怎么办？ ", "pos": ["p"], "neg": ["n"]}])
    with pytest.raises(ValueError, match="train/dev query overlap"):
        run_finetuning(config, dry_run=True)


@pytest.mark.parametrize("field, value", [
    ("epochs", float("nan")), ("epochs", float("inf")), ("epochs", 0),
    ("learning_rate", -1), ("learning_rate", float("nan")),
    ("batch_size", 0), ("batch_size", True), ("max_length", 0),
    ("max_steps", 0), ("seed", -1), ("model", " "), ("device", "bad"),
    ("distill_weight", -0.1), ("distill_weight", float("nan")), ("distill_weight", True),
])
def test_invalid_hyperparameters_are_rejected_before_import(config, field, value):
    with pytest.raises(ValueError):
        run_finetuning(replace(config, **{field: value}), dry_run=True)


def test_existing_output_is_protected_even_in_dry_run(config):
    config.output.mkdir()
    marker = config.output / "keep.txt"
    marker.write_text("original")
    with pytest.raises(ValueError, match="new or empty"):
        run_finetuning(config, dry_run=True)
    assert marker.read_text() == "original"


def test_missing_dependencies_have_actionable_error(config, monkeypatch):
    def missing(name):
        raise ImportError("No module named datasets")
    monkeypatch.setattr("prism.training.finetune.importlib.import_module", missing)
    with pytest.raises(RuntimeError, match=r"\[finetune\]"):
        run_finetuning(config)
    assert not config.output.exists()


def test_cli_validation_error_returns_nonzero(config, capsys):
    result = main(["--task", "embedding", "--train", str(config.train), "--dev", str(config.dev),
                   "--model", config.model, "--output", str(config.output), "--epochs", "nan", "--dry-run"])
    assert result == 1
    assert "positive and finite" in capsys.readouterr().err


def test_local_model_path_is_resolved_without_loading(config, tmp_path):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    result = run_finetuning(replace(config, model=str(model_path)), dry_run=True)
    assert result["model"] == str(model_path.resolve())


def test_complete_batch_plan_preserves_deferred_samples_across_epochs():
    class ApproximateSampler:
        def __init__(self):
            self.epoch = 0
            self.seed = 42
            self.calls = 0

        def __len__(self):
            return 1

        def __iter__(self):
            self.calls += 1
            yield [0]
            yield [1]

    sampler = _complete_no_duplicates_sampler(ApproximateSampler)()
    for epoch in range(3):
        sampler.epoch = epoch
        assert len(sampler) == 2
        assert sorted(index for batch in sampler for index in batch) == [0, 1]
    assert sampler.calls == 1


@pytest.mark.skipif(os.environ.get("RAG_TEST_FINETUNE") != "1", reason="Set RAG_TEST_FINETUNE=1 for offline tiny-model CPU training")
@pytest.mark.parametrize("task", ["embedding", "distill", "reranker"])
@pytest.mark.parametrize("max_steps", [-1, 1])
def test_offline_cpu_training_updates_and_reloads_weights(tmp_path, monkeypatch, task, max_steps):
    """One genuine gradient step, using random local weights and no model download."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    import torch
    from transformers import AutoModelForSequenceClassification, BertConfig, BertForSequenceClassification, BertModel, BertTokenizer
    from sentence_transformers import CrossEncoder, SentenceTransformer

    torch.set_num_threads(1)
    torch.manual_seed(1)
    base = tmp_path / "base"
    base.mkdir()
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "question", "answer", "document", "one", "two", "three", "four"]
    tokenizer = BertTokenizer(vocab={word: index for index, word in enumerate(words)}, model_max_length=32)
    tokenizer.save_pretrained(base)
    bert_config = BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                             intermediate_size=32, max_position_embeddings=64, num_labels=1)
    distill_weight = 0.5 if task == "distill" else 0.0
    if distill_weight:
        task = "embedding"
    if task == "embedding":
        BertModel(bert_config).save_pretrained(base)
        model_class = SentenceTransformer
        train_rows = [{"query": "question one", "pos": ["answer one"], "neg": ["document two", "document three"],
                       "pos_scores": [0.9], "neg_scores": [0.6, 0.1]}]
        dev_rows = [{"query": "question four", "pos": ["answer four"], "neg": ["document one"],
                     "pos_scores": [0.8], "neg_scores": [0.3]}]
    else:
        BertForSequenceClassification(bert_config).save_pretrained(base)
        model_class = CrossEncoder
        train_rows = [{"query": "question one", "passage": "answer one", "label": 1},
                      {"query": "question one", "passage": "document two", "label": 0}]
        dev_rows = [{"query": "question four", "passage": "answer four", "label": 1},
                    {"query": "question four", "passage": "document one", "label": 0}]
    before = model_class(str(base), local_files_only=True, device="cpu")
    state_before = {name: value.detach().clone() for name, value in before.state_dict().items()}
    cfg = FinetuneConfig(task=task, train=write_rows(tmp_path / "train.jsonl", train_rows),
                         dev=write_rows(tmp_path / "dev.jsonl", dev_rows), model=str(base),
                         output=tmp_path / "trained", max_steps=max_steps, batch_size=8, max_length=32,
                         learning_rate=1e-3, device="cpu", local_files_only=True, distill_weight=distill_weight)
    result = run_finetuning(cfg)
    assert result["status"] == "complete"
    expected_steps = 2 if task == "embedding" and max_steps == -1 else 1
    assert result["metrics"]["train"]["global_step"] == expected_steps
    assert math.isfinite(result["metrics"]["dev"]["eval_loss"])
    after = model_class(str(cfg.output / "model"), local_files_only=True, device="cpu")
    assert any(not torch.equal(state_before[name], value) for name, value in after.state_dict().items())
    if task == "reranker":
        # Production reranking uses transformers directly, so saved weights must support it too.
        AutoModelForSequenceClassification.from_pretrained(cfg.output / "model", local_files_only=True)
    assert (cfg.output / "model" / "config.json").is_file()
    assert json.loads((cfg.output / "training_manifest.json").read_text())["status"] == "complete"
    with pytest.raises(ValueError, match="new or empty"):
        run_finetuning(cfg)
