"""Optional, validated embedding/reranker fine-tuning; dry runs use only stdlib.

Run ``python -m prism.training.finetune --help``. Model dependencies are imported
only after data and output validation, and never during a dry run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FinetuneConfig:
    task: str
    train: Path
    dev: Path
    model: str
    output: Path
    epochs: float = 1.0
    learning_rate: float = 2e-5
    batch_size: int = 8
    max_length: int = 512
    max_steps: int = -1
    seed: int = 42
    device: str = "auto"
    local_files_only: bool = False
    query_instruction: str = ""
    distill_weight: float = 0.0

    def validate(self) -> None:
        if self.task not in {"embedding", "reranker"}:
            raise ValueError("task must be embedding or reranker")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty local directory or Hugging Face model ID")
        for name in ("epochs", "learning_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("batch_size", "max_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or (self.max_steps != -1 and self.max_steps < 1):
            raise ValueError("max_steps must be -1 or a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer between 0 and 2**32 - 1")
        if self.device not in {"auto", "cpu"}:
            raise ValueError("device must be auto or cpu")
        if not isinstance(self.local_files_only, bool):
            raise ValueError("local_files_only must be a boolean")
        if not isinstance(self.query_instruction, str):
            raise ValueError("query_instruction must be a string")
        if self.task == "reranker" and self.query_instruction:
            raise ValueError("query_instruction applies only to embedding training")
        weight = self.distill_weight
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError("distill_weight must be zero or a positive finite number")
        if self.task == "reranker" and weight:
            raise ValueError("distill_weight applies only to embedding training")


def normalize_text(value: str) -> str:
    """Protect exact query/text duplicates, including case and width variants."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def normalize_query(value: str) -> str:
    return re.sub(r"[\W_]+", "", normalize_text(value))


def _text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not normalize_text(value):
        raise ValueError(f"{location}: expected a non-empty string")
    return value.strip()


def _texts(value: Any, location: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}: expected a non-empty list of strings")
    result: dict[str, str] = {}
    for index, item in enumerate(value):
        text = _text(item, f"{location}[{index}]")
        result.setdefault(normalize_text(text), text)
    return list(result.values())


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    data = path.read_bytes()
    records = []
    for line_number, line in enumerate(data.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: each JSONL line must be an object")
        records.append(row)
    if not records:
        raise ValueError(f"{path}: dataset is empty")
    return records, hashlib.sha256(data).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value}")


def prepare_dataset(path: Path, task: str, *, distill: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate labels and emit only loss input columns, with fixed triplet shape.

    Every positive/negative combination is retained for embedding training;
    variable negative counts require no padding, truncation, or row dropping.
    With ``distill``, each triplet also carries ``label``: the teacher's
    [query-positive, query-negative] scores, which must be 0..1 relevance scores.
    """
    records, digest = _read_jsonl(path)
    examples: dict[tuple[str, ...], dict[str, Any]] = {}
    labels: dict[tuple[str, str], int] = {}
    queries = set()
    scored_rows = 0
    canonical_text: dict[str, str] = {}
    canonical_queries: dict[str, str] = {}

    def canonical(text: str) -> str:
        return canonical_text.setdefault(normalize_text(text), text)

    def register(query: str, passage: str, label: int, location: str) -> None:
        pair = (normalize_query(query), normalize_text(passage))
        if pair in labels and labels[pair] != label:
            raise ValueError(f"{location}: conflicting positive/negative labels for the same query and passage")
        labels[pair] = label

    for index, record in enumerate(records, 1):
        location = f"{path}:record {index}"
        query = _text(record.get("query"), f"{location}.query")
        if not normalize_query(query):
            raise ValueError(f"{location}.query: query must contain letters or numbers")
        query = canonical_queries.setdefault(normalize_query(query), query)
        queries.add(normalize_query(query))
        if task == "embedding":
            positives = [canonical(text) for text in _texts(record.get("pos"), f"{location}.pos")]
            negatives = [canonical(text) for text in _texts(record.get("neg"), f"{location}.neg")]
            has_scores = "pos_scores" in record or "neg_scores" in record
            if has_scores:
                scored_rows += 1
                for field, texts_field in (("pos_scores", "pos"), ("neg_scores", "neg")):
                    scores = record.get(field)
                    if not isinstance(scores, list) or len(scores) != len(record[texts_field]):
                        raise ValueError(f"{location}.{field}: scores must align with {texts_field}")
                    if any(isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) for score in scores):
                        raise ValueError(f"{location}.{field}: scores must be finite numbers")
                    if distill and any(not 0 <= score <= 1 for score in scores):
                        raise ValueError(f"{location}.{field}: distillation expects 0..1 teacher relevance scores")
            elif distill:
                raise ValueError(f"{location}: distillation requires pos_scores and neg_scores on every record")
            # Scores follow the first occurrence of each text, as the de-duplicated texts do.
            teacher: dict[str, float] = {}
            if distill:
                for texts_field, scores_field in (("pos", "pos_scores"), ("neg", "neg_scores")):
                    for text, score in zip(record[texts_field], record[scores_field]):
                        teacher.setdefault(normalize_text(text), float(score))
            for positive in positives:
                register(query, positive, 1, location)
            for negative in negatives:
                register(query, negative, 0, location)
            for positive in positives:
                for negative in negatives:
                    triplet_key = (normalize_query(query), normalize_text(positive), normalize_text(negative))
                    example = {"query": query, "positive": positive, "negative": negative}
                    if distill:
                        example["label"] = [teacher[triplet_key[1]], teacher[triplet_key[2]]]
                    examples.setdefault(triplet_key, example)
        elif task == "reranker":
            passage = canonical(_text(record.get("passage"), f"{location}.passage"))
            label = record.get("label")
            if isinstance(label, bool) or not isinstance(label, (int, float)) or label not in {0, 1}:
                raise ValueError(f"{location}.label: expected numeric 0 or 1")
            register(query, passage, int(label), location)
            pair_key = (normalize_query(query), normalize_text(passage))
            examples.setdefault(pair_key, {"query": query, "passage": passage, "label": float(label)})
        else:
            raise ValueError("task must be embedding or reranker")

    if task == "reranker" and set(labels.values()) != {0, 1}:
        raise ValueError(f"{path}: reranker dataset must contain both positive (1) and negative (0) labels")
    return list(examples.values()), {
        "path": str(path.resolve()), "sha256": digest, "input_records": len(records),
        "queries": len(queries), "training_examples": len(examples),
        "positive_pairs": sum(label == 1 for label in labels.values()),
        "negative_pairs": sum(label == 0 for label in labels.values()),
        ("rows_with_teacher_scores" if distill else "rows_with_ignored_teacher_scores"): scored_rows,
    }


def _check_output(output: Path) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"output must be a new or empty directory: {output}")


def _complete_no_duplicates_sampler(base_class: type) -> type:
    """Cache official no-duplicate batches so Trainer gets their exact length.

    The official sampler's approximate length can stop an epoch before deferred
    same-query triplets are visited. A fixed conflict-free batch composition
    keeps the step count stable across epochs; order is shuffled each epoch.
    """
    class CompleteNoDuplicatesBatchSampler(base_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._batch_plan: list[list[int]] | None = None
            super().__init__(*args, **kwargs)

        def _batches(self) -> list[list[int]]:
            if self._batch_plan is None:
                self._batch_plan = [list(batch) for batch in super().__iter__() if batch]
            return self._batch_plan

        def __len__(self) -> int:
            return len(self._batches())

        def __iter__(self):
            batches = self._batches()
            indices = list(range(len(batches)))
            rng = random.Random((self.seed or 0) + self.epoch)
            rng.shuffle(indices)
            for index in indices:
                batch = list(batches[index])
                rng.shuffle(batch)
                yield batch

    return CompleteNoDuplicatesBatchSampler


def complete_no_duplicates_batch_sampler(
    dataset: Any, batch_size: int, drop_last: bool,
    valid_label_columns: list[str] | None = None, generator: Any = None, seed: int = 0,
) -> Any:
    """Top-level callable stays pickleable when Trainer saves its arguments."""
    base = importlib.import_module("sentence_transformers.base.sampler").NoDuplicatesBatchSampler
    return _complete_no_duplicates_sampler(base)(
        dataset=dataset, batch_size=batch_size, drop_last=drop_last,
        valid_label_columns=valid_label_columns, generator=generator, seed=seed,
    )


def _contrastive_distill_loss(model: Any, weight: float) -> Any:
    """In-batch contrastive loss plus label-level distillation from the mining teacher.

    The teacher is the reranker that filtered the negatives; its 0..1 scores are
    sigmoid outputs, so the logit restores the scale its softmax is taken on. The
    student side reuses the contrastive loss's scaled cosine similarity.
    """
    torch = importlib.import_module("torch")
    losses = importlib.import_module("sentence_transformers.sentence_transformer.losses")
    embed_columns = importlib.import_module("sentence_transformers.base.losses.merged_forward").embed_columns

    class ContrastiveDistillLoss(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.contrastive = losses.MultipleNegativesRankingLoss(model)

        def forward(self, sentence_features: Any, labels: Any) -> Any:
            embeddings = embed_columns(self.model, sentence_features)
            query, positive, negative = embeddings
            student = torch.stack([torch.cosine_similarity(query, positive),
                                   torch.cosine_similarity(query, negative)], dim=1) * self.contrastive.scale
            target = torch.softmax(torch.logit(labels.detach().to(student.dtype), eps=1e-4), dim=1)
            distill = torch.nn.functional.kl_div(torch.log_softmax(student, dim=1), target, reduction="batchmean")
            return self.contrastive.compute_loss_from_embeddings(embeddings, labels) + weight * distill

        def get_config_dict(self) -> dict[str, Any]:
            return {**self.contrastive.get_config_dict(), "distill_weight": weight}

    return ContrastiveDistillLoss()


def _load_training_dependencies(task: str) -> dict[str, Any]:
    try:
        dataset = importlib.import_module("datasets").Dataset
        st = importlib.import_module("sentence_transformers")
        set_seed = importlib.import_module("transformers").set_seed
        if task == "embedding":
            losses = importlib.import_module("sentence_transformers.sentence_transformer.losses")
            return {"Dataset": dataset, "model": st.SentenceTransformer,
                    "trainer": st.SentenceTransformerTrainer, "args": st.SentenceTransformerTrainingArguments,
                    "loss": losses.MultipleNegativesRankingLoss, "sampler": complete_no_duplicates_batch_sampler,
                    "set_seed": set_seed}
        losses = importlib.import_module("sentence_transformers.cross_encoder.losses")
        return {"Dataset": dataset, "model": st.CrossEncoder, "trainer": st.CrossEncoderTrainer,
                "args": st.CrossEncoderTrainingArguments, "loss": losses.BinaryCrossEntropyLoss,
                "set_seed": set_seed}
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("Fine-tuning requires compatible training dependencies; install with "
                           "python -m pip install -e '.[finetune]' (sentence-transformers 6.1+). "
                           f"Import failed: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_finetuning(config: FinetuneConfig, *, dry_run: bool = False) -> dict[str, Any]:
    config.validate()
    output = config.output.expanduser().resolve()
    _check_output(output)
    distill = config.distill_weight > 0
    train, train_summary = prepare_dataset(config.train, config.task, distill=distill)
    dev, dev_summary = prepare_dataset(config.dev, config.task, distill=distill)
    train_queries = {normalize_query(row["query"]) for row in train}
    if train_queries & {normalize_query(row["query"]) for row in dev}:
        raise ValueError("train/dev query overlap after Unicode, case, punctuation and whitespace normalization; split by query group before mining")
    model_path = Path(config.model).expanduser()
    model_reference = str(model_path.resolve()) if model_path.is_dir() else config.model.strip()
    manifest = {
        "status": "dry_run" if dry_run else "running", "task": config.task, "model": model_reference,
        "output": str(output), "train": train_summary, "dev": dev_summary,
        "parameters": {key: value for key, value in asdict(config).items() if key not in {"train", "dev", "output", "model", "task"}},
        "loss": ("BinaryCrossEntropyLoss" if config.task == "reranker" else "MultipleNegativesRankingLoss"
                 + (f" + {config.distill_weight:g} * teacher KL divergence" if distill else "")),
        "model_output": str(output / "model"),
        "limitations": ["Dev loss is a candidate-set diagnostic; use held-out full-corpus retrieval evaluation before activation.",
                        "Normalized query checks do not detect paraphrases or source-document leakage."]
                        + (["NO_DUPLICATES prevents repeated text in a batch; semantic false negatives still need review.",
                            "Distillation copies the teacher's own errors and assumes 0..1 scores from one sigmoid-scaled reranker."
                            if distill else
                            "Teacher scores are validated but ignored; pass --distill-weight to distill them."]
                           if config.task == "embedding" else []),
    }
    if dry_run:
        return manifest

    deps = _load_training_dependencies(config.task)
    _check_output(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "training_manifest.json"
    # Claim this run before creating checkpoints or model files. Never resume or overwrite.
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
    try:
        deps["set_seed"](config.seed)
        model_kwargs: dict[str, Any] = {"local_files_only": config.local_files_only, "trust_remote_code": False}
        if config.device == "cpu":
            model_kwargs["device"] = "cpu"
        if config.task == "reranker":
            model_kwargs.update(num_labels=1, max_length=config.max_length)
        model = deps["model"](model_reference, **model_kwargs)
        if config.task == "embedding":
            model.max_seq_length = config.max_length
        args_kwargs: dict[str, Any] = {
            "output_dir": str(output / "checkpoints"), "num_train_epochs": config.epochs,
            "learning_rate": config.learning_rate, "per_device_train_batch_size": config.batch_size,
            "per_device_eval_batch_size": config.batch_size, "max_steps": config.max_steps,
            "seed": config.seed, "data_seed": config.seed,
            # Warmup can consume every update for small datasets, including ordinary one-epoch runs.
            "warmup_steps": 0,
            "eval_strategy": "epoch", "save_strategy": "epoch", "save_total_limit": 2,
            "load_best_model_at_end": True, "metric_for_best_model": "eval_loss", "greater_is_better": False,
            "report_to": [], "push_to_hub": False, "fp16": False, "bf16": False,
            "dataloader_drop_last": False, "use_cpu": config.device == "cpu", "logging_steps": 10,
        }
        if config.task == "embedding":
            args_kwargs["batch_sampler"] = deps["sampler"]
            if config.query_instruction:
                args_kwargs["prompts"] = {"query": config.query_instruction}
        trainer = deps["trainer"](
            model=model, args=deps["args"](**args_kwargs),
            loss=_contrastive_distill_loss(model, config.distill_weight) if distill else deps["loss"](model),
            train_dataset=deps["Dataset"].from_list(train), eval_dataset=deps["Dataset"].from_list(dev),
        )
        result = trainer.train()
        metrics = {"train": {**result.metrics, "global_step": result.global_step}, "dev": trainer.evaluate()}
        trainer.save_model(str(output / "model"))
        trainer.save_state()
        _write_json(output / "metrics.json", metrics)
        manifest["status"] = "complete"
        manifest["versions"] = {name: version(name) for name in ("sentence-transformers", "transformers", "datasets", "accelerate", "torch")}
        manifest["metrics"] = metrics
        _write_json(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        _write_json(manifest_path, manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["embedding", "reranker"])
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print a plan, without importing model dependencies or writing files")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--local-files-only", action="store_true", help="Load only local model files or existing Hugging Face cache")
    parser.add_argument("--query-instruction", default="", help="Embedding query prefix; use the identical prefix in production")
    parser.add_argument("--distill-weight", type=float, default=0.0,
                        help="Embedding only: weight of the KL term that distills pos_scores/neg_scores from the mining teacher (0 disables)")
    args = vars(parser.parse_args(argv))
    dry_run = args.pop("dry_run")
    try:
        result = run_finetuning(FinetuneConfig(**args), dry_run=dry_run)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Fine-tuning error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
