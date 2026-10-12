"""评估：没有度量的调参都是盲修。"""

from .coverage import CoverageAnalyzer, CoverageReport
from .dataset import GoldenExample, load_golden_set
from .metrics import hit_rate, mrr, ndcg, recall_at_k
from .multimodal import MultimodalEvalReport, MultimodalEvalRunner, MultimodalExample, build_samples
from .runner import EvalReport, EvalRunner

__all__ = ["CoverageAnalyzer", "CoverageReport", "EvalReport", "EvalRunner", "GoldenExample", "MultimodalEvalReport", "MultimodalEvalRunner", "MultimodalExample", "build_samples", "hit_rate", "load_golden_set", "mrr", "ndcg", "recall_at_k"]
