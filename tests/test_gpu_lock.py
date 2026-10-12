"""本地模型在 Apple GPU（MPS）上必须串行：PyTorch 的 MPS 后端并发使用会 abort 进程或卡死（见 prism/gpu.py）。

这里不加载真实模型：用假模型在"推理进行中"探测全局锁是否被当前线程持有。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from prism.gpu import MPS_LOCK, mps_guard, on_mps


def _held_by_another_thread() -> bool:
    """从另一个线程试着拿锁：拿不到 = 当前（调用方）线程正持有它。"""
    free: list[bool] = []

    def probe() -> None:
        ok = MPS_LOCK.acquire(blocking=False)
        if ok:
            MPS_LOCK.release()
        free.append(ok)

    t = threading.Thread(target=probe)
    t.start()
    t.join()
    return not free[0]


def test_guard_only_serializes_mps():
    assert on_mps("mps") and on_mps("mps:0") and not on_mps("cpu") and not on_mps("cuda:0") and not on_mps(None)
    assert mps_guard("mps") is MPS_LOCK
    with mps_guard("cpu"):
        assert not _held_by_another_thread()
    with mps_guard("mps"), mps_guard("mps:0"):      # 可重入：持锁的代码里再调别的本地模型不会自锁
        assert _held_by_another_thread()
    assert not _held_by_another_thread()


def test_sentence_transformer_encode_holds_lock_on_mps():
    from prism.index.embeddings import SentenceTransformerEmbedder

    seen: list[bool] = []

    class FakeModel:
        def __init__(self, device: str) -> None:
            self.device = device

        def encode(self, texts, **_kw):
            seen.append(_held_by_another_thread())
            return [[1.0, 0.0] for _ in texts]

    for device, expect in (("mps", True), ("cpu", False)):
        emb = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)   # 不加载真实的 2GB 模型
        emb.model, emb.batch_size, emb.normalize, emb.query_instruction, emb.dim = FakeModel(device), 8, True, "", 2
        assert emb.embed_queries(["q"]).shape == (1, 2)
        assert seen.pop() is expect

    # 大批量入库按段持锁：每段单独拿锁，段与段之间查询可以插进来
    emb.model = FakeModel("mps")
    n = SentenceTransformerEmbedder._LOCK_SLICE * 2 + 1
    assert emb.embed_documents([f"t{i}" for i in range(n)]).shape == (n, 2)
    assert seen == [True, True, True]
    assert emb.embed_documents([]).shape == (0, 2)


def test_cross_encoder_inference_holds_lock_on_mps():
    from prism.retrieval import rerank

    seen: list[bool] = []

    class Batch(dict):
        def to(self, _dev):
            return self

    class Logits:
        def __init__(self, n: int) -> None:
            self.n = n

        def view(self, *_):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [0.5] * self.n

    def tokenizer(queries, texts, **_kw):
        return Batch(n=len(texts))

    def net(n):
        seen.append(_held_by_another_thread())
        return SimpleNamespace(logits=Logits(n))

    assert rerank._cross_encoder_logits((tokenizer, net, "mps"), "q", ["a", "b"], 512, 16) == [0.5, 0.5]
    assert seen.pop() is True
    rerank._cross_encoder_logits((tokenizer, net, "cpu"), "q", ["a"], 512, 16)
    assert seen.pop() is False


def test_mineru_local_parse_holds_lock_on_mps(monkeypatch, tmp_path: Path):
    from prism.config import MinerUConfig
    from prism.ingest.parsers.mineru_parser import MinerUParser

    seen: list[bool] = []

    def fake_parse(path, **kw):
        seen.append(_held_by_another_thread())
        return "result"

    # 不导入真实的 MinerU：_run 里的 `from mineru.parser import parse` 直接拿到这个假模块
    monkeypatch.setitem(sys.modules, "mineru.parser", SimpleNamespace(parse=fake_parse))
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    parser = MinerUParser(MinerUConfig(api_url=None))
    for device, expect in (("mps", True), ("cpu", False)):
        monkeypatch.setenv("MINERU_DEVICE_MODE", device)
        assert parser._run(pdf) == "result"
        assert seen.pop() is expect
