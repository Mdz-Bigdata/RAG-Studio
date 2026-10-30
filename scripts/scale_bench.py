"""千万级压测：合成 N 个段落灌进"千万级配置"的索引，测导入吞吐、磁盘、冷启动、内存、检索延迟与召回。

合成数据：
- 文本：按 jieba 词典词频（Zipf）抽词拼成中文段落，每个 child 约 60 个词；入库时照常用 jieba 重新分词（真实开销）；
- 向量：1024 维，512 个簇中心 + 48 维低秩语义方向 + 小噪声——真实文本向量的近邻关系由少数语义方向决定；
  纯各向同性随机向量在高维里"人人等距"，ANN 召回会被低估到没有参考价值（原型实测 0.24）；
- 结构：每篇文档 5 个 parent × 4 个 child；5% 的文档已过期、10% 是机密（access_level 2），让时效与权限过滤真的起作用。
向量是现成给的，不跑向量模型——导入吞吐里不含向量化（它取决于 GPU / API，见报告里单独的估算）。

用法：
    .venv/bin/python scripts/scale_bench.py build --dir /data/bench10m --children 10000000
    .venv/bin/python scripts/scale_bench.py query --dir /data/bench10m          # 新进程里测：冷启动、常驻内存、延迟、召回
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import shutil
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DIM = 1024                     # build --dim 可改；query 从 bench_build.json 读回
CHILDREN_PER_PARENT, PARENTS_PER_DOC, WORDS = 4, 5, 60
CHILDREN_PER_DOC = CHILDREN_PER_PARENT * PARENTS_PER_DOC
TODAY = date(2026, 10, 30)


def scale_settings(directory: Path):
    from prism.config import Settings

    st = Settings.lite(index_dir=directory)
    st.storage.chunk_store = "sqlite"
    st.vector_store.backend = "lancedb"
    st.vector_store.lancedb.fts_tokenizer = "jieba"
    st.vector_store.lancedb.index_type = "IVF_HNSW_SQ"
    st.retrieval.sparse_backend = "lancedb_fts"
    return st


class SyntheticEmbedder:
    """查询时把问题文本映射成一个确定的随机向量：端到端延迟用（召回另用已知的查询向量测）。"""

    name = "synthetic"
    model_name = "synthetic-1024"
    dim = DIM

    def embed_documents(self, texts):
        out = np.stack([np.random.default_rng(abs(hash(t)) % (2 ** 32)).standard_normal(DIM).astype(np.float32) for t in texts]) if texts else np.zeros((0, DIM), np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def embed_queries(self, texts):
        return self.embed_documents(texts)

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class Corpus:
    def __init__(self, seed: int = 0) -> None:
        import jieba

        words, freqs = [], []
        for line in open(Path(jieba.__file__).parent / "dict.txt", encoding="utf-8"):
            w, f, *_ = line.split()
            words.append(w)
            freqs.append(int(f))
        self.words = np.array(words)
        p = np.array(freqs, dtype=np.float64)
        self.cdf = np.cumsum(p / p.sum())
        self.rng = np.random.default_rng(seed)
        r = np.random.default_rng(12345)                       # 簇与语义方向固定：build / query 两个进程要一致
        self.centers = r.standard_normal((512, DIM)).astype(np.float32)
        self.basis = r.standard_normal((48, DIM)).astype(np.float32) / np.sqrt(48)

    def texts(self, n: int) -> list[list[str]]:
        idx = np.searchsorted(self.cdf, self.rng.random((n, WORDS)))
        return [list(self.words[np.minimum(row, len(self.words) - 1)]) for row in idx]

    def vectors(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or self.rng
        c = rng.integers(0, len(self.centers), n)
        v = self.centers[c] * 0.35 + rng.standard_normal((n, 48)).astype(np.float32) @ self.basis * 2.0 + 0.05 * rng.standard_normal((n, DIM)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def make_doc(corpus: Corpus, n: int):
    from prism.schemas import Chunk, DocMetadata

    doc_id = f"doc{n:08d}"
    title = f"合成文档{n}"
    md = DocMetadata(title=title, source=title, file_format="md", parser="synthetic")
    if n % 20 == 0:
        md.valid_to = date(2025, 1, 1)                          # 5% 已过期
    if n % 10 == 3:
        md.access_level = 2                                     # 10% 机密
    words = corpus.texts(CHILDREN_PER_DOC)
    chunks: list[Chunk] = []
    order = 0
    for pi in range(PARENTS_PER_DOC):
        pid = f"{doc_id}:p{pi:04d}"
        kids = []
        for ci in range(CHILDREN_PER_PARENT):
            text = "".join(words[pi * CHILDREN_PER_PARENT + ci])
            kids.append(Chunk(chunk_id=f"{pid}:c{ci:03d}", doc_id=doc_id, kind="child", text=text, parent_id=pid, order=order, breadcrumbs=[title, f"第{pi + 1}节"], token_count=WORDS))
            order += 1
        chunks.append(Chunk(chunk_id=pid, doc_id=doc_id, kind="parent", text="\n".join(k.text for k in kids), child_ids=[k.chunk_id for k in kids],
                            order=order, breadcrumbs=[title, f"第{pi + 1}节"], token_count=WORDS * CHILDREN_PER_PARENT))
        chunks.extend(kids)
    return doc_id, md, chunks


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)


def du(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p))], 1)


# --------------------------------------------------------------------------------------
def build(args) -> dict:
    global DIM
    from prism.index import Index
    from prism.ingest.bulk import BulkIndexer

    DIM = SyntheticEmbedder.dim = args.dim

    if args.dir.exists():
        shutil.rmtree(args.dir)
    st = scale_settings(args.dir)
    idx = Index(st, embedder=SyntheticEmbedder(), directory=args.dir)
    bulk = BulkIndexer(idx, batch_chunks=args.batch, tokenize_workers=args.workers)
    corpus = Corpus()
    docs = math.ceil(args.children / CHILDREN_PER_DOC)
    t0 = time.perf_counter()
    for n in range(docs):
        if n % 5000 == 0 and shutil.disk_usage(args.dir).free < args.min_free_gb * 1e9:
            sys.exit(f"磁盘剩余不足 {args.min_free_gb} GB，压测中止（数据在 {args.dir}，可以直接删除）")
        doc_id, md, chunks = make_doc(corpus, n)
        bulk.add(doc_id, md, chunks, vectors=corpus.vectors(CHILDREN_PER_DOC))
        if n and n % max(1, docs // 20) == 0:
            el = time.perf_counter() - t0
            print(f"  {n}/{docs} 篇，{bulk.stats.children / el:,.0f} child/s，RSS {rss_mb():,.0f} MB", flush=True)
    t_write = time.perf_counter() - t0
    stats = bulk.finish()
    total = time.perf_counter() - t0
    out = {"children": stats.children, "chunks": stats.chunks, "docs": stats.docs, "dim": DIM, "writeSeconds": round(t_write, 1),
           "indexSeconds": round(stats.seconds["indexes"], 1), "totalSeconds": round(total, 1), "childPerSecond": round(stats.children / t_write, 1),
           "phases": stats.as_dict()["seconds"], "buildPeakRssMB": round(rss_mb()), "diskGB": {"total": round(du(args.dir), 2), "lancedb": round(du(args.dir / "lancedb"), 2),
           "sqlite": round(sum(p.stat().st_size for p in args.dir.glob("chunks.sqlite*")) / 1e9, 2)}, "ann": idx.vectors._ann_meta()}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    (args.dir / "bench_build.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def query(args) -> dict:
    global DIM
    from prism.index import Index
    from prism.retrieval import HybridRetriever

    DIM = SyntheticEmbedder.dim = json.loads((args.dir / "bench_build.json").read_text(encoding="utf-8"))["dim"]

    st = scale_settings(args.dir)
    st.vector_store.search.nprobes_ratio = args.nprobes_ratio
    rss0 = rss_mb()
    t0 = time.perf_counter()
    idx = Index.load(st, args.dir, embedder=SyntheticEmbedder(), strict=False)
    t_load = time.perf_counter() - t0
    r = HybridRetriever(st, idx)
    corpus = Corpus(seed=999)
    rng = np.random.default_rng(7)
    n_docs = len(idx.store.docs)

    t = time.perf_counter()
    r.retrieve("冷启动第一次查询", top_k=5, on_date=TODAY)
    t_first = time.perf_counter() - t
    rss_loaded = rss_mb()

    # 已知答案的查询：拿库里某个 child 的向量加噪声当查询向量，问题文本取它正文里的几个词
    probes = []
    for _ in range(args.queries):
        d = int(rng.integers(0, n_docs))
        cid = f"doc{d:08d}:p{int(rng.integers(0, PARENTS_PER_DOC)):04d}:c{int(rng.integers(0, CHILDREN_PER_PARENT)):03d}"
        c = idx.store.get(cid)
        v = idx.vectors.get_vectors([cid])[cid]
        q = v + 0.3 * corpus.vectors(1, rng)[0]
        words = idx.bm25.tokenize(c.text)
        probes.append((cid, q / np.linalg.norm(q), "".join(words[3:7])))

    def timed(fn) -> list[float]:
        out = []
        for i, p in enumerate(probes):
            t = time.perf_counter()
            fn(i, p)
            out.append((time.perf_counter() - t) * 1000)
        return out

    scope = r._allowed_ids({"user": {"level": 0}}, TODAY, st.retrieval)
    dense = timed(lambda i, p: idx.vectors.search(p[1], 50))
    dense_f = timed(lambda i, p: idx.vectors.search(p[1], 50, scope))
    sparse = timed(lambda i, p: idx.bm25.search(p[2], 50))
    sparse_f = timed(lambda i, p: idx.bm25.search(p[2], 50, scope))
    full = timed(lambda i, p: r.retrieve(p[2], top_k=5, on_date=TODAY, filters={"user": {"level": 0}}))
    rss_queries = rss_mb()                                     # 召回测量要做精确全扫描，会把全部向量读一遍：在它之前记常驻内存
    # 召回：ANN vs 精确全扫描（同一查询向量）
    rec = []
    hit_self = 0
    for cid, q, _ in probes[: args.recall_queries]:
        exact = {x for x, _ in idx.vectors.search_exact(q, 10)}
        ann = [x for x, _ in idx.vectors.search(q, 10)]
        rec.append(len(exact & set(ann)) / 10)
        hit_self += cid in ann
    sparse_hit = sum(p[0] in {x for x, _ in idx.bm25.search(p[2], 10)} for p in probes) / len(probes)
    out = {"children": len(idx.vectors), "docs": n_docs, "loadSeconds": round(t_load, 2), "firstQuerySeconds": round(t_first, 2),
           "rssMB": {"beforeLoad": round(rss0), "afterLoadAndFirstQuery": round(rss_loaded), "afterQueries": round(rss_queries), "afterExactScans": round(rss_mb())},
           "scope": repr(scope), "nprobes": idx.vectors._probes(), "nprobesRatio": args.nprobes_ratio, "partitions": idx.vectors._ann_meta().get("partitions"),
           "latencyMs": {k: {"p50": pct(v, 0.5), "p95": pct(v, 0.95)} for k, v in
                         {"dense": dense, "dense+filter": dense_f, "sparse": sparse, "sparse+filter": sparse_f, "hybridRetrieve": full}.items()},
           "recallAt10": round(float(np.mean(rec)), 3), "denseFindsSource@10": round(hit_self / len(rec), 3), "sparseFindsSource@10": round(sparse_hit, 3)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    (args.dir / f"bench_query_{args.nprobes_ratio}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dir", type=Path, required=True)
    b.add_argument("--children", type=int, default=1_000_000)
    b.add_argument("--batch", type=int, default=20000)
    b.add_argument("--workers", type=int, default=8, help="jieba 分词进程数")
    b.add_argument("--dim", type=int, default=1024)
    b.add_argument("--min-free-gb", type=float, default=25, help="磁盘剩余低于它就中止（建索引还要临时空间）")
    q = sub.add_parser("query")
    q.add_argument("--dir", type=Path, required=True)
    q.add_argument("--queries", type=int, default=100)
    q.add_argument("--recall-queries", type=int, default=20)
    q.add_argument("--nprobes-ratio", type=float, default=0.3)
    args = ap.parse_args()
    import logging

    logging.basicConfig(level=logging.WARNING)
    {"build": build, "query": query}[args.cmd](args)


if __name__ == "__main__":
    main()
