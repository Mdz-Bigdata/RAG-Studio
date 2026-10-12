"""命令行：ingest / query / ask / eval / mine-negatives / image-search / popo-export / stats。

    rag ingest data/raw                  # 解析 + 切片 + 建索引
    rag query "K8s Ingress 502 怎么排查"  # 只看检索链路（每路 / 融合 / 精排 / 上下文）
    rag ask "年假可以跨年吗"               # 检索 + LLM 回答（需配置 LLM）
    rag eval data/eval/golden_qa.jsonl   # 分阶段消融评测 + 错误归因
    rag image-search "销售额柱状图"        # 文搜图；--image a.png 图搜图；两者都给 = 图文混合；--to-text 图搜文
    rag popo-export report.pdf           # 为 MinerU-Popo 准备输入
    rag stats
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import Settings

app = typer.Typer(help="生产级 RAG 检索优化参考实现", no_args_is_help=True, add_completion=False)
console = Console()


def _settings(config: str | None, lite: bool) -> Settings:
    if lite:
        return Settings.lite()
    return Settings.load(config)


def _load_index(settings: Settings):
    from .index import Index

    if not Index.exists(settings.paths.index_dir):
        console.print(f"[red]索引不存在：{settings.paths.index_dir}，请先执行 rag ingest[/red]")
        raise typer.Exit(1)
    return Index.load(settings)


def _llm(settings: Settings, aux: bool = False):
    from .generation import build_llm

    try:
        return build_llm(settings.generation.llm, aux=aux)
    except Exception as exc:
        console.print(f"[yellow]LLM 不可用（{exc}），将以无 LLM 模式运行[/yellow]")
        return None


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., help="文件或目录"),
    config: str | None = typer.Option(None, "--config", "-c"),
    lite: bool = typer.Option(False, "--lite", help="哈希向量 + BM25，无模型下载（冒烟 / 测试）"),
    fresh: bool = typer.Option(False, "--fresh", help="丢弃已有索引重建"),
) -> None:
    from .index import Index
    from .ingest import IngestPipeline

    settings = _settings(config, lite)
    index = Index(settings) if fresh or not Index.exists(settings.paths.index_dir) else Index.load(settings)
    pipeline = IngestPipeline(settings, index=index, llm=_llm(settings, aux=True) if settings.enrich.contextual_summary or settings.enrich.hypothetical_questions else None)
    report = pipeline.ingest_paths(paths)
    console.print(f"[green]{report.summary()}[/green]")
    for path, err in report.failed:
        console.print(f"[red]失败[/red] {path}: {err}")
    console.print(json.dumps(index.stats(), ensure_ascii=False))


@app.command()
def query(
    question: str,
    config: str | None = typer.Option(None, "--config", "-c"),
    lite: bool = typer.Option(False, "--lite"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    no_transform: bool = typer.Option(False, "--no-transform", help="关闭查询改写 / HyDE"),
    show_routes: bool = typer.Option(False, "--routes", help="打印每一路 top-10"),
) -> None:
    from .retrieval import HybridRetriever

    settings = _settings(config, lite)
    index = _load_index(settings)
    retriever = HybridRetriever(settings, index, llm=None if no_transform else _llm(settings, aux=True))
    result = retriever.retrieve(question, top_k=top_k, use_transform=not no_transform)
    _print_trace(result, index, show_routes)


@app.command()
def ask(
    question: str,
    config: str | None = typer.Option(None, "--config", "-c"),
    lite: bool = typer.Option(False, "--lite"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
) -> None:
    from .generation import AnswerGenerator
    from .retrieval import HybridRetriever

    settings = _settings(config, lite)
    index = _load_index(settings)
    llm = _llm(settings)
    retriever = HybridRetriever(settings, index, llm=_llm(settings, aux=True) if llm else None)
    result = retriever.retrieve(question, top_k=top_k)
    answer = AnswerGenerator(settings.generation, llm).answer(result)
    console.rule("回答")
    console.print(answer.text)
    console.rule("引用")
    for i, ctx in enumerate(answer.contexts, start=1):
        mark = "*" if i in answer.citations else " "
        console.print(f"{mark}[{i}] {ctx.citation_label()}  score={ctx.score:.3f}")
    if answer.usage:
        console.print(f"[dim]model={answer.model} usage={answer.usage}[/dim]")


@app.command("eval")
def eval_cmd(
    golden: Path = typer.Argument(None),
    config: str | None = typer.Option(None, "--config", "-c"),
    lite: bool = typer.Option(False, "--lite"),
    output: Path | None = typer.Option(None, "--output", "-o", help="把报告写成 Markdown"),
    no_transform: bool = typer.Option(True, "--no-transform/--transform", help="评测默认关闭 LLM 查询变换，保证可复现"),
) -> None:
    from .eval import EvalRunner, load_golden_set
    from .retrieval import HybridRetriever

    settings = _settings(config, lite)
    golden = golden or settings.eval.golden_set
    examples = load_golden_set(golden)
    index = _load_index(settings)
    retriever = HybridRetriever(settings, index, llm=None if no_transform else _llm(settings, aux=True))
    report = EvalRunner(retriever, ks=settings.eval.ks, ablation=settings.eval.ablation).run(examples)
    md = report.to_markdown()
    console.print(md)
    if output:
        output.write_text(md, encoding="utf-8")
        console.print(f"[green]已写入 {output}[/green]")
    failed = [e for e in report.examples if e.error_type]
    if failed:
        console.rule("失败样例")
        for e in failed:
            console.print(f"- [{e.error_type}] {e.question}")


@app.command("mine-negatives")
def mine_negatives(
    annotations: Path = typer.Argument(..., help="训练标注 JSONL：优先 relevant_chunk_ids，可用 answer_keywords"),
    output: Path = typer.Option(..., "--output", "-o", help="新导出目录（存在时拒绝覆盖）"),
    held_out: Path | None = typer.Option(None, "--held-out", help="独立评测集；其归一化问题不参与挖掘"),
    config: str | None = typer.Option(None, "--config", "-c"),
    mining_config: Path | None = typer.Option(None, "--mining-config", help="MiningConfig 参数 JSON"),
    lite: bool = typer.Option(False, "--lite", help="哈希向量 + BM25，仅流程自检"),
    no_teacher: bool = typer.Option(False, "--no-teacher", help="只召回候选，全部标为待人工审核"),
    allow_unverified: bool = typer.Option(False, "--allow-unverified", help="明确导出已人工审核的无教师候选"),
    synthetic: int | None = typer.Option(None, "--synthetic", min=0, max=4, help="每个问题让文本模型改写正例生成的负例数；正文会发送给文本模型"),
    include_synthetic: bool = typer.Option(False, "--include-synthetic", help="明确导出已人工筛选的改写负例"),
) -> None:
    from .eval import load_golden_set
    from .retrieval import HybridRetriever
    from .training import HardNegativeMiner, MiningConfig, export_training

    try:
        if output.exists():
            raise ValueError(f"导出目录已存在：{output}，请选择新目录")
        params = json.loads(mining_config.read_text(encoding="utf-8")) if mining_config else {}
        if no_teacher:
            params["use_teacher"] = False
        if synthetic is not None:
            params["synthetic_negatives"] = synthetic
        cfg = MiningConfig.model_validate(params)
        examples = load_golden_set(annotations)
        held = load_golden_set(held_out) if held_out else []
        if not examples:
            raise ValueError("训练标注不能为空")
        settings = _settings(config, lite)
        index = _load_index(settings)
        retriever = HybridRetriever(settings, index, llm=None)
        llm = _llm(settings, aux=True) if cfg.synthetic_negatives else None
        report = HardNegativeMiner(retriever, cfg, llm=llm).run(examples, held_out=held)
        counts = export_training(report, output, allow_unverified=allow_unverified, include_synthetic=include_synthetic)
    except (OSError, ValueError) as exc:
        console.print(f"[red]难负样本挖掘失败：{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(json.dumps({"stats": report.stats, "exported": counts, "output": str(output)}, ensure_ascii=False))
    for warning in report.warnings:
        console.print(f"[yellow]{warning}[/yellow]")
    if report.stats["review_required"] and not allow_unverified:
        console.print("[yellow]待审核候选仅保留在 mining-report.json；审核后可显式使用 --allow-unverified 导出。[/yellow]")
    if report.stats["synthetic_negatives"] and not include_synthetic:
        console.print("[yellow]改写负例仅保留在 mining-report.json 的 synthetic_neg；人工筛选后可显式使用 --include-synthetic 导出。[/yellow]")


@app.command()
def coverage(
    intents: Path = typer.Argument(None, help="意图文件：每行一个用户问题（.txt），或黄金集 .jsonl；省略则用 eval.golden_set"),
    config: str | None = typer.Option(None, "--config", "-c"),
    lite: bool = typer.Option(False, "--lite"),
    high: float = typer.Option(0.6, "--high", help="≥ 该分数为高匹配"),
    partial: float = typer.Option(0.35, "--partial", help="≥ 该分数为部分匹配"),
    output: Path = typer.Option(None, "--output", "-o", help="Markdown 报告输出路径"),
) -> None:
    """语料准备：意图空间 × 知识空间覆盖分析（哪些知识还没准备好、哪些材料与目标无关）。"""
    from .eval import CoverageAnalyzer
    from .retrieval import HybridRetriever

    settings = _settings(config, lite)
    index = _load_index(settings)
    path = intents or settings.eval.golden_set
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    items = [json.loads(ln) for ln in lines] if str(path).endswith(".jsonl") else lines
    report = CoverageAnalyzer(HybridRetriever(settings, index), high=high, partial=partial).run(items)
    md = report.to_markdown()
    console.print(md)
    if output:
        output.write_text(md, encoding="utf-8")
        console.print(f"\n报告已写入 {output}")


@app.command("image-search")
def image_search(
    query: str = typer.Argument("", help="文字查询（文搜图 / 图文混合）"),
    image: Path | None = typer.Option(None, "--image", "-i", help="查询图片（图搜图 / 图文混合 / 图搜文）"),
    to_text: bool = typer.Option(False, "--to-text", help="以图搜文：返回与图片相关的正文段落"),
    alpha: float | None = typer.Option(None, "--alpha", help="图文混合的文本权重 α"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    config: str | None = typer.Option(None, "--config", "-c"),
    lite: bool = typer.Option(False, "--lite"),
) -> None:
    """多模态检索：文搜图 / 图搜图 / 图文混合 / 图搜文。"""
    from PIL import Image

    from .multimodal import MultimodalRetriever

    settings = _settings(config, lite)
    index = _load_index(settings)
    r = MultimodalRetriever(settings, index, vlm=None if lite else _llm(settings))
    img = Image.open(image).convert("RGB") if image else None
    if img is None and not query:
        console.print("[red]请提供文字查询或 --image[/red]")
        raise typer.Exit(1)
    if img is not None and to_text:
        res = r.image_to_text(img, query=query, top_k=top_k)
    elif img is not None and query:
        res = r.hybrid(query, img, alpha=alpha, top_k=top_k)
    elif img is not None:
        res = r.image_to_image(img, top_k=top_k)
    else:
        res = r.text_to_image(query, top_k=top_k)
    console.rule(f"{res.mode}（图文向量：{res.embedder.get('model') or index.images.stats().get('image_model')}）")
    for n in res.notes:
        console.print(f"  [dim]{n}[/dim]")
    if res.query_description:
        console.print(f"  查询图片识别：{res.query_description[:200]}")
    t = Table("#", "score", "routes", "图片", "来源", "摘要")
    for h in res.images:
        routes = " ".join(f"{k}#{v.rank}" for k, v in h.routes.items())
        t.add_row(str(h.rank), f"{h.score:.3f}", routes, str(index.images.path_of(h.record)), h.record.label()[:40], h.record.brief(50))
    if res.images:
        console.print(t)
    for i, ctx in enumerate(res.contexts, start=1):
        console.print(f"[bold][{i}][/bold] {ctx.citation_label()}  score={ctx.score:.3f}")
        console.print("   " + ctx.chunk.text.replace("\n", " ")[:200])


@app.command("popo-export")
def popo_export(
    pdf: Path,
    config: str | None = typer.Option(None, "--config", "-c"),
) -> None:
    """用 MinerU 解析 PDF 并生成 MinerU-Popo 所需输入（post-process/ + eval_pdf_dir/ + outputs/label_normalization/）。"""
    from .ingest import IngestPipeline

    settings = _settings(config, False)
    paths = IngestPipeline(settings).export_popo_inputs(pdf)
    for k, v in paths.items():
        console.print(f"{k}: {v}")
    console.print(
        "\n下一步（在 MinerU-Popo 仓库内）：\n"
        f"  export POPO_MODEL_PATH=models/MinerU-Popo\n"
        f"  python post_processing/run_inference.py --input-dir {paths['normalized'].parent} --output-dir <popo>/outputs/inference/mineru --limit 0\n"
        f"  python post_processing/get_json_tree.py --input-dir <popo>/outputs/inference/mineru --output-dir <popo>/outputs/build_tree/mineru --txt-dir <popo>/outputs/build_tree_txt/mineru\n"
        "然后在 config.yaml 打开 parsing.popo.enabled，重新 rag ingest 即使用文档树。"
    )


@app.command("download-reranker")
def download_reranker(
    model: str = typer.Option(None, "--model", "-m", help="默认取 config.yaml 的 retrieval.rerank.model（BAAI/bge-reranker-v2-m3，约 2.3GB）"),
    config: str | None = typer.Option(None, "--config", "-c"),
) -> None:
    """把本地重排模型的权重下载到 Hugging Face 缓存。国内网络可先 `export HF_ENDPOINT=https://hf-mirror.com`。"""
    from .retrieval.rerank import cross_encoder_cached, download_cross_encoder

    name = model or _settings(config, False).retrieval.rerank.model
    if cross_encoder_cached(name):
        console.print(f"[green]{name} 已在本机缓存，无需下载[/green]")
        return
    console.print(f"正在下载 {name} …")
    console.print(f"[green]完成：{download_cross_encoder(name)}[/green]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """启动 Web 后端（FastAPI）。存在 frontend/dist 时同一端口直接托管前端页面。"""
    import uvicorn

    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


@app.command()
def stats(config: str | None = typer.Option(None, "--config", "-c"), lite: bool = typer.Option(False, "--lite")) -> None:
    settings = _settings(config, lite)
    index = _load_index(settings)
    console.print(json.dumps(index.stats(), ensure_ascii=False, indent=2))
    t = Table("doc_id", "title", "parser", "version", "valid_to")
    for doc_id, md in index.store.docs.items():
        t.add_row(doc_id, (md.title or "")[:40], md.parser, md.version or "-", str(md.valid_to or "-"))
    console.print(t)


def _print_trace(result, index, show_routes: bool) -> None:
    tr = result.trace
    console.rule("查询层")
    for q in tr.queries:
        console.print(f"  · {q}")
    if tr.hyde_document:
        console.print(f"  HyDE: {tr.hyde_document[:120]}…")
    for n in tr.notes:
        console.print(f"  [dim]{n}[/dim]")
    if show_routes:
        console.rule("召回层（每路 top-10）")
        for route, items in tr.routes.items():
            console.print(f"[bold]{route}[/bold]")
            for sc in items[:10]:
                c = index.store.get(sc.chunk_id)
                console.print(f"   {sc.rank:>2}. {sc.score:.4f} {sc.chunk_id}  {(c.text if c else '')[:50]!r}")
    console.rule(f"融合层（候选 {len(tr.fused)}）→ 精排层（top {len(tr.reranked)}）")
    for sc in tr.reranked:
        c = index.store.get(sc.chunk_id)
        console.print(f"  {sc.rank:>2}. {sc.score:.4f} {sc.chunk_id}  {(c.text if c else '')[:60]!r}")
    console.rule("上下文（送入 LLM 的顺序）")
    for i, ctx in enumerate(result.contexts, start=1):
        console.print(f"[bold][{i}][/bold] {ctx.citation_label()}  score={ctx.score:.3f}  tokens={ctx.chunk.token_count}")
        console.print("   " + ctx.chunk.text.replace("\n", " ")[:200])
    console.print(f"[dim]timings(ms): { {k: round(v, 1) for k, v in tr.timings_ms.items()} }[/dim]")


if __name__ == "__main__":
    app()
