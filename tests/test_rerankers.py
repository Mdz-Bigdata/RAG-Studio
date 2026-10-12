"""Reranker：默认 BAAI/bge-reranker-v2-m3（本地交叉编码器）+ Cohere Rerank v2 + 通用 /rerank 接口。

- Cohere / 通用接口：起一个本地 HTTP 桩，校验请求的路径、鉴权头、请求体字段和响应解析，不访问外网；
- 本地交叉编码器：注入假的加载器验证懒加载、分数归一化、兜底与提示，不加载真实权重；
- `RAG_TEST_RERANKER=1 pytest tests/test_rerankers.py -k live`：用本机缓存的真实 bge-reranker-v2-m3 打分。
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from prism.config import RerankConfig, Settings
from prism.retrieval import rerank as R
from prism.retrieval.rerank import (
    APIReranker,
    CohereReranker,
    CrossEncoderReranker,
    LexicalReranker,
    LLMReranker,
    NoopReranker,
    build_api_reranker,
    build_reranker,
    is_cohere_url,
)

CANDIDATES = [("a", "今天天气不错"), ("b", "年假可以跨年使用，最晚次年 3 月 31 日前休完"), ("c", "K8s Ingress 502 排查"), ("d", "年假申请流程")]


# --------------------------------------------------------------------------------------
# HTTP 桩
# --------------------------------------------------------------------------------------
class _Stub:
    """按脚本依次返回响应，并记录收到的请求。"""

    def __init__(self, responses: list[tuple[int, dict, dict]]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                stub.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
                status, payload, headers = stub.responses.pop(0) if stub.responses else (500, {"message": "no scripted response"}, {})
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):  # 静音
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    created: list[_Stub] = []

    def make(responses):
        s = _Stub(responses)
        created.append(s)
        return s

    yield make
    for s in created:
        s.close()


COHERE_OK = {"results": [{"index": 1, "relevance_score": 0.98}, {"index": 3, "relevance_score": 0.41}], "id": "x", "meta": {"billed_units": {"search_units": 1}}}


# --------------------------------------------------------------------------------------
# Cohere Rerank v2
# --------------------------------------------------------------------------------------
def test_cohere_request_and_response_shape(stub):
    server = stub([(200, COHERE_OK, {})])
    rr = CohereReranker("co-test-key", "rerank-v3.5", f"{server.url}/v2", max_tokens_per_doc=1024)
    out = rr.rerank("年假可以跨年吗", CANDIDATES, top_k=2)
    assert out == [("b", 0.98), ("d", 0.41)]                      # index → 候选 id，分数降序
    req = server.requests[0]
    assert req["path"] == "/v2/rerank"
    assert req["headers"]["authorization"] == "Bearer co-test-key"
    assert req["headers"]["x-client-name"] == "prism"
    assert req["body"] == {"model": "rerank-v3.5", "query": "年假可以跨年吗", "documents": [t for _, t in CANDIDATES], "top_n": 2, "max_tokens_per_doc": 1024}


def test_cohere_defaults_and_api_key_sources(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("CO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="COHERE_API_KEY"):
        CohereReranker()
    monkeypatch.setenv("COHERE_API_KEY", "from-env")
    rr = CohereReranker()
    assert rr.api_url == "https://api.cohere.com/v2/rerank" and rr.model_name == "rerank-v4.0-pro" and rr.api_key == "from-env"
    assert "max_tokens_per_doc" not in rr.extra_payload          # 不传 = 用服务端默认 4096
    assert CohereReranker("explicit").api_key == "explicit"       # 显式参数优先于环境变量
    assert rr.rerank("q", [], 5) == []                            # 空候选不发请求


def test_cohere_errors_are_readable_and_429_is_retried(stub, monkeypatch):
    monkeypatch.setattr(R.time, "sleep", lambda _s: None)
    bad = stub([(401, {"message": "invalid api token"}, {})])
    with pytest.raises(RuntimeError, match="401.*invalid api token"):
        CohereReranker("wrong", "rerank-v3.5", f"{bad.url}/v2").rerank("q", CANDIDATES, 2)
    assert len(bad.requests) == 1                                 # 鉴权错误不重试

    flaky = stub([(429, {"message": "rate limited"}, {"Retry-After": "0"}), (200, COHERE_OK, {})])
    out = CohereReranker("k", "rerank-v3.5", f"{flaky.url}/v2").rerank("q", CANDIDATES, 2)
    assert [cid for cid, _ in out] == ["b", "d"] and len(flaky.requests) == 2


def test_api_reranker_batches_large_candidate_sets(stub):
    # 超过单次上限时分批，相关度分数逐对计算，可以直接合并排序
    server = stub([(200, {"results": [{"index": 0, "relevance_score": 0.2}, {"index": 1, "relevance_score": 0.9}]}, {}), (200, {"results": [{"index": 0, "score": 0.5}]}, {})])
    rr = APIReranker(server.url, None, "m", max_documents=2)
    out = rr.rerank("q", [("a", "A"), ("b", "B"), ("c", "C")], top_k=3)
    assert out == [("b", 0.9), ("c", 0.5), ("a", 0.2)]            # 第二批用的是 score 字段（Jina / SiliconFlow 的变体）
    assert [len(r["body"]["documents"]) for r in server.requests] == [2, 1]
    assert "authorization" not in server.requests[0]["headers"]   # 没给 key 就不带鉴权头


def test_url_based_factory_and_config_provider(monkeypatch):
    assert is_cohere_url("https://api.cohere.com/v2") and is_cohere_url("https://api.cohere.com/v2/rerank")
    assert not is_cohere_url("https://api.jina.ai/v1") and not is_cohere_url("https://evil.example/cohere.com") and not is_cohere_url(None)
    assert isinstance(build_api_reranker("https://api.cohere.com/v2", "k", "rerank-v3.5"), CohereReranker)
    jina = build_api_reranker("https://api.jina.ai/v1", "k", "jina-reranker-v3")
    assert type(jina) is APIReranker and jina.api_url == "https://api.jina.ai/v1/rerank"

    monkeypatch.setenv("COHERE_API_KEY", "env-key")
    rr = build_reranker(RerankConfig(provider="cohere"))
    assert isinstance(rr, CohereReranker) and rr.model_name == "rerank-v4.0-pro" and rr.api_url == "https://api.cohere.com/v2/rerank"
    rr = build_reranker(RerankConfig(provider="cohere", cohere_model="rerank-v3.5", api_key="cfg-key"))
    assert rr.model_name == "rerank-v3.5" and rr.api_key == "cfg-key"
    # auto + Cohere 地址 → Cohere；auto + 其它地址 → 通用协议
    assert isinstance(build_reranker(RerankConfig(api_url="https://api.cohere.com/v2", api_key="k")), CohereReranker)
    assert type(build_reranker(RerankConfig(api_url="https://api.siliconflow.cn/v1", api_key="k"))) is APIReranker


# --------------------------------------------------------------------------------------
# 本地交叉编码器（默认 bge-reranker-v2-m3）
# --------------------------------------------------------------------------------------
@pytest.fixture
def fake_cross_encoder(monkeypatch):
    """不加载真实权重：logit = 文本里"年假"出现的次数 × 4 − 3。记录加载与打分调用。"""
    calls = {"load": [], "score": 0, "cached": True}
    monkeypatch.setattr(R, "cross_encoder_available", lambda: True)
    monkeypatch.setattr(R, "cross_encoder_cached", lambda model: calls["cached"])

    def load(model, device, local_only):
        calls["load"].append((model, local_only))
        return ("tok", "net", device or "cpu")

    def logits(bundle, query, texts, max_length, batch_size):
        calls["score"] += 1
        return [text.count("年假") * 4.0 - 3.0 for text in texts]

    monkeypatch.setattr(R, "_load_cross_encoder", load)
    monkeypatch.setattr(R, "_cross_encoder_logits", logits)
    return calls


def test_default_model_is_bge_reranker_v2_m3():
    assert RerankConfig().model == "BAAI/bge-reranker-v2-m3" == R.DEFAULT_RERANK_MODEL
    assert Settings().retrieval.rerank.provider == "auto"


def test_auto_prefers_local_cross_encoder(fake_cross_encoder, monkeypatch):
    rr = build_reranker(RerankConfig())
    assert isinstance(rr, CrossEncoderReranker) and rr.model_name == "BAAI/bge-reranker-v2-m3"
    assert isinstance(rr.fallback, LexicalReranker) and rr.state == "idle" and fake_cross_encoder["load"] == []   # 懒加载
    assert isinstance(build_reranker(RerankConfig(), llm=object()).fallback, LLMReranker)
    # 没装 torch / transformers → LLM → 词法
    monkeypatch.setattr(R, "cross_encoder_available", lambda: False)
    assert isinstance(build_reranker(RerankConfig(), llm=object()), LLMReranker)
    assert isinstance(build_reranker(RerankConfig()), LexicalReranker)
    assert isinstance(build_reranker(RerankConfig(enabled=False)), NoopReranker)


def test_cross_encoder_scores_are_sigmoid_normalized_and_sorted(fake_cross_encoder):
    rr = CrossEncoderReranker(fallback=LexicalReranker())
    out = rr.rerank("年假", CANDIDATES, top_k=3)
    assert [cid for cid, _ in out] == ["b", "d", "a"] or [cid for cid, _ in out][:2] == ["b", "d"]
    scores = dict(out)
    assert scores["b"] == pytest.approx(1 / (1 + 2.718281828 ** -1), abs=1e-6)   # logit 1 → 0.731
    assert scores["a"] == pytest.approx(1 / (1 + 2.718281828 ** 3), abs=1e-6)    # logit -3 → 0.047
    assert all(0.0 < s < 1.0 for s in scores.values())
    assert rr.state == "ready" and fake_cross_encoder["load"] == [("BAAI/bge-reranker-v2-m3", True)]   # 有缓存：只读本地，不联网
    rr.rerank("年假", CANDIDATES, top_k=1)
    assert len(fake_cross_encoder["load"]) == 1 and rr.pop_note() is None


def test_missing_weights_fall_back_without_blocking(fake_cross_encoder):
    fake_cross_encoder["cached"] = False
    rr = CrossEncoderReranker(fallback=LexicalReranker(), auto_download=False)
    out = rr.rerank("年假 流程", CANDIDATES, top_k=2)
    assert out and fake_cross_encoder["score"] == 0 and fake_cross_encoder["load"] == []      # 用的是词法兜底，没碰模型
    note = rr.pop_note()
    assert "尚未下载" in note and "lexical" in note and rr.pop_note() is None
    assert rr.status()["state"] == "idle" and rr.status()["cached"] is False

    # auto_download：后台加载，当前请求仍由兜底完成；加载完成后切到模型
    gate = threading.Event()
    real_load = R._load_cross_encoder

    def slow_load(model, device, local_only):
        gate.wait(5)
        return real_load(model, device, local_only)

    R._load_cross_encoder = slow_load
    try:
        bg = CrossEncoderReranker(fallback=LexicalReranker(), auto_download=True)
        bg.rerank("年假", CANDIDATES, top_k=2)
        assert bg.state == "loading" and "后台下载" in bg.pop_note()
        bg.rerank("年假", CANDIDATES, top_k=2)
        assert "正在下载" in bg.pop_note() and fake_cross_encoder["score"] == 0
        gate.set()
        for _ in range(200):
            if bg.state == "ready":
                break
            threading.Event().wait(0.01)
        assert bg.state == "ready"
        assert [cid for cid, _ in bg.rerank("年假", CANDIDATES, top_k=2)] == ["b", "d"] and fake_cross_encoder["score"] == 1
    finally:
        R._load_cross_encoder = real_load
        gate.set()


def test_load_failure_is_reported_and_blocking_mode_raises(fake_cross_encoder, monkeypatch):
    def boom(model, device, local_only):
        raise OSError("weights corrupted")

    monkeypatch.setattr(R, "_load_cross_encoder", boom)
    soft = CrossEncoderReranker(fallback=LexicalReranker())
    assert soft.rerank("年假", CANDIDATES, top_k=2)                  # 兜底照常返回
    assert "加载失败" in soft.pop_note() and soft.state == "failed" and "weights corrupted" in soft.status()["error"]
    with pytest.raises(OSError):                                     # 没有兜底（CLI 显式选择）：错误直接抛出
        CrossEncoderReranker().rerank("年假", CANDIDATES, top_k=2)


def test_retriever_surfaces_reranker_note_in_trace(built_index, fake_cross_encoder):
    from prism.retrieval import HybridRetriever

    fake_cross_encoder["cached"] = False
    rr = CrossEncoderReranker(fallback=LexicalReranker(), auto_download=False)
    result = HybridRetriever(built_index.settings, built_index, reranker=rr).retrieve("年假可以跨年使用吗", top_k=3, use_transform=False)
    assert result.contexts and any("尚未下载" in n for n in result.trace.notes)


def test_kill_switch_env(monkeypatch):
    monkeypatch.setenv("RAG_DISABLE_LOCAL_RERANKER", "1")
    assert R.cross_encoder_available() is False


# --------------------------------------------------------------------------------------
# 真实模型（需要显式打开；权重须已在本机缓存）
# --------------------------------------------------------------------------------------
LIVE = os.environ.get("RAG_TEST_RERANKER") == "1"


@pytest.mark.skipif(not LIVE, reason="设置 RAG_TEST_RERANKER=1 才加载真实的 bge-reranker-v2-m3")
def test_live_bge_reranker_v2_m3_ranks_relevant_first():
    assert R.cross_encoder_available() and R.cross_encoder_cached("BAAI/bge-reranker-v2-m3"), "先执行 rag download-reranker"
    rr = CrossEncoderReranker()
    docs = [
        ("outline", "第4集:AgentScope 工具调用。ReAct Agent 架构，自定义工具开发，智能推理流程。"),
        ("price", "课程价格:¥399，10集完整教程，从零到一构建企业级语音助手。"),
        ("people", "适合人群：Python 后端开发者、AI Agent 爱好者、语音交互应用开发者。"),
        ("noise", "K8s Ingress 502 排查手册：先看 upstream 健康检查，再看 backend timeout 配置。"),
    ]
    for query, expected in (("第4集讲什么", "outline"), ("这门课多少钱", "price"), ("哪些人适合学", "people"), ("网关 502 怎么查", "noise")):
        out = rr.rerank(query, docs, top_k=4)
        assert out[0][0] == expected, (query, out)
        # 绝对分数随问法波动很大（同义改写的相关对只有 0.1~0.2），但和无关文档之间差两个数量级以上
        assert out[0][1] > 0.1 and out[0][1] > 20 * out[1][1], (query, out)
    # 库里没有答案的问题：所有候选都接近 0 —— min_score 取 0.01~0.05 就能干净地拒答
    assert max(s for _, s in rr.rerank("公司食堂几点开饭", docs, top_k=4)) < 0.01
    assert rr.state == "ready" and rr.status()["device"] in ("cpu", "mps", "cuda")
