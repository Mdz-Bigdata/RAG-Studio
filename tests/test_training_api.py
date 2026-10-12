"""难负样本 API 的请求约束、索引读锁和训练 / 评测隔离，不依赖真实模型。"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from app.api import eval as eval_api
    from app.db import get_db

    class RepositoryStub:
        def __init__(self, _db):
            pass

        def get_kb(self, kb_id):
            return None if kb_id == "missing" else object()

    monkeypatch.setattr(eval_api, "Repository", RepositoryStub)
    app = FastAPI()
    app.include_router(eval_api.router, prefix="/api/v1/eval")
    app.dependency_overrides[get_db] = lambda: object()
    with TestClient(app) as session:
        yield session


def test_mining_api_forwards_explicit_labels_and_reserves(client, monkeypatch):
    from app.services import eval_service

    captured = {}

    def mine(kb_id, examples, *, held_out, config):
        captured.update(kb_id=kb_id, examples=examples, held_out=held_out, config=config)
        return {"items": [], "kb_id": kb_id}

    monkeypatch.setattr(eval_service, "mine_hard_negatives", mine)
    response = client.post("/api/v1/eval/demo/hard-negatives", json={
        "examples": [{"question": "退货几天内可申请", "relevant_chunk_ids": ["refund:p0"]}],
        "heldOut": [{"question": "独立评测的问题"}], "config": {"use_teacher": False},
    })
    assert response.status_code == 200, response.text
    assert response.json() == {"items": [], "kb_id": "demo"}
    assert captured["examples"][0]["relevant_chunk_ids"] == ["refund:p0"]
    assert captured["held_out"][0]["question"] == "独立评测的问题"
    assert captured["config"].candidate_pool == 100 and not captured["config"].use_teacher


@pytest.mark.parametrize("body", [
    {}, {"examples": []}, {"examples": [{"question": "test"}] * 501},
    {"examples": [{"question": "test"}], "heldOut": [{"question": "test"}] * 501},
    {"examples": [{"question": "test"}], "config": {"candidate_pool": 9}},
    {"examples": [{"question": "test"}], "config": {"rank_min": 20, "rank_max": 10}},
    {"examples": [{"question": "test"}], "config": {"absolute_margin": -0.1}},
    {"examples": [{"question": "test"}], "config": {"unknown_field": True}},
])
def test_mining_request_bounds(client, body):
    assert client.post("/api/v1/eval/demo/hard-negatives", json=body).status_code == 422


def test_missing_kb_and_mining_error(client, monkeypatch):
    from app.services import eval_service

    body = {"examples": [{"question": "test", "relevant_chunk_ids": ["p0"]}]}
    assert client.post("/api/v1/eval/missing/hard-negatives", json=body).status_code == 404

    def bad_labels(*_args, **_kwargs):
        raise ValueError("标注无法解析")

    monkeypatch.setattr(eval_service, "mine_hard_negatives", bad_labels)
    response = client.post("/api/v1/eval/demo/hard-negatives", json=body)
    assert response.status_code == 400 and response.json()["detail"] == "标注无法解析"

    def private_error(*_args, **_kwargs):
        raise RuntimeError("/private/index-path synthetic-private-credential")

    monkeypatch.setattr(eval_service, "mine_hard_negatives", private_error)
    response = client.post("/api/v1/eval/demo/hard-negatives", json=body)
    assert response.status_code == 500
    assert "synthetic-private-credential" not in response.text
    assert "/private/" not in response.text


def test_service_holds_index_read_lock_and_preserves_provenance(built_index, monkeypatch):
    from app.services import eval_service, index_service, retrieval_service
    from prism.retrieval import HybridRetriever
    from prism.training import MiningConfig

    retriever = HybridRetriever(built_index.settings, built_index)
    active = []

    @contextmanager
    def reading(kb_id):
        active.append(kb_id)
        try:
            yield
        finally:
            active.pop()

    def get_retriever(kb_id):
        assert active == [kb_id]
        return retriever

    monkeypatch.setattr(index_service, "reading", reading)
    monkeypatch.setattr(retrieval_service, "retriever", get_retriever)
    original = {cid: chunk.model_dump_json() for cid, chunk in built_index.store.chunks.items()}
    positive = next(chunk for chunk in built_index.store.chunks.values() if chunk.kind == "child" and "SO-8817" in chunk.text)
    data = eval_service.mine_hard_negatives("demo", [
        {"question": "订单 SO-8817 的状态", "relevant_chunk_ids": [positive.chunk_id]},
        {"question": " 保留问题？ ", "relevant_chunk_ids": [positive.chunk_id]},
    ], held_out=[{"question": "保留问题"}], config=MiningConfig(use_teacher=False, num_negatives=1))
    assert not active
    assert data["kb_id"] == "demo"
    assert data["runtimeModels"]["embedderOffline"] and not data["models"]["teacherRequested"]
    assert any(item["reason"] == "held_out_question" for item in data["skipped"])
    assert data["items"] and all(item["query"] != " 保留问题？ " and item["requires_review"] for item in data["items"])
    assert {cid: chunk.model_dump_json() for cid, chunk in built_index.store.chunks.items()} == original
