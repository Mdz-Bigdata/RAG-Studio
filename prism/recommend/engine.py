"""个性化推荐（LanceDB）：rec_items（候选条目）+ rec_events（用户交互），兴趣向量召回 → 热度加权 → MMR 多样化。

- 条目：知识库里的文档 / 章节（parent）/ 生成的问题 / 图片，向量与检索用同一个文本向量空间；
- 交互：ask（提问，存问题向量）、view / click / cite（看过、点过、引用过）、like / dislike（显式反馈）；
  赞 / 踩是"评价"：同一用户对同一条目只能评价一次，赞踩互斥（查重和写入在同一把锁里，连点、两个一起点也只记第一次）；
- 用户兴趣向量 = Σ 权重 × 时间衰减 × 条目（或问题）向量，dislike 以负权重扣除；再叠加长期记忆里的偏好（可选）；
- 冷启动（还没有交互）按热度推荐；每条推荐都带"为什么推荐"——与它最接近的那次交互。
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import RecommendConfig
from ..index.embeddings import BaseEmbedder
from ..lance import connect, open_table, sql_in, sql_str, table_info, write_lock

logger = logging.getLogger(__name__)

ITEMS, EVENTS = "rec_items", "rec_events"
EVENT_WEIGHTS = {"like": 1.0, "ask": 0.8, "click": 0.6, "cite": 0.5, "view": 0.3, "dislike": -1.0}
VOTE_KINDS = ("like", "dislike")
_DAY = 86400.0
_VOTE_LOCK = threading.Lock()


class AlreadyVoted(ValueError):
    """同一用户对同一条目已经评价过（赞 / 踩只能二选一、只能一次）。vote 是之前那次评价。"""

    def __init__(self, item_id: str, vote: str) -> None:
        super().__init__(f"已经{'点赞' if vote == 'like' else '点踩'}过这条推荐：每人每条只能评价一次，赞和踩不能同时选")
        self.item_id = item_id
        self.vote = vote


@dataclass
class RecItem:
    id: str
    kb_id: str
    kind: str                 # doc | section | question | image
    ref_id: str               # doc_id / chunk_id / image_id
    title: str
    text: str = ""
    tags: list[str] = field(default_factory=list)
    popularity: float = 0.0
    created_at: float = 0.0


def _item_schema(dim: int):
    import pyarrow as pa

    return pa.schema([pa.field("id", pa.string()), pa.field("kb_id", pa.string()), pa.field("kind", pa.string()), pa.field("ref_id", pa.string()),
                      pa.field("title", pa.string()), pa.field("text", pa.string()), pa.field("tags", pa.list_(pa.string())),
                      pa.field("popularity", pa.float32()), pa.field("created_at", pa.float64()), pa.field("vector", pa.list_(pa.float32(), dim))])


def _event_schema(dim: int):
    import pyarrow as pa

    return pa.schema([pa.field("id", pa.string()), pa.field("user_id", pa.string()), pa.field("item_id", pa.string()), pa.field("kb_id", pa.string()),
                      pa.field("kind", pa.string()), pa.field("weight", pa.float32()), pa.field("text", pa.string()), pa.field("ts", pa.float64()),
                      pa.field("vector", pa.list_(pa.float32(), dim))])


_ITEM_COLS = list(RecItem.__dataclass_fields__)
_EVENT_COLS = ["id", "user_id", "item_id", "kb_id", "kind", "weight", "text", "ts"]


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


class LanceRecommender:
    def __init__(self, directory: str | Path, embedder: BaseEmbedder, cfg: RecommendConfig | None = None) -> None:
        self.directory = Path(directory)
        self.embedder = embedder
        self.cfg = cfg or RecommendConfig()
        self._meta = self.directory / "rec.meta.json"
        self._check_embedder()

    @property
    def db(self):
        return connect(self.directory)

    @property
    def dim(self) -> int:
        return int(self.embedder.dim or len(self.embedder.embed_documents(["维度探测"])[0]))

    def _sig(self) -> dict[str, Any]:
        return {"embedder": self.embedder.name, "model": getattr(self.embedder, "model_name", None), "dim": self.dim}

    def _items(self, create: bool = False):
        t = open_table(self.db, ITEMS)
        if t is None and create:
            t = self.db.create_table(ITEMS, schema=_item_schema(self.dim), exist_ok=True)
            self._meta.write_text(json.dumps(self._sig()), encoding="utf-8")
        return t

    def _events(self, create: bool = False):
        t = open_table(self.db, EVENTS)
        if t is None and create:
            t = self.db.create_table(EVENTS, schema=_event_schema(self.dim), exist_ok=True)
        return t

    def _check_embedder(self) -> None:
        """换了向量模型：条目按标题 + 正文重算；提问事件按问题文本重算；其余事件改用条目的新向量。"""
        items = self._items()
        if items is None:
            return
        built = json.loads(self._meta.read_text(encoding="utf-8")) if self._meta.exists() else {}
        sig = self._sig()
        if (built.get("embedder"), built.get("model"), built.get("dim")) == (sig["embedder"], sig["model"], sig["dim"]):
            return
        logger.info("向量模型已更换，重算推荐条目与交互向量")
        rows = items.search().select(_ITEM_COLS).limit(10_000_000).to_list()
        recs = [RecItem(**{k: r[k] for k in _ITEM_COLS}) for r in rows]
        with write_lock(self.directory, ITEMS):
            self.db.create_table(ITEMS, schema=_item_schema(sig["dim"]), mode="overwrite")
        self._meta.write_text(json.dumps(sig), encoding="utf-8")
        if recs:
            self.upsert_items(recs)
        events = self._events()
        if events is not None:
            erows = events.search().select(_EVENT_COLS).limit(10_000_000).to_list()
            with write_lock(self.directory, EVENTS):
                self.db.create_table(EVENTS, schema=_event_schema(sig["dim"]), mode="overwrite")
            vec_of = self._item_vectors([r["item_id"] for r in erows if r["item_id"]])
            asks = [r for r in erows if not r["item_id"] and r["text"]]
            ask_vecs = dict(zip((r["id"] for r in asks), self.embedder.embed_queries([r["text"] for r in asks]))) if asks else {}
            keep = [(r, vec_of.get(r["item_id"]) if r["item_id"] else ask_vecs.get(r["id"])) for r in erows]
            self._write_events([(r, v) for r, v in keep if v is not None])

    # ----------------------------------------------------------------------------------
    def upsert_items(self, items: list[RecItem], vectors: np.ndarray | None = None) -> int:
        if not items:
            return 0
        import pyarrow as pa

        if vectors is None:
            vectors = self.embedder.embed_documents([f"{it.title}\n{it.text}"[:1500] for it in items])
        now = time.time()
        cols: dict[str, list[Any]] = {k: [getattr(it, k) for it in items] for k in _ITEM_COLS}
        cols["created_at"] = [it.created_at or now for it in items]
        cols["vector"] = list(np.asarray(vectors, dtype=np.float32))
        data = pa.Table.from_pydict(cols, schema=_item_schema(np.asarray(vectors).shape[1]))
        tbl = self._items(create=True)
        with write_lock(self.directory, ITEMS):
            # 热度是交互累计出来的：同步条目时不能把它清零
            old = {r["id"]: r["popularity"] for r in tbl.search().where(sql_in("id", [it.id for it in items])).select(["id", "popularity"]).limit(len(items)).to_list()}
            if old:
                data = data.set_column(data.schema.get_field_index("popularity"), "popularity", pa.array([float(old.get(it.id, it.popularity)) for it in items], type=pa.float32()))
            tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(data)
        return len(items)

    def remove_items(self, *, kb_id: str | None = None, keep_ids: set[str] | None = None) -> int:
        """同步时删掉知识库里已不存在的条目。"""
        tbl = self._items()
        if tbl is None:
            return 0
        where = f"kb_id = {sql_str(kb_id)}" if kb_id else "true"
        ids = [r["id"] for r in tbl.search().where(where).select(["id"]).limit(10_000_000).to_list()]
        drop = [i for i in ids if keep_ids is None or i not in keep_ids]
        with write_lock(self.directory, ITEMS):
            for i in range(0, len(drop), 500):
                tbl.delete(sql_in("id", drop[i:i + 500]))
        return len(drop)

    def _item_vectors(self, ids: list[str]) -> dict[str, np.ndarray]:
        tbl = self._items()
        if tbl is None or not ids:
            return {}
        out: dict[str, np.ndarray] = {}
        uniq = list(dict.fromkeys(ids))
        for i in range(0, len(uniq), 500):
            part = uniq[i:i + 500]
            for r in tbl.search().where(sql_in("id", part)).select(["id", "vector"]).limit(len(part)).to_list():
                out[r["id"]] = np.asarray(r["vector"], dtype=np.float32)
        return out

    def get_items(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        tbl = self._items()
        if tbl is None or not ids:
            return {}
        rows = tbl.search().where(sql_in("id", ids)).select(_ITEM_COLS).limit(len(ids)).to_list()
        return {r["id"]: r for r in rows}

    # ----------------------------------------------------------------------------------
    def _write_events(self, pairs: list[tuple[dict[str, Any], np.ndarray]]) -> None:
        if not pairs:
            return
        import pyarrow as pa

        cols: dict[str, list[Any]] = {k: [r[k] for r, _ in pairs] for k in _EVENT_COLS}
        cols["vector"] = [np.asarray(v, dtype=np.float32) for _, v in pairs]
        tbl = self._events(create=True)
        with write_lock(self.directory, EVENTS):
            tbl.add(pa.Table.from_pydict(cols, schema=_event_schema(self.dim)))

    def vote_of(self, user_id: str, item_id: str) -> str | None:
        """这个用户对这个条目的评价（最早那次赞 / 踩）；没评价过返回 None。"""
        tbl = self._events()
        if tbl is None or not item_id:
            return None
        rows = tbl.search().where(f"user_id = {sql_str(user_id)} AND item_id = {sql_str(item_id)} AND {sql_in('kind', VOTE_KINDS)}").select(["kind", "ts"]).limit(10_000).to_list()
        return min(rows, key=lambda r: r["ts"])["kind"] if rows else None

    def log_event(self, user_id: str, *, kind: str, item_id: str = "", kb_id: str = "", text: str = "", weight: float | None = None) -> dict[str, Any]:
        """记录一次交互。ask 用问题文本的向量；其余用条目向量（并累加条目热度）。
        赞 / 踩：同一用户对同一条目已经评价过就抛 AlreadyVoted——查重与写入在同一把锁里，同时到达的两次请求也只记第一次。"""
        if kind in VOTE_KINDS:
            with _VOTE_LOCK:
                prev = self.vote_of(user_id, item_id)
                if prev:
                    raise AlreadyVoted(item_id, prev)
                return self._log_event(user_id, kind=kind, item_id=item_id, kb_id=kb_id, text=text, weight=weight)
        return self._log_event(user_id, kind=kind, item_id=item_id, kb_id=kb_id, text=text, weight=weight)

    def _log_event(self, user_id: str, *, kind: str, item_id: str, kb_id: str, text: str, weight: float | None) -> dict[str, Any]:
        if kind not in EVENT_WEIGHTS:
            raise ValueError(f"未知交互类型：{kind}")
        if kind == "ask":
            if not text.strip():
                raise ValueError("ask 事件需要问题文本")
            vec = self.embedder.embed_queries([text])[0]
        else:
            vec = self._item_vectors([item_id]).get(item_id)
            if vec is None:
                raise KeyError(f"推荐条目不存在：{item_id}")
        w = EVENT_WEIGHTS[kind] if weight is None else float(weight)
        row = {"id": uuid.uuid4().hex, "user_id": user_id, "item_id": item_id if kind != "ask" else "", "kb_id": kb_id, "kind": kind, "weight": w, "text": text[:500], "ts": time.time()}
        self._write_events([(row, vec)])
        if kind != "ask" and w > 0:
            items = self._items()
            with write_lock(self.directory, ITEMS):
                items.update(where=f"id = {sql_str(item_id)}", values_sql={"popularity": f"popularity + {w}"})
        return row

    def dedupe_votes(self, *, dry_run: bool = False) -> dict[str, Any]:
        """清理"每人每条只能评价一次"上线前留下的重复评价：每个用户对每个条目只保留最早那次赞 / 踩，其余删除；
        再按交互记录重算热度（重复的点赞当时都给热度加过分）。dry_run 只算出要删的事件和热度变化，不写入——先导出备份再执行。
        返回 {"removed": [被删事件], "popularity": {条目: (旧, 新)}}。"""
        events = self._events()
        if events is None:
            return {"removed": [], "popularity": {}}
        with _VOTE_LOCK:
            rows = events.search().where(sql_in("kind", VOTE_KINDS)).select(_EVENT_COLS).limit(10_000_000).to_list()
            rows.sort(key=lambda r: r["ts"])
            first: set[tuple[str, str]] = set()
            dup: list[dict[str, Any]] = []
            for r in rows:
                key = (r["user_id"], r["item_id"])
                if key in first:
                    dup.append(r)
                else:
                    first.add(key)
            if dup and not dry_run:
                with write_lock(self.directory, EVENTS):
                    for i in range(0, len(dup), 500):
                        events.delete(sql_in("id", [r["id"] for r in dup[i:i + 500]]))
            changes = self.recompute_popularity(dry_run=dry_run, exclude_event_ids={r["id"] for r in dup} if dry_run else None)
        return {"removed": dup, "popularity": changes}

    def recompute_popularity(self, *, dry_run: bool = False, exclude_event_ids: set[str] | None = None) -> dict[str, tuple[float, float]]:
        """热度 = 所有用户在该条目上正向交互的权重之和（点赞 1、点开 0.6、被引用 0.5、浏览 0.3；点踩不加）。
        返回有变化的条目 {id: (旧热度, 新热度)}；dry_run 时不写入。"""
        items, events = self._items(), self._events()
        if items is None:
            return {}
        totals: dict[str, float] = {}
        if events is not None:
            for r in events.search().where("item_id != ''").select(["id", "item_id", "weight"]).limit(10_000_000).to_list():
                if r["weight"] > 0 and not (exclude_event_ids and r["id"] in exclude_event_ids):
                    totals[r["item_id"]] = totals.get(r["item_id"], 0.0) + float(r["weight"])
        current = {r["id"]: float(r["popularity"]) for r in items.search().select(["id", "popularity"]).limit(10_000_000).to_list()}
        changes = {i: (round(old, 4), round(totals.get(i, 0.0), 4)) for i, old in current.items() if abs(old - totals.get(i, 0.0)) > 1e-3}
        if changes and not dry_run:
            with write_lock(self.directory, ITEMS):
                for i, (_, new) in changes.items():
                    items.update(where=f"id = {sql_str(i)}", values={"popularity": new})
        return changes

    def events(self, user_id: str, limit: int = 300) -> list[dict[str, Any]]:
        tbl = self._events()
        if tbl is None:
            return []
        rows = tbl.search().where(f"user_id = {sql_str(user_id)}").select(_EVENT_COLS + ["vector"]).limit(100_000).to_list()
        rows.sort(key=lambda r: r["ts"])
        voted: set[str] = set()
        kept: list[dict[str, Any]] = []
        for r in rows:                                   # 同一条目只算最早那次评价（限制上线前连点留下的重复评价不计）
            if r["kind"] in VOTE_KINDS and r["item_id"]:
                if r["item_id"] in voted:
                    continue
                voted.add(r["item_id"])
            kept.append(r)
        kept.reverse()
        return kept[:limit]

    # ----------------------------------------------------------------------------------
    def profile_vector(self, events: list[dict[str, Any]], extra_interest: list[np.ndarray] | None = None) -> np.ndarray | None:
        now = time.time()
        acc = np.zeros(self.dim, dtype=np.float32)
        total = 0.0
        for e in events:
            decay = 0.5 ** (max(0.0, now - e["ts"]) / (self.cfg.half_life_days * _DAY))
            w = float(e["weight"]) * decay * (0.5 if e["weight"] < 0 else 1.0)
            acc += w * np.asarray(e["vector"], dtype=np.float32)
            total += max(w, 0.0)
        for v in extra_interest or []:          # 长期记忆里的偏好：固定权重，不衰减
            acc += 0.5 * np.asarray(v, dtype=np.float32)
            total += 0.5
        if total <= 0 or float(np.linalg.norm(acc)) < 1e-9:
            return None
        return _norm(acc)

    def recommend(self, user_id: str, *, kb_id: str | None = None, kinds: list[str] | None = None, k: int = 8, extra_interest: list[np.ndarray] | None = None) -> dict[str, Any]:
        items = self._items()
        if items is None or not items.count_rows():
            return {"mode": "empty", "items": [], "profile": None}
        evs = self.events(user_id)
        votes = {e["item_id"]: e["kind"] for e in evs if e["kind"] in VOTE_KINDS}
        seen = {e["item_id"] for e in evs if e["item_id"]} if self.cfg.exclude_seen else {e["item_id"] for e in evs if e["kind"] == "dislike"}
        where = []
        if kb_id:
            where.append(f"kb_id = {sql_str(kb_id)}")
        if kinds:
            where.append(sql_in("kind", kinds))
        if seen:
            where.append(f"NOT ({sql_in('id', seen)})")
        filt = " AND ".join(where) or "true"
        profile = self.profile_vector(evs, extra_interest)
        if profile is None:
            rows = items.search().where(filt).select(_ITEM_COLS).limit(100_000).to_list()
            rows.sort(key=lambda r: (r["popularity"], r["created_at"]), reverse=True)
            return {"mode": "popular", "items": [{**self._public(r), "score": round(float(r["popularity"]), 4), "reason": "热门内容（还没有你的交互记录）", "myVote": votes.get(r["id"])} for r in rows[:k]], "profile": None}
        rows = items.search(profile).distance_type("cosine").where(filt, prefilter=True).select(_ITEM_COLS + ["vector", "_distance"]).limit(max(k * 5, 30)).to_list()
        if not rows:
            return {"mode": "personal", "items": [], "profile": self._profile_summary(evs)}
        max_pop = max(float(r["popularity"]) for r in rows) or 1.0
        for r in rows:
            r["_sim"] = 1.0 - float(r["_distance"])
            r["_score"] = r["_sim"] + self.cfg.popularity_weight * float(r["popularity"]) / max_pop
        picked = self._mmr(rows, k)
        positives = [e for e in evs if e["weight"] > 0][:50]
        return {"mode": "personal", "items": [{**self._public(r), "score": round(r["_score"], 4), "similarity": round(r["_sim"], 4), "reason": self._reason(r, positives), "myVote": votes.get(r["id"])} for r in picked],
                "profile": self._profile_summary(evs)}

    def _mmr(self, rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        lam = self.cfg.mmr_lambda
        vecs = {r["id"]: np.asarray(r["vector"], dtype=np.float32) for r in rows}
        picked: list[dict[str, Any]] = []
        pool = sorted(rows, key=lambda r: r["_score"], reverse=True)
        while pool and len(picked) < k:
            def mmr(r):
                red = max((float(vecs[r["id"]] @ vecs[p["id"]]) for p in picked), default=0.0)
                return lam * r["_score"] - (1 - lam) * red
            best = max(pool, key=mmr)
            picked.append(best)
            pool.remove(best)
        return picked

    def _reason(self, row: dict[str, Any], positives: list[dict[str, Any]]) -> str:
        if not positives:
            return "与你的兴趣相近"
        v = np.asarray(row["vector"], dtype=np.float32)
        best = max(positives, key=lambda e: float(np.asarray(e["vector"], dtype=np.float32) @ v))
        if best["kind"] == "ask":
            return f"因为你问过「{best['text'][:24]}」"
        title = self.get_items([best["item_id"]]).get(best["item_id"], {}).get("title", "")
        verb = {"like": "喜欢", "click": "看过", "cite": "引用过", "view": "浏览过"}.get(best["kind"], "关注过")
        return f"与你{verb}的「{title[:24]}」相似" if title else "与你的兴趣相近"

    def _profile_summary(self, evs: list[dict[str, Any]]) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for e in evs:
            counts[e["kind"]] = counts.get(e["kind"], 0) + 1
        recent = [{"kind": e["kind"], "text": e["text"], "itemId": e["item_id"], "ts": e["ts"]} for e in evs[:10]]
        return {"events": len(evs), "byKind": counts, "recent": recent}

    @staticmethod
    def _public(r: dict[str, Any]) -> dict[str, Any]:
        return {"id": r["id"], "kbId": r["kb_id"], "kind": r["kind"], "refId": r["ref_id"], "title": r["title"], "text": r["text"][:200], "tags": list(r.get("tags") or []), "popularity": round(float(r["popularity"]), 3)}

    def similar(self, item_id: str, k: int = 6) -> list[dict[str, Any]]:
        vec = self._item_vectors([item_id]).get(item_id)
        if vec is None:
            raise KeyError(item_id)
        rows = self._items().search(vec).distance_type("cosine").where(f"id != {sql_str(item_id)}", prefilter=True).select(_ITEM_COLS + ["_distance"]).limit(k).to_list()
        return [{**self._public(r), "similarity": round(1.0 - float(r["_distance"]), 4)} for r in rows]

    def stats(self) -> dict[str, Any]:
        items, events = self._items(), self._events()
        by_kind: dict[str, int] = {}
        if items is not None:
            for r in items.search().select(["kind"]).limit(10_000_000).to_list():
                by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        users = set()
        if events is not None:
            users = {r["user_id"] for r in events.search().select(["user_id"]).limit(10_000_000).to_list()}
        return {"items": sum(by_kind.values()), "itemsByKind": by_kind, "events": events.count_rows() if events is not None else 0, "users": len(users),
                "tables": [table_info(t) for t in (items, events) if t is not None], **self._sig()}


def item_dict(it: RecItem) -> dict[str, Any]:
    return asdict(it)
