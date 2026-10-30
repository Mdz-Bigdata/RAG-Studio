"""多模态记忆：记忆在图文向量空间（WeMM 等跨模态模型）里的第二份向量 + 附带的图片 / 短视频。

    LanceDB 表 memories_mm   id / user_id / agent_id / media（相对路径）/ media_type / vector
    <lancedb>/memory_media/<id>.<ext>   附带的媒体文件

- 每条记忆都有一行：纯文字记忆存文字向量，带图的存"图 + 文字"联合向量，带视频的存"视频 + 文字"联合向量。
  所以文字能召回图片 / 视频记忆，上传一张图也能召回相关的文字记忆；
- 与文本向量那一路（memories 表）的关系：召回时两路各取候选，相似度取较大者，再走相关度 × 新近度 × 重要度的综合排序；
- 图文向量模型换了：用保存的文字与媒体文件整表重算。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..lance import connect, open_table, sql_in, sql_str, write_lock

logger = logging.getLogger(__name__)

TABLE = "memories_mm"
MEDIA_DIR = "memory_media"
MEDIA_TYPES = {"image": ("jpg", "jpeg", "png", "webp", "gif", "bmp"), "video": ("mp4", "mov", "webm", "m4v", "mkv")}


def _schema(dim: int):
    import pyarrow as pa

    return pa.schema([pa.field("id", pa.string()), pa.field("user_id", pa.string()), pa.field("agent_id", pa.string()), pa.field("media", pa.string()),
                      pa.field("media_type", pa.string()), pa.field("vector", pa.list_(pa.float32(), dim))])


class MemoryMediaIndex:
    def __init__(self, directory: str | Path, embedder: Any, *, video_frames: int = 16) -> None:
        self.directory = Path(directory)
        self.embedder = embedder
        self.video_frames = video_frames
        self._meta_path = self.directory / f"{TABLE}.meta.json"
        self._tbl = None

    @property
    def db(self):
        return connect(self.directory)

    @property
    def media_root(self) -> Path:
        return self.directory / MEDIA_DIR

    def _signature(self) -> dict[str, Any]:
        sig = self.embedder.signature()
        return {"embedder": sig["embedder"], "model": sig["model"]}

    def _table(self, dim: int | None = None):
        if self._tbl is None:
            self._tbl = open_table(self.db, TABLE)
            if self._tbl is None and dim:
                self._tbl = self.db.create_table(TABLE, schema=_schema(dim), exist_ok=True)
                self._meta_path.write_text(json.dumps(self._signature(), ensure_ascii=False), encoding="utf-8")
        return self._tbl

    # --- 向量 -------------------------------------------------------------------------------
    def _vector(self, content: str, media: str, media_type: str) -> np.ndarray:
        emb = self.embedder
        path = self.media_root / media if media else None
        if path is not None and path.is_file() and media_type == "image":
            from PIL import Image

            with Image.open(path) as im:
                image = im.convert("RGB")
            if hasattr(emb, "embed_image_texts") and content.strip():
                return emb.embed_image_texts([(image, content)])[0]
            return emb.embed_images([image])[0]
        if path is not None and path.is_file() and media_type == "video" and getattr(emb, "video_native", False):
            from ..multimodal.video import iter_frames, probe
            from ..multimodal.wemm import VideoClip

            info = probe(path)
            fps = max(0.2, min(2.0, self.video_frames / max(info.duration, 1.0)))
            frames = list(iter_frames(path, fps=fps, max_side=512, max_seconds=600, info=info))[: self.video_frames * 2]
            if frames:
                _, joint = emb.embed_videos([VideoClip(frames=[f for _, f in frames], times=[t for t, _ in frames], text=content)])
                return joint[0]
        return emb.embed_texts([content or " "])[0]

    def _rows(self, recs: list[Any], media: dict[str, tuple[str, str]]):
        import pyarrow as pa

        vecs = [self._vector(r.content, *media.get(r.id, ("", ""))) for r in recs]
        dim = len(vecs[0])
        return pa.Table.from_pydict({"id": [r.id for r in recs], "user_id": [r.user_id for r in recs], "agent_id": [r.agent_id for r in recs],
                                     "media": [media.get(r.id, ("", ""))[0] for r in recs], "media_type": [media.get(r.id, ("", ""))[1] for r in recs],
                                     "vector": [np.asarray(v, dtype=np.float32) for v in vecs]}, schema=_schema(dim))

    def check(self, all_records: list[Any]) -> None:
        """图文向量模型换了 / 有记忆还没有多模态向量：重算（记忆量小，整表重写）。"""
        tbl = self._table()
        built = json.loads(self._meta_path.read_text(encoding="utf-8")) if self._meta_path.exists() else {}
        have = {r["id"] for r in tbl.search().select(["id"]).limit(1_000_000).to_list()} if tbl is not None else set()
        if tbl is not None and built == self._signature() and {r.id for r in all_records} <= have:
            return
        if not all_records:
            return
        media = self.media_of([r.id for r in all_records])
        logger.info("重算 %d 条记忆的多模态向量（%s）", len(all_records), self._signature()["model"])
        rows = self._rows(all_records, media)
        with write_lock(self.directory, TABLE):
            self._tbl = self.db.create_table(TABLE, rows, schema=rows.schema, mode="overwrite")
        self._meta_path.write_text(json.dumps(self._signature(), ensure_ascii=False), encoding="utf-8")

    # --- 写入 -------------------------------------------------------------------------------
    def upsert(self, rec: Any, *, media: tuple[bytes, str] | None = None) -> None:
        """media = (文件字节, 扩展名)；不给时沿用这条记忆已有的媒体。"""
        old = self.media_of([rec.id]).get(rec.id, ("", ""))
        if media is not None:
            data, ext = media
            ext = ext.lower().lstrip(".")
            kind = next((k for k, exts in MEDIA_TYPES.items() if ext in exts), None)
            if kind is None:
                raise ValueError(f"不支持的媒体格式：{ext}（图片 {'/'.join(MEDIA_TYPES['image'])}，视频 {'/'.join(MEDIA_TYPES['video'])}）")
            self.media_root.mkdir(parents=True, exist_ok=True)
            if old[0] and old[0] != f"{rec.id}.{ext}":
                (self.media_root / old[0]).unlink(missing_ok=True)
            (self.media_root / f"{rec.id}.{ext}").write_bytes(data)
            old = (f"{rec.id}.{ext}", kind)
        rows = self._rows([rec], {rec.id: old})
        tbl = self._table(dim=rows.schema.field("vector").type.list_size)
        with write_lock(self.directory, TABLE):
            tbl.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(rows)

    def remove(self, ids: list[str]) -> None:
        tbl = self._table()
        if tbl is None or not ids:
            return
        for media, _ in self.media_of(ids).values():
            if media:
                (self.media_root / media).unlink(missing_ok=True)
        with write_lock(self.directory, TABLE):
            tbl.delete(sql_in("id", ids))

    def prune(self, keep_ids: set[str]) -> int:
        """删掉主表里已经不存在的记忆（遗忘 / 清空之后）对应的行与媒体文件。"""
        tbl = self._table()
        if tbl is None:
            return 0
        gone = [r["id"] for r in tbl.search().select(["id"]).limit(1_000_000).to_list() if r["id"] not in keep_ids]
        self.remove(gone)
        return len(gone)

    # --- 读取 -------------------------------------------------------------------------------
    def media_of(self, ids: list[str]) -> dict[str, tuple[str, str]]:
        tbl = self._table()
        if tbl is None or not ids:
            return {}
        rows = tbl.search().where(sql_in("id", ids)).select(["id", "media", "media_type"]).limit(len(ids)).to_list()
        return {r["id"]: (r["media"], r["media_type"]) for r in rows if r["media"]}

    def media_path(self, memory_id: str) -> Path | None:
        media = self.media_of([memory_id]).get(memory_id)
        if not media:
            return None
        p = (self.media_root / media[0]).resolve()
        return p if p.is_file() and p.parent == self.media_root.resolve() else None

    def search(self, *, query: str = "", image: Any = None, user_id: str, agent_id: str | None, limit: int) -> dict[str, float]:
        """文字和 / 或图片查询 → {记忆 id: 余弦}（两种查询都给时取较大者）。"""
        tbl = self._table()
        if tbl is None or not tbl.count_rows():
            return {}
        where = f"user_id = {sql_str(user_id)}" + (f" AND agent_id = {sql_str(agent_id)}" if agent_id else "")
        vecs = []
        if query.strip():
            vecs.append(self.embedder.embed_query(query, "memory"))
        if image is not None:
            vecs.append(self.embedder.embed_images([image])[0])
        out: dict[str, float] = {}
        for v in vecs:
            for r in tbl.search(np.asarray(v, dtype=np.float32)).distance_type("cosine").where(where, prefilter=True).select(["id", "_distance"]).limit(limit).to_list():
                out[r["id"]] = max(out.get(r["id"], -1.0), 1.0 - float(r["_distance"]))
        return out

    def count(self) -> dict[str, int]:
        tbl = self._table()
        if tbl is None:
            return {"rows": 0, "media": 0}
        rows = tbl.search().select(["media"]).limit(1_000_000).to_list()
        return {"rows": len(rows), "media": sum(1 for r in rows if r["media"])}
