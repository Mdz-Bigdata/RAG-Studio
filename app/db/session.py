from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from ..config import app_config
from .models import Base

# 不用连接池：一个请求里常常嵌套开第二个会话（接口的 get_db + 服务内部的 SessionLocal）。默认池上限 15，
# 并发一高（多个页面同时轮询、启动时模型还在加载），所有请求都攥着一个连接等第二个，整体卡死 30 秒后报 500。
# SQLite 开连接很便宜，按需开、用完即关就不存在上限。
engine = create_engine(app_config.sqlite_url, connect_args={"check_same_thread": False}, poolclass=NullPool)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# 旧库补列：create_all 不会给已存在的表加列，这里按需 ALTER（SQLite 只支持 ADD COLUMN）
_ADDED_COLUMNS = {
    "documents": {"department": "VARCHAR(100)", "access_level": "INTEGER DEFAULT 0", "allowed_departments": "VARCHAR(500)", "auto_meta": "TEXT"},
}


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in _ADDED_COLUMNS.items():
            if not insp.has_table(table):
                continue
            existing = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in cols.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
