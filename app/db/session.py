from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..config import app_config
from .models import Base

engine = create_engine(app_config.sqlite_url, connect_args={"check_same_thread": False})
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
