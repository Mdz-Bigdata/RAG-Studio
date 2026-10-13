"""数据库会话不能因为并发嵌套而互相等死。"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.pool import NullPool

from app.db.session import SessionLocal, engine


def test_many_concurrent_sessions_do_not_wait_on_a_pool_limit():
    # 接口拿着一个会话再进服务层开第二个，是常见路径；带上限的连接池在 16 个并发会话时就会卡住 30 秒
    assert isinstance(engine.pool, NullPool)
    sessions = [SessionLocal() for _ in range(40)]
    try:
        assert [s.execute(text("select 1")).scalar() for s in sessions] == [1] * 40
    finally:
        for s in sessions:
            s.close()
