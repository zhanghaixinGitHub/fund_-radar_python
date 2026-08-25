"""SQLAlchemy 引擎工厂；M0 启动阶段不会主动建立数据库连接。"""

from functools import lru_cache

from sqlalchemy import Engine, create_engine

from app.core.config import get_settings


@lru_cache
def get_engine() -> Engine:
    """按需创建并缓存连接池引擎，供 M1 迁移和仓储层使用。"""
    return create_engine(get_settings().ai_database_url, pool_pre_ping=True)
