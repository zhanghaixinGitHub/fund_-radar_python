"""SQLAlchemy engine factory; no connection is opened during M0 startup."""

from functools import lru_cache

from sqlalchemy import Engine, create_engine

from app.core.config import get_settings


@lru_cache
def get_engine() -> Engine:
    """Create a pooled engine on demand for M1 migrations and repositories."""
    return create_engine(get_settings().ai_database_url, pool_pre_ping=True)
