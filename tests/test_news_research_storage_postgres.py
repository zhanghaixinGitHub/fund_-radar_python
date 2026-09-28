"""本机随机隔离 schema 内验证真实 PostgreSQL 事务，测试数据不进入 public。"""

import importlib.util
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.core.config import get_settings
from app.schemas.news_research import NewsResearchBundle
from app.services import news_research_storage as storage
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError
from tests.test_news_research_storage import payload

pytestmark = pytest.mark.skipif(os.getenv("RUN_NEWS_RESEARCH_PG_TESTS") != "1", reason="显式启用本机隔离 PG")


@pytest.fixture
def database(monkeypatch):
    url = make_url(get_settings().ai_database_url)
    assert url.host in {"localhost", "127.0.0.1", "::1"} and url.database == "fund_ai"
    schema = "news_test_" + uuid4().hex
    control = create_engine(url, hide_parameters=True, connect_args={"connect_timeout": 5})
    engine = None
    try:
        with control.begin() as c:
            c.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
            c.exec_driver_sql(f'SET LOCAL search_path TO "{schema}", pg_catalog')
            file = Path(__file__).resolve().parents[1] / "alembic/versions/20260926_30_news_research_storage.py"
            spec = importlib.util.spec_from_file_location("news_migration_test", file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.op = Operations(MigrationContext.configure(c))
            module.upgrade()
        engine = create_engine(
            url,
            hide_parameters=True,
            pool_size=3,
            max_overflow=0,
            connect_args={
                "connect_timeout": 5,
                "options": f"-c search_path={schema},pg_catalog -c statement_timeout=5000 -c lock_timeout=3000",
            },
        )
        monkeypatch.setattr(storage, "get_nav_sample_storage_engine", lambda: engine)
        yield engine
    finally:
        if engine:
            engine.dispose()
        assert re.fullmatch(r"news_test_[0-9a-f]{32}", schema)
        with control.begin() as c:
            owner = c.execute(
                text("SELECT nspowner=current_user::regrole FROM pg_namespace WHERE nspname=:s"), {"s": schema}
            ).scalar_one_or_none()
            if owner:
                c.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
        control.dispose()


def counts(engine):
    with engine.connect() as c:
        return tuple(
            c.exec_driver_sql("SELECT count(*) FROM " + t).scalar_one()
            for t in ("news_research_bundle", "news_research_card")
        )


def test_durable_exact_retry_and_explicit_revision(database):
    b = NewsResearchBundle.model_validate(payload())
    first = storage.save_research_bundle(b)
    retry = storage.save_research_bundle(b)
    assert first["created"] and not retry["created"] and counts(database) == (1, 1)
    assert storage.read_research_bundle(first["bundle_hash"]) == b.model_dump(mode="json")
    changed = payload()
    changed["records"][0]["card"]["unknowns"].append("另一个未知项")
    with pytest.raises(ValueError, match="REVISION_CONTENT_CONFLICT"):
        storage.save_research_bundle(NewsResearchBundle.model_validate(changed))
    changed["revision"] = "v2"
    with pytest.raises(ValueError, match="NEEDS_PARENT"):
        storage.save_research_bundle(NewsResearchBundle.model_validate(changed))
    changed["supersedes_bundle_hash"] = first["bundle_hash"]
    second = storage.save_research_bundle(NewsResearchBundle.model_validate(changed))
    assert second["created"] and counts(database) == (2, 2)
    assert storage.read_research_bundle(first["bundle_hash"]) == b.model_dump(mode="json")
    with database.connect() as c:
        assert (
            c.exec_driver_sql(
                "SELECT count(*) FROM news_research_card WHERE business_eligible OR prediction_eligible "
                "OR training_eligible OR confidence IS NOT NULL"
            ).scalar_one()
            == 0
        )


def test_failure_rolls_back_whole_batch(database, monkeypatch):
    monkeypatch.setattr(storage.repository, "read_bundle", lambda *args: {"wrong": "readback"})
    with pytest.raises(ValueError, match="ROUNDTRIP"):
        storage.save_research_bundle(NewsResearchBundle.model_validate(payload()))
    assert counts(database) == (0, 0)


def test_concurrent_retries_create_one_version(database):
    b = NewsResearchBundle.model_validate(payload())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: storage.save_research_bundle(b), range(2)))
    assert sum(r["created"] for r in results) == 1 and counts(database) == (1, 1)


def test_database_rejects_history_rewrite(database):
    storage.save_research_bundle(NewsResearchBundle.model_validate(payload()))
    for sql in ("UPDATE news_research_card SET confidence=0", "DELETE FROM news_research_bundle"):
        with pytest.raises(DBAPIError), database.begin() as c:
            c.exec_driver_sql(sql)
    assert counts(database) == (1, 1)
