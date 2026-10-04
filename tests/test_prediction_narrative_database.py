"""可选本地 PostgreSQL 验证：临时 schema 与全部数据最终回滚，不动现有预测。"""

import os
import runpy
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.db.session import get_engine
from app.repositories import prediction_narrative as store
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

pytestmark = pytest.mark.skipif(os.getenv("NARRATIVE_DB_TESTS") != "1", reason="显式开启本地数据库事务测试")


@pytest.fixture
def database(monkeypatch):
    schema = "narrative_test_" + uuid4().hex
    with get_engine().connect() as c:
        transaction = c.begin()
        try:
            c.execute(text(f'CREATE SCHEMA "{schema}"'))
            c.execute(text(f'SET LOCAL search_path TO "{schema}"'))
            migration = runpy.run_path(
                Path(__file__).parents[1] / "alembic/versions/20260930_32_prediction_narrative.py"
            )
            with Operations.context(MigrationContext.configure(c)):
                migration["upgrade"]()
                runpy.run_path(Path(__file__).parents[1] / "alembic/versions/20260930_33_narrative_revision.py")[
                    "upgrade"
                ]()

            class TransactionEngine:
                @contextmanager
                def connect(self):
                    yield c

                @contextmanager
                def begin(self):
                    with c.begin_nested():
                        yield c

            monkeypatch.setattr(store, "get_engine", lambda: TransactionEngine())
            yield c
        finally:
            transaction.rollback()


def claim(identity, version="v1"):
    return store.claim("daily", identity, "002112", "a" * 64, "b" * 64, version, "test-model")


def test_exclusive_claim_global_limit_and_wrong_owner(database):
    first, second, third = uuid4(), uuid4(), uuid4()
    owner = claim(first)
    assert owner
    assert claim(first) is None
    assert claim(second)
    assert claim(third) is None
    assert not store.finish("daily", first, uuid4(), payload={"summary": "错误持有人"})
    assert store.finish("daily", first, owner, payload={"summary": "成功保存"})
    assert claim(third)


def test_ready_snapshot_is_immutable_even_after_style_change(database):
    identity = uuid4()
    owner = claim(identity)
    assert store.finish("daily", identity, owner, payload={"summary": "原解释"})
    next_owner = claim(identity, "v2")
    assert next_owner
    assert store.finish("daily", identity, next_owner, payload={"summary": "补充原因"})
    assert claim(identity, "v2") is None
    assert store.read("daily", identity, "v1")["payload"]["summary"] == "原解释"
    assert store.read("daily", identity, "v2")["payload"]["summary"] == "补充原因"
    with pytest.raises(DBAPIError), database.begin_nested():
        database.execute(
            text("UPDATE prediction_narrative SET payload='{}'::jsonb WHERE source_id=:id"), {"id": identity}
        )


def test_failure_cooldown_and_three_attempt_budget_per_revision(database):
    identity = uuid4()
    owner = claim(identity)
    assert store.finish("daily", identity, owner, failure="INVALID_OUTPUT")
    assert claim(identity) is None
    database.execute(
        text("UPDATE prediction_narrative SET retry_after=clock_timestamp()-interval '1 second' WHERE source_id=:id"),
        {"id": identity},
    )
    owner = claim(identity)
    assert owner
    assert store.finish("daily", identity, owner, failure="INVALID_OUTPUT")
    database.execute(
        text("UPDATE prediction_narrative SET retry_after=clock_timestamp()-interval '1 second' WHERE source_id=:id"),
        {"id": identity},
    )
    owner = claim(identity)
    assert owner
    assert store.finish("daily", identity, owner, failure="INVALID_OUTPUT")
    assert claim(identity) is None
    assert store.read("daily", identity, "v1")["attempts"] == 3


def test_expired_lease_recovers_without_accepting_late_response(database):
    identity = uuid4()
    owner = claim(identity)
    database.execute(
        text("UPDATE prediction_narrative SET lease_until=clock_timestamp()-interval '1 second' WHERE source_id=:id"),
        {"id": identity},
    )
    next_owner = claim(identity)
    assert next_owner and next_owner != owner
    assert not store.finish("daily", identity, owner, payload={"summary": "迟到结果"})
    assert store.finish("daily", identity, next_owner, payload={"summary": "有效结果"})


def test_previous_revision_fallback_and_cross_revision_identity_check(database):
    identity = uuid4()
    owner = claim(identity, "PREDICTION_NARRATIVE_ZH_V2")
    assert store.finish("daily", identity, owner, payload={"summary": "旧版"})
    assert store.previous("daily", identity, "PREDICTION_NARRATIVE_ZH_V3")["payload"]["summary"] == "旧版"
    assert (
        store.claim("daily", identity, "002112", "x" * 64, "b" * 64, "PREDICTION_NARRATIVE_ZH_V3", "test-model") is None
    )
    assert claim(identity, "PREDICTION_NARRATIVE_ZH_V3")
