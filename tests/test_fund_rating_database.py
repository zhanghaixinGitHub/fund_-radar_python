"""真实 PostgreSQL、随机隔离 schema；评级算例只存在于测试 schema，退出后删除。"""

import os
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.core.config import get_settings
from app.models.fund_rating import RatingBatch, RatingCurrent, RatingMethodology, RatingResult
from app.repositories.fund_rating import RatingConflict, lock_category, publish, withdraw
from app.services.fund_rating import _category, read
from app.services.fund_rating_inputs import digest, empty_evidence
from app.services.fund_rating_rules import candidate
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session


@pytest.fixture
def db():
    if os.getenv("RATING_POSTGRES_TEST") != "1":
        pytest.skip("显式 RATING_POSTGRES_TEST=1 才创建本地隔离 schema")
    url = get_settings().ai_database_url
    admin = create_engine(url)
    schema = "rating_test_" + uuid4().hex
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, connect_args={"options": f"-c search_path={schema},public"})
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("rating_migration", "alembic/versions/20261010_35_fund_rating.py")
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        with engine.begin() as conn:
            with Operations.context(MigrationContext.configure(conn)):
                migration.upgrade()
            conn.execute(text("CREATE TABLE fund_share_class(fund_code varchar(6) PRIMARY KEY)"))
            conn.execute(text("INSERT INTO fund_share_class VALUES ('002112'),('001412')"))
        yield engine
    finally:
        engine.dispose()
        # 仅删除本 fixture 生成并严格校验的随机 schema，不触及 public。
        assert schema.startswith("rating_test_") and len(schema) == 44
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def members():
    config = candidate("ACTIVE_EQUITY")
    result = []
    for i in range(30):
        code = "002112" if i == 0 else "001412" if i == 1 else f"{800000 + i}"
        metrics = {m: str(i + 1) for terms in config["formula"].values() for m, _, _ in terms}
        metrics["rolling"] = [str(i / 100)] * 25
        result.append(
            {
                "fund_code": code,
                "product_id": str(i),
                "family": "ACTIVE_EQUITY",
                "representative": True,
                "admitted": True,
                "reasons": [],
                "metrics": metrics,
                "valid_until": datetime.now(UTC) + timedelta(days=2),
                "evidence_hash": None,
                "public": empty_evidence({}, "ACTIVE_EQUITY", date(2026, 9, 30)),
                "facts": {"nav": "2026-09-30", "holdings": "2026-06-30", "manager": "2026-10-10", "fees": "2026-10-10"},
                "bundle": {"sources": [], "category_label": "隔离测试类别", "history": {"dates": ["2023-09-30"]}},
            }
        )
    return result


def activate_fixture(db):
    config = candidate("ACTIVE_EQUITY")
    with Session(db) as session, session.begin():
        session.add(
            RatingMethodology(
                methodology_id=digest(config),
                family="ACTIVE_EQUITY",
                config=config,
                active=True,
                validation={
                    "fixture": "isolated only",
                    "report": {
                        "categories": [
                            {"category": name, "eligible": True} for name in ("test-equity", "test", "crash")
                        ]
                    },
                },
                created_at=datetime.now(UTC),
            )
        )


def test_real_migration_atomic_publish_read_and_idempotency(db, tmp_path):
    activate_fixture(db)
    data = members()
    a = _category(db, tmp_path, "test-equity", data, date(2026, 9, 30), datetime.now(UTC))
    b = _category(db, tmp_path, "test-equity", data, date(2026, 9, 30), datetime.now(UTC))
    assert a["batch"] == b["batch"] and a["rated"] == 30
    batch = read(["002112", "999999", "002112"], engine=db)
    assert len(batch["items"]) == 2 and batch["items"][1]["status"] == "NOT_FOUND"
    detail = read(["002112"], detail=True, rating_ref=a["batch"], engine=db)
    assert detail["grade"] == batch["items"][0]["grade"]
    assert len(detail["dimensions"]) == 8
    assert not any(key in str(detail) for key in ("dimensionScore", "raw_score", "methodology", "weights"))
    with pytest.raises(LookupError):
        read(["999999"], detail=True, engine=db)


def test_partial_publish_failure_rolls_back_all_rows(db, tmp_path, monkeypatch):
    activate_fixture(db)
    import app.services.fund_rating as service

    monkeypatch.setattr(service, "publish", lambda *_: (_ for _ in ()).throw(RuntimeError("injected crash")))
    with pytest.raises(RuntimeError):
        _category(db, tmp_path, "crash", members(), date(2026, 9, 30), datetime.now(UTC))
    with Session(db) as session:
        assert not session.scalars(select(RatingBatch)).all()
        assert not session.scalars(select(RatingResult)).all()
        assert not session.scalars(select(RatingCurrent)).all()


def test_withdraw_expiry_reference_and_older_task(db, tmp_path):
    activate_fixture(db)
    data = members()
    old = _category(db, tmp_path, "test", data, date(2026, 8, 31), datetime.now(UTC))
    new = _category(db, tmp_path, "test", data, date(2026, 9, 30), datetime.now(UTC))
    with Session(db) as session, session.begin():
        with pytest.raises(RatingConflict, match="OLDER"):
            publish(session, session.get(RatingBatch, old["batch"]))
    with Session(db) as session, session.begin():
        publish(session, session.get(RatingBatch, old["batch"]), rollback_reason="隔离测试回退")
    assert read(["002112"], engine=db)["items"][0]["ratingRef"] == old["batch"]
    with Session(db) as session, session.begin():
        withdraw(session, old["batch"], "隔离测试撤回")
    assert read(["002112"], engine=db)["items"][0]["status"] == "WITHDRAWN"
    assert read(["002112"], detail=True, rating_ref=old["batch"], engine=db)["grade"] is None
    assert read(["002112"], detail=True, rating_ref=new["batch"], engine=db)["grade"] is not None


def test_database_lock_and_constraints(db, tmp_path):
    with Session(db) as a, a.begin(), Session(db) as b, b.begin():
        lock_category(a, "same")
        with pytest.raises(RatingConflict):
            lock_category(b, "same")
    activate_fixture(db)
    result = _category(db, tmp_path, "test", members(), date(2026, 9, 30), datetime.now(UTC))
    from sqlalchemy.exc import DBAPIError

    with Session(db) as session, pytest.raises(DBAPIError), session.begin():
        session.execute(text("UPDATE fund_rating_result SET score=0 WHERE batch_id=:id"), {"id": result["batch"]})


def test_candidate_rule_cannot_publish_a_grade(db, tmp_path):
    result = _category(db, tmp_path, "candidate", members(), date(2026, 9, 30), datetime.now(UTC))
    assert result["rated"] == 0
    value = read(["002112"], detail=True, engine=db)
    assert value["grade"] is None
    assert all(d["grade"] is None for d in value["dimensions"])
    assert value["dimensions"][0]["metrics"]
    assert "原件已核验" in value["summary"]


def test_input_corruption_is_service_failure_not_not_rated(db, tmp_path):
    activate_fixture(db)
    result = _category(db, tmp_path, "test", members(), date(2026, 9, 30), datetime.now(UTC))
    with Session(db) as session, session.begin():
        batch = session.get(RatingBatch, result["batch"])
        # 仅在隔离 schema 注入物理损坏，正常服务无法修改不可变快照。
        session.execute(text("ALTER TABLE fund_rating_batch DISABLE TRIGGER tg_rating_batch_guard"))
        corrupt = deepcopy(batch.input_snapshot)
        corrupt["as_of"] = "2000-01-01"
        batch.input_snapshot = corrupt
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        read(["002112"], engine=db)


@pytest.mark.parametrize("revoke", ["family", "category"])
def test_rule_revocation_invalidates_whole_mixed_batch(db, tmp_path, revoke):
    """首行缺资料不能使同批已评级行绕过规则撤回，类别资格也不能沿用过期审核。"""
    activate_fixture(db)
    data = members()
    # 保持30个可比产品，另加一个代码更靠前的未评级行。
    missing = deepcopy(data[0])
    missing.update(
        fund_code="000001", admitted=False, representative=False, reasons=["REQUIRED_INPUT_MISSING"], metrics={}
    )
    data.insert(0, missing)
    with db.begin() as conn:
        conn.execute(text("INSERT INTO fund_share_class VALUES ('000001')"))
    result = _category(db, tmp_path, "test", data, date(2026, 9, 30), datetime.now(UTC))
    assert result["rated"] == 30
    with Session(db) as session, session.begin():
        method = session.get(RatingMethodology, digest(candidate("ACTIVE_EQUITY")))
        if revoke == "family":
            method.active = False
        else:
            method.validation = {"report": {"categories": []}}
    values = read(["000001", "002112"], engine=db)["items"]
    assert all(v["status"] == "WITHDRAWN" and v["grade"] is None for v in values)
