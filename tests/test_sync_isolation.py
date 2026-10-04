"""同步隔离故障注入；SQLite 临时库验证事务，所有外部接口均替身，禁止训练与真实写入。"""

from contextlib import nullcontext
from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.repositories.feature_snapshot import FeatureSnapshotWriteStats
from app.repositories.fund_sync import WriteStats
from app.services import multi_prediction_sync as multi
from app.services import stock_feature_snapshot as features
from app.services import tushare_fund_sync as funds
from app.services.sync_jobs import LocalSyncJobManager
from sqlalchemy import create_engine, text
from tests.test_multi_prediction_sync import setup_worker
from tests.test_stock_feature_snapshot import _input_with_points
from tests.test_sync_all_jobs import wait_finished


@pytest.fixture
def isolated_db(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "isolation.db"))
    with engine.begin() as c:
        c.execute(text("CREATE TABLE saved (code TEXT PRIMARY KEY, value TEXT)"))
    yield engine
    engine.dispose()


def test_fund_transaction_rollback_preserves_other_funds_and_retry_is_idempotent(isolated_db):
    worker = object.__new__(funds.TushareFundSyncService)
    worker._engine, worker._batch_size = isolated_db, 1
    worker._start_run = lambda **kw: (uuid4(), uuid4())
    worker._complete_run = lambda *args: None
    fail = {"B"}

    def write(session, *, source_id, records):
        code, sequence = records[0]
        session.execute(
            text("INSERT INTO saved(code,value) VALUES(:c,:v) ON CONFLICT DO NOTHING"), {"c": code, "v": str(sequence)}
        )
        if code in fail and sequence == 2:
            raise RuntimeError("injected second chunk failure")
        return WriteStats(created_count=1)

    def run():
        return worker._sync_detail_funds(
            ("A", "B", "C"),
            sync_type="TEST",
            fetch=lambda c: [(c, 1), (c, 2)],
            normalize=lambda values, code: (values, 0),
            write=write,
        )

    result = run()
    assert result.status == "PARTIAL_SUCCESS" and len(result.issues) == 1
    with isolated_db.connect() as c:
        assert c.execute(text("SELECT code FROM saved ORDER BY code")).scalars().all() == ["A", "C"]
    fail.clear()
    assert run().status == "SUCCEEDED"
    assert run().status == "SUCCEEDED"
    with isolated_db.connect() as c:
        assert c.execute(text("SELECT count(*) FROM saved")).scalar() == 3


def test_feature_calculation_and_commit_failures_are_isolated(monkeypatch, isolated_db):
    source = SimpleNamespace(source_code="TEST", source_sync_run_id=uuid4(), source_sync_finished_at=None)
    inputs = [replace(_input_with_points(252), fund_code=c) for c in ("A", "B", "C", "D")]
    monkeypatch.setattr(features, "get_enabled_feature_source", lambda *args: source)
    monkeypatch.setattr(features, "list_stock_feature_inputs", lambda *args, **kw: inputs)
    original = features.build_stock_feature_snapshot

    def build(item):
        if item.fund_code == "B":
            raise ValueError("injected calculate failure")
        return original(item)

    def save(session, *, records):
        code = records[0].fund_code
        session.execute(text("INSERT INTO saved(code,value) VALUES(:c, :v)"), {"c": code, "v": "1"})
        if code == "C":
            raise ValueError("injected save failure")
        return FeatureSnapshotWriteStats(created_count=1)

    monkeypatch.setattr(features, "build_stock_feature_snapshot", build)
    monkeypatch.setattr(features, "upsert_feature_snapshots", save)
    result = features.StockFeatureSnapshotService()._build_with_engine(isolated_db)
    assert result.status == "PARTIAL_SUCCESS" and len(result.issues) == 2 and result.created_count == 2
    with isolated_db.connect() as c:
        assert c.execute(text("SELECT code FROM saved ORDER BY code")).scalars().all() == ["A", "D"]


@pytest.mark.parametrize("fault", ["create", "timeout", "archive", "verify"])
def test_multi_batches_continue_and_do_not_count_followup_as_fund_failure(monkeypatch, fault):
    worker = setup_worker(monkeypatch, count=201)
    original = multi.create_task
    calls = []

    def create(batch):
        calls.append(batch)
        if len(calls) == 1 and fault == "create":
            raise RuntimeError("create failed")
        task = original(batch)
        if fault == "timeout" and len(calls) == 1:
            task["items"] = [{**i, "status": "PENDING"} for i in task["items"]]
            monkeypatch.setattr(multi, "WAIT_SECONDS", 0)
        else:
            monkeypatch.setattr(multi, "WAIT_SECONDS", 600)
        return task

    monkeypatch.setattr(multi, "create_task", create)
    if fault == "archive":
        original_post = worker.service._client.post

        def post(url, **kwargs):
            if kwargs.get("json", {}).get("taskId") == "1":
                raise RuntimeError("archive failed")
            return original_post(url, **kwargs)

        monkeypatch.setattr(worker.service._client, "post", post)
    if fault == "verify":
        monkeypatch.setattr(multi, "verify_outcomes", lambda: (_ for _ in ()).throw(RuntimeError("verify failed")))
    manager = LocalSyncJobManager(multi_prediction_service_factory=lambda: worker.service)
    try:
        result = wait_finished(manager, manager.start_multi_predictions().job_id)
        assert len(calls) == 3
        assert result.status == "PARTIAL_SUCCESS"
        assert result.fetched_count == 804
        assert sum(result.result_summary["counts"].values()) == 804
        assert result.created_count + result.updated_count + result.skipped_count == 804
        if fault in {"archive", "verify"}:
            assert result.skipped_count == 0
            assert len(result.result_summary["followupIssues"]) == 1
        else:
            assert result.created_count == 303
            assert result.result_summary["counts"]["WAITING" if fault == "timeout" else "ERROR"] == 300
    finally:
        manager.close()


def test_prediction_period_identity_never_reuses_another_date():
    from app.services.prediction_generation import prediction_period_key

    value = {"fundCode": "002112", "horizonId": "T5_V1", "startDate": "2026-09-30"}
    assert prediction_period_key(value) == prediction_period_key(dict(value))
    assert prediction_period_key(value) != prediction_period_key(value | {"startDate": "2026-10-09"})


def test_missing_source_identity_does_not_block_known_fund_details(isolated_db):
    worker = object.__new__(funds.TushareFundSyncService)
    worker._engine = isolated_db
    worker._provider = SimpleNamespace(resolve_fund_basics_by_fund_codes=lambda codes: ())
    known = SimpleNamespace(fund_code="000001", source_fund_code="000001.OF")
    unknown = SimpleNamespace(fund_code="000002", source_fund_code=None)
    codes, issues = worker._resolve_detail_source_codes((known, unknown))
    assert codes == ("000001.OF",) and len(issues) == 1 and "000002" in issues[0]


def test_detail_phase_failure_does_not_stop_unrelated_phases(monkeypatch, isolated_db):
    worker = object.__new__(funds.TushareFundSyncService)
    worker._engine = isolated_db
    worker._start_run = lambda **kw: (uuid4(), uuid4())
    worker._record_failure = lambda *args: None
    worker._complete_run = lambda *args: None
    monkeypatch.setattr(funds, "_market_nav_incremental_lock", lambda _engine: nullcontext())
    monkeypatch.setattr(
        funds,
        "list_active_market_sync_targets",
        lambda session: (SimpleNamespace(fund_code="000001", source_fund_code="000001.OF"),),
    )
    calls = []

    def fail(*args):
        calls.append("PROFILE")
        raise RuntimeError("injected profile failure")

    worker._sync_market_detail_profiles = fail

    def succeed(name):
        def action(*args, **kwargs):
            calls.append(name)
            return funds.SyncOutcome(uuid4(), name, None, 1, 1, 0, 0)

        return action

    worker._sync_market_nav_history = succeed("NAV")
    worker._sync_market_detail_managers = succeed("MANAGER")
    worker._sync_market_detail_shares = succeed("SHARE")
    worker._sync_market_detail_dividends = succeed("DIVIDEND")
    result = worker.sync_market_details(history_end_date=date(2026, 9, 30))
    assert calls == ["PROFILE", "NAV", "MANAGER", "SHARE", "DIVIDEND"]
    assert result.overall_outcome.status == "PARTIAL_SUCCESS"
    assert result.overall_outcome.created_count == 4
