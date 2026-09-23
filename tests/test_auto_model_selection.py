"""自动流程的协议、迁移和并发边界；隔离schema不切换真实模型。"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from app.services import auto_model_contract as contract
from app.services.prediction_contract import PredictionFailure
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from tests.test_prediction_database import database  # noqa: F401


def test_policy_missing_fails_closed(tmp_path, monkeypatch):
    assert contract.auto_policy()["maximumBundles"] == 4
    monkeypatch.setattr(contract, "POLICY_FILE", tmp_path / "missing.json")
    with pytest.raises(PredictionFailure, match="配置缺失"):
        contract.auto_policy()


def test_weekly_recovery_and_rolling_window():
    policy = contract.auto_policy()
    now = datetime(2026, 9, 23, 15, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert contract.weekly_slot(now, policy) == "2026-09-20T02:00:00+08:00"
    window = contract.rolling_window(now, policy)
    assert window == dict(
        trainStart="2023-09-22", trainEnd="2025-09-22", validationEnd="2026-03-22", selectionEnd="2026-09-22"
    )
    assert contract.thaw_value(contract.freeze_value({"now": now})) == {"now": now}


def test_research_admission_does_not_block_live_worker():
    from threading import Event

    from app.services.prediction_generation import _executor as live_worker
    from app.services.prediction_research import ResearchExecutor

    gate = Event()
    research = ResearchExecutor()
    try:
        first = research.submit(lambda: gate.wait(5))
        assert research.submit(lambda: None) is None
        # 同时占用研究线程，日常预测的独立线程仍能调度；真实预测耗时另由运行证据验收。
        assert live_worker.submit(lambda: "LIVE_WORKER_AVAILABLE").result(timeout=1) == "LIVE_WORKER_AVAILABLE"
        gate.set()
        assert first.result(timeout=1)
    finally:
        gate.set()
        research.shutdown(wait=True)


def test_additive_migration_and_immutable_input(database):  # noqa: F811
    from uuid import uuid4

    cycle = uuid4()
    with database.begin() as c:
        c.execute(
            text("""INSERT INTO prediction_auto_cycle
              (cycle_id,request_key,scope_hash,data_hash,protocol_hash,status,trigger_reason,spec)
          VALUES(:id,:hash,:hash,:hash,:hash,'QUEUED','INITIAL','{}')"""),
            {"id": cycle, "hash": "a" * 64},
        )
        c.execute(
            text("""INSERT INTO prediction_auto_input(cycle_id,input_key,content_hash,payload)
          VALUES(:id,'test',:hash,'{}')"""),
            {"id": cycle, "hash": "b" * 64},
        )
        assert c.execute(text("SELECT count(*) FROM model_route")).scalar() == 3
    with database.begin() as c, pytest.raises(Exception, match="append only"):
        c.execute(text("UPDATE prediction_auto_input SET payload='[]'"))
    with database.begin() as c, pytest.raises(IntegrityError):
        c.execute(
            text("""INSERT INTO prediction_auto_cycle
              (cycle_id,request_key,scope_hash,data_hash,protocol_hash,status,trigger_reason,spec)
          VALUES(:id,:key,:scope,:key,:key,'QUEUED','WEEKLY','{}')"""),
            {"id": uuid4(), "key": "c" * 64, "scope": "a" * 64},
        )
