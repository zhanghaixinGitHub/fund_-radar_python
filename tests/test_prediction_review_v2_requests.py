"""正式切换后只走新请求，不能悄悄生成旧版或把缺净值算作完成。"""

from contextlib import nullcontext
from datetime import datetime
from types import SimpleNamespace

import pytest
from app.services import prediction_review_v2_requests as service
from app.services.prediction_contract import PredictionFailure


def test_adopted_version_dispatches_before_legacy_generator(monkeypatch):
    from app.services import prediction_generation

    expected = {"status": "PENDING_ANALYSIS", "createdItems": 0}
    monkeypatch.setattr(service, "enabled_v2", lambda: True)
    monkeypatch.setattr(service, "request_batch", lambda codes, **kw: expected if codes == ["013330"] else None)
    monkeypatch.setattr(prediction_generation, "get_engine", lambda: pytest.fail("正式第二版不得创建旧版生成任务"))
    assert prediction_generation.create_task(["013330"]) is expected


def test_new_task_read_does_not_read_legacy_task_or_modify_it(monkeypatch):
    from app.services import prediction_generation

    expected = {"status": "WAITING_FOR_NAV", "createdItems": 0}
    monkeypatch.setattr(service, "find_task", lambda task: expected)
    monkeypatch.setattr(prediction_generation, "get_engine", lambda: pytest.fail("不能把新任务当成旧任务"))
    assert prediction_generation.task_status("test") is expected


@pytest.mark.parametrize("ready,expected", [(False, "WAITING_FOR_NAV"), (True, "PENDING_ANALYSIS")])
def test_request_retains_new_window_and_never_invents_completed_direction(monkeypatch, ready, expected):
    memory = {}

    class Connection:
        def execute(self, statement, values):
            if str(statement).startswith("INSERT"):
                import json

                memory["value"] = json.loads(values["payload"])
            return SimpleNamespace(scalar_one=lambda: memory["value"])

    monkeypatch.setattr(service, "enabled_v2", lambda: True)
    monkeypatch.setattr(service, "current_holdings", lambda: {"scope": ["013330"], "owner_hash": "owner"})
    monkeypatch.setattr(service, "now", lambda: datetime.fromisoformat("2026-11-12T17:00:00+08:00"))
    monkeypatch.setattr(
        service, "_readiness", lambda codes, instant: ({"base_nav_date": "2026-11-12"}, {"013330": ready})
    )
    monkeypatch.setattr(service, "get_engine", lambda: SimpleNamespace(begin=lambda: nullcontext(Connection())))
    result = service.request_batch(["013330"])
    assert result["status"] == expected
    assert result["requiredBaseNavDate"] == "2026-11-12"
    assert result["createdItems"] == 0 and result["pendingItems"] == 4
    assert result["generatedAt"] is None
    assert all("direction" not in item["result"] for item in result["items"])
    assert {item["horizonId"] for item in result["items"]} == set(service.HORIZONS)
    assert "_ownerHash" not in result


def test_removed_holding_rejects_before_reading_or_writing_requests(monkeypatch):
    monkeypatch.setattr(service, "enabled_v2", lambda: True)
    monkeypatch.setattr(service, "current_holdings", lambda: {"scope": []})
    monkeypatch.setattr(service, "_readiness", lambda *args: pytest.fail("已清仓不得继续准备新请求"))
    with pytest.raises(PredictionFailure, match="当前持仓"):
        service.request_batch(["013330"])


@pytest.mark.parametrize("mismatch", [None, "old_window", "owner", "after_deadline", "unverified_snapshot"])
def test_only_real_current_review_completes_request_and_conditional_is_separate(mismatch):
    payload = {
        "analysisPolicyHash": "policy",
        "_ownerHash": "owner",
        "planningWindow": {"base_nav_date": "2026-11-12", "target_nav_date": "2026-11-13"},
        "items": [{"fundCode": "013330", "horizonId": h, "status": "PENDING"} for h in ["T1", "M6_V1"]],
    }
    identity = {
        "owner_hash": "owner",
        "base_date": "2026-11-12",
        "target_date": "2026-11-13",
        "snapshot_verified": True,
        "materials_input_hash": "actual-material",
        "ex_ante_eligible": True,
        "end_date": "2026-11-13",
    }
    if mismatch == "old_window":
        identity["base_date"] = "2026-11-11"
    if mismatch == "owner":
        identity["owner_hash"] = "another-owner"
    if mismatch == "unverified_snapshot":
        identity["snapshot_verified"] = False
    evaluated = "2026-11-13T15:01:00+08:00" if mismatch == "after_deadline" else "2026-11-13T00:30:00+08:00"
    prepared = {
        "policyHash": "policy",
        "archive": {
            "cases": [
                {
                    "case": {"fund_code": "013330", "horizon": h, "decision": "UP"},
                    "analysis": {"evaluated_at": evaluated},
                    "workflow_identity": {**identity, "end_date": None if h == "M6_V1" else identity["end_date"]},
                }
                for h in ["T1", "M6_V1"]
            ]
        },
    }
    result = service._completed_payload(
        payload, prepared, "release", datetime.fromisoformat("2026-11-13T00:31:00+08:00")
    )
    if mismatch:
        assert result is None
    else:
        assert result["status"] == "SUCCEEDED" and result["pendingItems"] == 0
        assert result["createdItems"] == 1 and result["conditionalItems"] == 1
