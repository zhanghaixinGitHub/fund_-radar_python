"""评价缺口必须有审计依据，损坏、未来日期和未覆盖基金不得变成零分。"""

from datetime import datetime, timedelta

import pytest
from app.services.direction_1d_protocol import ZONE
from app.services.fund_evaluation_status import batch
from app.services.fund_exposure_common import save


def evidence(tmp_path, **override):
    value = {
        "at": datetime.now(ZONE).isoformat(),
        "funds": [
            {"fund_code": "002112", "fund_type": "MIXED", "invest_type": "混合型", "last_nav_date": "2026-09-28"}
        ],
        "source_categories": [{"type": "MIXED", "invest_type": "混合型", "share_count": 16}],
    }
    value.update(override)
    save(tmp_path / "coverage/20260929-120000.json", value)


def test_known_and_unknown_keep_null_score_and_distinct_coverage(tmp_path):
    evidence(tmp_path)
    known, unknown = batch(["002112", "999999", "002112"], root=tmp_path)["items"]
    assert known["coverageCheckedAt"] and unknown["coverageCheckedAt"] is None
    assert known["score"] is unknown["score"] is None
    assert known["rank"] is known["scoreDate"] is None
    assert "同类资料不足" in known["reasons"][0]
    assert unknown["navAsOfDate"] is None


def test_absent_audit_is_unknown_not_zero(tmp_path):
    item = batch(["002112"], root=tmp_path)["items"][0]
    assert item["score"] is item["coverageCheckedAt"] is None


@pytest.mark.parametrize("codes", [[], ["002112"] * 101, [".."], ["００２１１２"]])
def test_bounded_ascii_scope(codes, tmp_path):
    with pytest.raises(ValueError):
        batch(codes, root=tmp_path)


def test_future_coverage_is_rejected(tmp_path):
    evidence(tmp_path, at=(datetime.now(ZONE) + timedelta(days=1)).isoformat())
    with pytest.raises(ValueError):
        batch(["002112"], root=tmp_path)


def test_corrupt_audit_does_not_silently_downgrade_to_empty(tmp_path):
    evidence(tmp_path)
    (tmp_path / "coverage/20260929-120000.json").write_text('{"payload":{}}', encoding="utf-8")
    with pytest.raises(ValueError):
        batch(["002112"], root=tmp_path)


def test_api_auth_scope_and_single_batch_read(monkeypatch):
    from app.api.dependencies import require_service_token
    from app.api.routes import fund_materials as routes
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    assert client.get("/evaluation-status?fundCodes=002112").status_code == 403
    app.dependency_overrides[require_service_token] = lambda: None
    calls = []

    def read_batch(codes):
        calls.append(codes)
        return {"items": []}

    monkeypatch.setattr(routes.fund_evaluation_status, "batch", read_batch)
    for value in ("", "../bad", "002112,", ",".join(["002112"] * 101)):
        assert client.get("/evaluation-status", params={"fundCodes": value}).status_code == 422
    assert calls == []
    assert client.get("/evaluation-status?fundCodes=002112,001412").status_code == 200
    assert calls == [["002112", "001412"]]
