"""已核实境内实际投向的多周期衔接，不借旧名称过滤放宽境外准入。"""

from contextlib import nullcontext
from datetime import date, datetime
from types import SimpleNamespace

import pytest
from app.services import comprehensive_multi_prediction as service
from app.services.direction_1d_protocol import digest
from app.services.prediction_contract import PredictionFailure


@pytest.mark.parametrize("data", [{}, {"calendar_evidence": None}, {"calendar_evidence": {}}])
def test_existing_fund_without_supplementary_calendar_keeps_original_path(data, monkeypatch):
    from app.services import prediction_features

    observed = []
    existing_calendar = object()

    def read(code, cutoff):
        observed.append((code, cutoff))
        return {"calendar": existing_calendar}

    monkeypatch.setattr(prediction_features, "read_fund_data", read)
    now = datetime.fromisoformat("2026-10-17T21:00:00+08:00")
    assert service.review_calendar(data, "013330", now) is existing_calendar
    assert observed == [("013330", now)]


def test_verified_domestic_packet_uses_own_calendar_and_rechecks_each_gate(monkeypatch):
    from app.services import comprehensive_calendar, prediction_features, trading_calendar

    definition = SimpleNamespace(coverage_start=date(2026, 1, 1), coverage_end=date(2026, 12, 31))
    past = SimpleNamespace(sessions=(date(2026, 10, 14),), definition=definition, content_hash="past")
    current = SimpleNamespace(sessions=(date(2026, 10, 15),), definition=definition, content_hash="current")
    monkeypatch.setattr(trading_calendar, "load_calendar", lambda: past)
    monkeypatch.setattr(trading_calendar, "load_current_calendar", lambda: current)
    monkeypatch.setattr(service, "get_engine", lambda: SimpleNamespace(connect=lambda: nullcontext(None)))
    monkeypatch.setattr(service.repo, "profiles", lambda *args: [{"fund_code": "008603"}])
    seen = []

    def verify(profile, material, documents, window):
        seen.append((profile, material, documents, window))
        return material.get("originals_verified", False)

    monkeypatch.setattr(comprehensive_calendar, "verified_domestic_window", verify)
    monkeypatch.setattr(prediction_features, "read_fund_data", lambda *args: pytest.fail("不得回到旧名称过滤"))
    packet = {
        "fund_code": "008603",
        "calendar_evidence": {"policy": "CN_NAV_DISCLOSED_DOMESTIC_ASSETS_V1"},
        "time_context": {"baseline_complete": True},
        "window": {"calendar_version": digest(["past", "current"])},
        "documents": [{"id": "own-report", "body": "原文"}],
        "originals_verified": True,
    }
    at = datetime.fromisoformat("2026-10-14T18:00:00+08:00")
    result = service.review_calendar(packet, "008603", at)
    assert result.calendar_id == "CN_NAV_DISCLOSED_DOMESTIC_ASSETS_V1"
    assert result.required_base(at) == date(2026, 10, 14)
    assert seen[0][2] == {"own-report": packet["documents"][0]}
    packet["originals_verified"] = False
    with pytest.raises(PredictionFailure, match="仍待核验"):
        service.review_calendar(packet, "008603", at)
    packet["originals_verified"] = True
    packet["time_context"]["baseline_complete"] = False
    with pytest.raises(PredictionFailure, match="仍待核验"):
        service.review_calendar(packet, "008603", at)
    packet["time_context"]["baseline_complete"] = True
    packet["window"]["calendar_version"] = "changed"
    with pytest.raises(PredictionFailure, match="仍待核验"):
        service.review_calendar(packet, "008603", at)
