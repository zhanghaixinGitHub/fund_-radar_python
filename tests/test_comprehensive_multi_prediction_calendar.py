"""已核实境内实际投向的多周期衔接，不借旧名称过滤放宽境外准入。"""

from contextlib import nullcontext
from datetime import date, datetime, timedelta
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


@pytest.mark.parametrize(
    "market,closed,expected",
    [
        ("HK", ["2026-12-24"], date(2026, 12, 23)),
        ("JP", ["2026-11-03"], date(2026, 11, 2)),
    ],
)
@pytest.mark.parametrize("separator", ["", "/期货", "、期货"])
@pytest.mark.parametrize("common", ["", "共同"])
def test_foreign_calendar_extension_stops_before_first_actual_difference(market, closed, expected, separator, common):
    """即使主窗口核验通过，也不能越过中外假期差异或半日市推断未来估值。"""
    import json

    from app.services.fund_valuation_sessions import hk_closed_dates

    days = tuple(
        date(2026, 10, 1) + timedelta(days=i)
        for i in range(92)
        if (date(2026, 10, 1) + timedelta(days=i)).weekday() < 5
    )
    own = {
        "kind": "PROSPECTUS",
        "codes": ["019449"],
        "body_hash": "own",
        "body": "本基金的估值日为本基金相关的证券" + separator + "交易场所的" + common + "交易日以及规定的非交易日。",
    }
    if market == "HK":
        body = "Holiday Schedule for Year 2026\nSaturdays, Sundays\n" + "\n".join(
            [*(f"{d} January 2026" for d in range(1, 16)), "24 December 2026"]
        )
        foreign = {
            "kind": "EXCHANGE_CALENDAR",
            "body_hash": "hk",
            "body": body,
            "url": "https://www.hkex.com.hk/cal.pdf",
        }
        assert date(2026, 12, 24) in hk_closed_dates(foreign, 2026)
    else:
        foreign = {
            "kind": "EXCHANGE_CALENDAR",
            "body_hash": "jp",
            "url": "https://www.jpx.co.jp/cal",
            "body": json.dumps({"year": 2026, "closed_dates": closed}),
        }
    e = {
        "policy": "CN_NAV_HK_CONNECT_WINDOW_V1" if market == "HK" else "QDII_CNY_VERIFIED_SESSION_V1",
        "prospectus_ref": "own",
        "calendar_ref": "foreign",
        "calendar_refs": {market: "foreign"},
    }
    data = {
        "fund_code": "019449",
        "calendar_evidence": e,
        "window": {"target_nav_date": "2026-10-20"},
        "companies": [{"code": "00179.HK" if market == "HK" else "8035.JP", "quote": {"valuation_market": market}}],
        "market": {},
    }
    actual, end, source = service._matched_future_sessions(
        data, {"own": own, "foreign": foreign}, days, date(2026, 12, 31), "cn"
    )
    assert end == expected and max(actual) <= expected and source == digest(["cn", "own", foreign["body_hash"]])
    own["codes"] = ["000001"]
    with pytest.raises(PredictionFailure, match="条款"):
        service._matched_future_sessions(data, {"own": own, "foreign": foreign}, days, date(2026, 12, 31), "cn")
