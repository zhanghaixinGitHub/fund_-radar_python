"""净值覆盖技术算例，不是实际基金、官方估值日历或校准样本。"""

import calendar
from datetime import date, datetime

import pytest
from app.services.fund_quality_v2_nav import rebuild_nav_coverage


@pytest.fixture
def original(monkeypatch):
    dates = [
        date(y, m, calendar.monthrange(y, m)[1])
        for y in range(2023, 2027)
        for m in range(1, 13)
        if date(2023, 9, 1) <= date(y, m, 1) <= date(2026, 9, 1)
    ]
    monkeypatch.setattr("app.services.direction_1d_protocol.calendar", lambda: (dates, "TEST_CALENDAR"))
    days = list(map(str, dates))
    return {
        "fund_code": "123456",
        "history": {
            "calendar_hash": "TEST_CALENDAR",
            "dates": days,
            "month_ends": days,
            "rows": [{"nav_date": d, "ann_date": d, "adjusted_nav": "1.23", "content_hash": "TEST_ONLY"} for d in days],
        },
    }


def test_own_nav_is_rebuilt_without_claiming_a_benchmark(original):
    facts = rebuild_nav_coverage(original, fund_code="123456", as_of=date(2026, 9, 30))
    assert facts["status"] == "OWN_ADJUSTED_NAV_WINDOW_REBUILT"
    assert len(facts["curve"]["adjusted_nav"]) == 37
    assert "NOT_ASSESSED" in facts["benchmark_status"]


def test_missing_adjusted_value_is_reported_and_never_zero_filled(original):
    original["history"]["rows"][10]["adjusted_nav"] = None
    facts = rebuild_nav_coverage(original, fund_code="123456", as_of=date(2026, 9, 30))
    assert "curve" not in facts
    assert facts["missing_adjusted_nav_dates"] == [original["history"]["dates"][10]]


def test_economic_day_and_later_publication_are_not_conflated(original):
    original["history"]["rows"][-1]["ann_date"] = "2026-10-08"
    facts = rebuild_nav_coverage(original, fund_code="123456", as_of=date(2026, 9, 30))
    assert facts["late_announcements"] == [{"nav_date": "2026-09-30", "announced_on": "2026-10-08"}]
    assert "curve" not in facts and facts["status"] == "OWN_NAV_PUBLICATION_AFTER_AS_OF"
    with pytest.raises(ValueError, match="IDENTITY"):
        rebuild_nav_coverage(original, fund_code="654321", as_of=date(2026, 9, 30))


def test_later_true_cutoff_keeps_economic_date_without_rolling_or_filling_nav(original):
    original["history"]["rows"][-1]["ann_date"] = "2026-10-01"
    facts = rebuild_nav_coverage(original, fund_code="123456", as_of=date(2026, 10, 1))
    assert facts["status"] == "OWN_ADJUSTED_NAV_WINDOW_REBUILT"
    assert facts["curve"]["dates"][-1] == facts["valuation_end"] == "2026-09-30"
    assert facts["availability_cutoff"] == "2026-10-01"


def test_unknown_publication_is_not_imputed_from_nav_or_local_capture_date(original):
    original["history"]["rows"][-1]["ann_date"] = None
    facts = rebuild_nav_coverage(original, fund_code="123456", as_of=date(2026, 9, 30))
    assert facts["unknown_announcement_dates"] == ["2026-09-30"]
    assert facts["status"] == "OWN_NAV_PUBLICATION_DATE_UNVERIFIED" and "curve" not in facts


def test_candidate_fields_from_an_older_full_bundle_cannot_bypass_publication_gate():
    from app.services.fund_quality_v2_materials import apply_extraction

    packet = {"metrics": {"relative_return_3y": {"value": "0.2"}},
              "inventory": {"relative_return_3y": {"value": "0.2", "missing_roles": ["benchmark"]}},
              "as_of_date": "2026-09-30"}
    extracted = {"facts": {"own_nav_coverage": {"late_announcements": [
        {"nav_date": "2026-09-30", "announced_on": "2026-10-01"}], "source_hashes": ["TEST_ONLY"]}},
        "document_errors": {}, "metrics": {"relative_return_3y": {"value": "0.2"}}, "findings": {}}
    apply_extraction(packet, extracted)
    assert "relative_return_3y" not in packet["metrics"]
    assert packet["inventory"]["relative_return_3y"]["reason"] == "NAV_ANNOUNCEMENT_AFTER_AS_OF"
    assert "value" not in packet["inventory"]["relative_return_3y"]
    assert "missing_roles" not in packet["inventory"]["relative_return_3y"]


@pytest.fixture
def database_original(original):
    original.update(
        schema="QUALITY_DATABASE_NAV_SNAPSHOT_V1",
        origin="LOCAL_READ_ONLY_NORMALIZED_SOURCE_RECORDS",
        as_of_date="2026-09-30",
        captured_at="2026-10-17T01:00:00+08:00",
        source_registry={
            "source_id": "TEST_ONLY",
            "source_code": "TUSHARE_PRO_FUND",
            "enabled": True,
            "authorization_verified_at": "2026-10-01T00:00:00+08:00",
            "authorized_api_names": ["fund_nav"],
            "retention_days": 30,
        },
        query="SELECT TEST_ONLY FROM nav_daily",
        query_scope={"fund_codes": ["123456"], "to": "2026-09-30"},
    )
    for row in original["history"]["rows"]:
        row.update(fund_code="123456", created_at="2026-10-01T00:00:00+08:00", updated_at="2026-10-01T00:00:00+08:00")
    return original


@pytest.mark.parametrize("change", [None, "missing_nav", "wrong_fund", "future_row", "wrong_as_of", "future_capture"])
def test_database_snapshot_retains_origin_and_checks_records_instead_of_success_flags(database_original, change):
    from app.services.fund_quality_v2_database_nav import database_nav_coverage

    if change == "missing_nav":
        database_original["history"]["rows"][0]["adjusted_nav"] = None
    elif change == "wrong_fund":
        database_original["history"]["rows"][0]["fund_code"] = "654321"
    elif change == "future_row":
        database_original["history"]["rows"][0]["updated_at"] = "2026-10-18T00:00:00+08:00"
    elif change == "wrong_as_of":
        database_original["as_of_date"] = "2026-10-31"
    elif change == "future_capture":
        database_original["captured_at"] = "2026-10-18T00:00:00+08:00"
    arguments = dict(
        fund_code="123456", as_of=date(2026, 9, 30), frozen=datetime.fromisoformat("2026-10-17T02:00:00+08:00")
    )
    if change in {"wrong_fund", "future_row", "wrong_as_of", "future_capture"}:
        with pytest.raises(ValueError, match="QUALITY_DATABASE_NAV"):
            database_nav_coverage(database_original, **arguments)
    else:
        result = database_nav_coverage(database_original, **arguments)
        assert "NOT_RAW_HTTP" in result["source_kind"]
        assert ("curve" in result) == (change is None)
