"""同步失败修复的边界回归：不联网、不写业务数据，不降低原有完整性要求。"""

from datetime import date, datetime

import pytest
from app.integrations.fund_report_layout_v3 import parse_text
from app.services.fund_exposure_quotes import published_quote_end
from app.services.nav_repair import expected_dates
from tests.test_fund_report_sections_v2 import pages


@pytest.mark.parametrize("row", [
    "1 600000浦发银行 10 100.00 50.00",
    "1 600000 浦发银 10 100.0 50.00\n行 0",
    "1 600000 浦发银\n行\n10 100.00 50.00",
])
def test_report_layout_preserves_identity_amount_and_weight(row):
    body = pages()[0].replace("1 600000 浦发银行 10 100.00 50.00", row)
    report = parse_text([body], "德邦鑫星价值2021年第1季度报告")
    assert report["holdings"] == [{
        "reported_rank": 1, "stock_code": "600000.SH", "stock_name": "浦发银行",
        "quantity_shares": "10", "market_value_cny": "100.00", "nav_weight_pct": "50.00",
    }]


def test_industry_split_names_and_total_decimal_are_reconstructed_from_original_digits():
    body = pages()[0].replace("C 制造业 100.00 50.00", "制\n\nC 100.00 50.00\n造业")
    body = body.replace("合计 100.00 50.00", "合计 100.0 50.00\n0")
    report = parse_text([body], "德邦鑫星价值2021年第1季度报告")
    assert report["stock_value_cny"] == "100.00"
    assert report["reported_industries"][0]["name"] == "制造业"


@pytest.mark.parametrize("change", [
    ("10 100.00 50.00", "10 100.0 50.00"),
    ("10 100.00 50.00", "10 100.0 50.00\n12"),
    ("C 制造业 100.00", "C 制造业 99.00"),
    ("1 600000", "2 600000"),
    ("002112", "999999"),
])
def test_bad_or_missing_digits_identity_ranks_and_totals_still_fail(change):
    with pytest.raises(ValueError):
        parse_text([pages()[0].replace(*change)], "德邦鑫星价值2021年第1季度报告")


def test_preopen_nav_does_not_require_daily_but_real_interior_gap_remains():
    sessions = tuple(date(2026, 9, day) for day in (21, 22, 23, 24, 25, 28, 29, 30))
    existing = {sessions[0], sessions[3], sessions[4], sessions[6]}
    args = {"now": datetime.fromisoformat("2026-09-30T09:00:00+08:00"), "found_date": sessions[0]}
    wanted = expected_dates(sessions, existing, sessions[-1], purchase_start_date=sessions[4],
                            redemption_start_date=sessions[3], **args)
    assert tuple(day for day in wanted if day not in existing) == (date(2026, 9, 28),)
    conservative = expected_dates(sessions, existing, sessions[-1], **args)
    assert date(2026, 9, 22) in conservative
    with pytest.raises(ValueError, match="OPEN_DATE_INVALID"):
        expected_dates(sessions, existing, sessions[-1], purchase_start_date=date(2026, 9, 18), **args)


@pytest.mark.parametrize("at,expected", [
    ("2026-09-30T09:00:00+08:00", "2026-09-29"),
    ("2026-09-30T16:59:59+08:00", "2026-09-29"),
    ("2026-09-30T17:00:00+08:00", "2026-09-30"),
    ("2026-09-30T09:00:00+00:00", "2026-09-30"),
    ("2026-10-03T18:00:00+08:00", "2026-09-30"),
])
def test_daily_quote_cutoff_uses_publication_time_and_session_calendar(at, expected):
    sessions = (date(2026, 9, 29), date(2026, 9, 30), date(2026, 10, 9))
    assert published_quote_end(sessions, datetime.fromisoformat(at)) == date.fromisoformat(expected)
