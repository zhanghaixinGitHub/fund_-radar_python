"""半年按自然月定目标，到期真实估值日核对，不改短周期及历史规则。"""

from datetime import date

import pytest
from app.services.prediction_contract import PredictionFailure, ValuationCalendar
from app.services.prediction_natural_month import RULE, actual_month_endpoint, natural_month_target


@pytest.mark.parametrize("base,target", [
    ("2026-11-18", "2027-05-18"), ("2026-08-31", "2027-02-28"),
    ("2023-08-31", "2024-02-29"), ("2026-05-30", "2026-11-30"),
])
def test_six_calendar_months_clamp_without_future_calendar(base, target):
    start = date.fromisoformat(base).toordinal() + 1
    result = natural_month_target(base, date.fromordinal(start).isoformat())
    assert result["endDate"] == target
    assert result["endDateStatus"] == "RESOLVED"
    assert result["endDateBasis"] == "CALENDAR_MONTH"
    assert result["outcomeRule"] == RULE


def test_actual_endpoint_uses_last_valuation_session_and_waits_for_missing_nav():
    friday, saturday, monday = date(2027, 5, 14), date(2027, 5, 15), date(2027, 5, 17)
    calendar = ValuationCalendar("verified", (friday, monday), friday, monday, "source")
    assert actual_month_endpoint(saturday, calendar, {friday}) == {
        "targetDate": "2027-05-15", "actualNavDate": "2027-05-14", "calendarDayGap": 1,
    }
    with pytest.raises(PredictionFailure, match="净值尚未公布"):
        actual_month_endpoint(monday, calendar, {friday})
    with pytest.raises(PredictionFailure, match="日历未覆盖"):
        actual_month_endpoint(date(2027, 5, 18), calendar, {friday, monday})


def test_formal_publication_preserves_natural_endpoint_and_not_conditional_kind():
    from app.services.prediction_review_v2_publication import _public_period

    period = {"base_date": "2026-11-18", "start_date": "2026-11-19", "end_date": "2027-05-18",
              "end_date_basis": "CALENDAR_MONTH", "outcome_rule": RULE}
    analysis = {"primary_driver": "未来半年主要经营驱动有持续性", "counter_tradeoff": "估值波动仍可能抵消经营改善",
                "change_conditions": ["经营驱动未持续兑现则重新判断"]}
    row = {"horizon": "M6_V1", "period": period, "decision": "UP", "source": {"kind": "STRICT_V2_TRIAL"}}
    view = _public_period(row, analysis, {}, {}, "2026-11-18T16:00:00+08:00")
    assert view["kind"] == "DATED_ANALYSIS"
    assert view["endDate"] == "2027-05-18" and view["outcomeRule"] == RULE
