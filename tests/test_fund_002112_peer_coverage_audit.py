"""审计的关键防误判边界；用合成日期/报告，不启动训练或访问数据库。"""

from datetime import date, timedelta

import pytest
from app.services.fund_002112_peer_coverage_audit import nav_gate, normalize_section_spacing, report_gate


def test_spacing_does_not_rewrite_numbers_or_subsections():
    text = "5.3报告期末股票投资明细\n5.4报告期末债券\n5.3.1报告期末\n1 600001 名称 15.30 5.3\n"
    assert normalize_section_spacing(text) == (
        "5.3 报告期末股票投资明细\n5.4 报告期末债券\n5.3.1报告期末\n1 600001 名称 15.30 5.3\n"
    )


def test_nav_reason_priority_and_same_day_original_policy():
    days = [str(date(2023, 1, 1) + timedelta(days=i)) for i in range(62)]
    mapping = {d: {"ann_date": days[-1]} for d in days}
    assert nav_gate(mapping, days)["reason"] is None  # 此处复核旧政策，不能偷换为第三轮的严格前一日。
    mapping[days[-2]]["ann_date"] = "2023-12-31"
    assert nav_gate(mapping, days)["reason"] == "NAV_NOT_PUBLIC_BY_TARGET"
    mapping[days[1]]["ann_date"] = None
    assert nav_gate(mapping, days)["reason"] == "NAV_PUBLICATION_UNKNOWN"
    del mapping[days[0]]
    assert nav_gate(mapping, days)["reason"] == "NAV_GAP"


@pytest.mark.parametrize("days", [[], ["2023-01-01"] * 62])
def test_incomplete_or_duplicate_window_is_rejected(days):
    with pytest.raises(ValueError, match="INVALID_NAV_WINDOW"):
        nav_gate({}, days)


def test_latest_report_must_not_fall_back_to_previous_quarter():
    old = {"report_end": "2022-12-31", "report_type": "QUARTER", "available_at": "2023-01-21T08:00:00+08:00"}
    annual = {"report_end": "2022-12-31", "report_type": "ANNUAL", "available_at": "2023-04-01T08:00:00+08:00"}
    assert report_gate([old], [old, annual], "2023-03-31")["reason"] is None
    assert report_gate([old], [old, annual], "2023-04-03")["reason"] == "TRAINING_LATEST_DISCLOSURE_MISSING"
    assert report_gate([], [old, annual], "2023-04-03")["reason"] == "EXPOSURE_NO_AVAILABLE_REPORT"


def test_future_annual_report_does_not_repair_past_training():
    future = {"report_end": "2023-12-31", "report_type": "ANNUAL", "available_at": "2024-03-29T08:00:00+08:00"}
    assert report_gate([future], [future], "2023-12-29")["reason"] == "EXPOSURE_NO_AVAILABLE_REPORT"
