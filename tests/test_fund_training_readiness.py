"""独立历史扩展的真实边界：不放宽标签、日历、缺失与时间规则。"""

from copy import deepcopy
from datetime import date, timedelta

import pytest
from app.integrations.dbfund_reports import parse_text
from app.services import fund_training_readiness as readiness
from app.services.fund_exposure_features import calculate
from tests.test_fund_exposure import bundle


def test_extended_calendar_keeps_official_corrections_and_c_share_start():
    days, version = readiness.sessions()
    assert days[0] == date(2015, 11, 16)
    assert days[-1] == date(2024, 12, 31)
    assert len(days) == len(set(days)) == 2221
    assert date(2018, 12, 31) not in days
    assert date(2019, 5, 2) not in days
    assert date(2020, 1, 31) not in days
    assert date(2020, 2, 3) in days
    assert len(version) == 64


def test_calendar_hash_cannot_be_silently_changed(tmp_path, monkeypatch):
    path = tmp_path / "calendar.json"
    path.write_text(
        readiness.CALENDAR_FILE.read_text(encoding="utf-8").replace('"2019-05-04"', '"2019-05-01"'), encoding="utf-8"
    )
    monkeypatch.setattr(readiness, "CALENDAR_FILE", path)
    with pytest.raises(ValueError, match="CALENDAR_HASH"):
        readiness.sessions()


def history():
    days = readiness.sessions()[0][:70]
    rows = [
        {"date": str(d), "nav": str(1 + i / 10000), "ann_date": str(d + timedelta(days=1)), "source_hash": str(i)}
        for i, d in enumerate(days)
    ]
    return {"rows": rows}, days


def test_nav_windows_reject_unknown_and_future_publication_and_gaps():
    nav, days = history()
    rows, _ = readiness.nav_rows(nav, days)
    assert rows and all(len(r["nav_input_dates"]) == 61 for r in rows)
    for kind, reason in [
        ("unknown", "NAV_PUBLICATION_UNKNOWN"),
        ("future", "NAV_NOT_PUBLIC_BY_TARGET"),
        ("gap", "NAV_GAP"),
    ]:
        changed = deepcopy(nav)
        if kind == "gap":
            changed["rows"].pop(35)
        else:
            changed["rows"][35]["ann_date"] = None if kind == "unknown" else "2026-09-25"
        accepted, excluded = readiness.nav_rows(changed, days)
        assert not accepted
        assert {r["reason"] for r in excluded} == {reason}


def test_no_a_share_backfill_and_exact_flat_classification():
    nav, days = history()
    nav["rows"][61]["nav"] = nav["rows"][60]["nav"]
    rows, _ = readiness.nav_rows(nav, days)
    assert rows[0]["fund_code"] == "002112"
    assert rows[0]["actual_direction"] == "FLAT"
    assert rows[0]["target"] == str(days[61])
    assert rows[0]["mature_at"] > rows[0]["as_of"]


def old_table():
    return [
        """德邦鑫星价值 002112 报告送出日期：2016年8月26日
期末基金资产净值 600000.00 400000.00
所有者权益合计 1000000.00 800000.00
7.1 期末基金资产组合情况
1 权益投资 200040.00 20.00
7.2 期末按行业分类的股票投资组合
C 制造业 200000.00 20.00
M 科学研究和技术服务业 40.00 －
合计 200040.00 20.00
7.3 期末按公允价值占基金资产净值比例大小排序的所有股票投 资明 细
1 600519 公司甲 10 200000.00 20.00
2 603909 公司乙 10 40.00 －
7.4 报告期内股票投资组合的重大变动
"""
    ]


def test_old_pdf_dash_is_explicit_derivation_not_zero():
    parsed = parse_text(old_table(), "2016年半年度报告", derive_missing_weights=True)
    row = parsed["holdings"][1]
    assert row["reported_nav_weight_pct"] is None
    assert float(row["nav_weight_pct"]) == 0.004
    assert row["denominator_nav_cny"] == "1000000.00"
    assert row["weight_basis"] == "PUBLISHED_VALUE_DIVIDED_BY_PUBLISHED_FUND_NAV"
    assert parsed["reported_industries"][1]["nav_weight_pct"] is None
    with pytest.raises(ValueError, match="FULL_HOLDING_TOTAL_MISMATCH"):
        parse_text(old_table(), "2016年半年度报告")


def test_old_pdf_derivation_requires_whole_fund_denominator():
    pages = [old_table()[0].replace("所有者权益合计 1000000.00", "所有者权益合计 600000.00")]
    with pytest.raises(ValueError, match="DENOMINATOR_MISMATCH"):
        parse_text(pages, "2016年半年度报告", derive_missing_weights=True)


def test_research_calendar_cannot_enter_live_prediction():
    data, base, at = bundle()
    with pytest.raises(ValueError, match="NOT_FOR_LIVE"):
        calculate(data, base, at, live=True, research_sessions=readiness.sessions()[0])
