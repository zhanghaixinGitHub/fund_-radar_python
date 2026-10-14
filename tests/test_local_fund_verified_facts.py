"""关键单位、合并空列、阈值边界及冻结经理统计的独立计算回归。"""

from datetime import date, timedelta
from decimal import Decimal

import pytest
from scripts.local_fund_verified_facts import (
    annual_average_bases,
    metrics_from_frozen_navs,
    quantity_facts,
    table_facts,
    threshold_tables,
    total_table_text,
    vesting_ratio,
)


def test_share_conversion_and_currency_keep_original_offsets():
    body = "代表【18,138,687】股。零息本金39.6 亿美元；金额11亿元。"
    rows = quantity_facts(body)
    share = next(r for r in rows if r["unit"] == "股")
    assert Decimal(share["base_value"]) / Decimal(100000000) == Decimal("0.18138687")
    dollar = next(r for r in rows if r["unit"] == "亿美元")
    assert dollar["dimension"] == "USD" and Decimal(dollar["base_value"]) == Decimal(3960000000)
    assert all(body[r["start"] : r["end"]] == r["raw"] for r in rows)
    assert all(r["subject_period_relation"] == "REQUIRES_SOURCE_BINDING" for r in rows)


def test_bad_comma_and_price_per_share_are_not_total_amounts():
    assert quantity_facts("1,2亿元；120元/股") == []


def test_yuan_unit_does_not_invent_currency_without_literal_binding():
    rows = quantity_facts("金额11亿元；人民币11亿元；本金39.6亿美元。")
    assert [r["currency"] for r in rows] == ["NOT_BOUND", "CNY", "USD"]
    assert rows[0]["dimension"] == "YUAN" and rows[0]["base_value"] == "1100000000"


def test_annual_average_does_not_change_profit_positive_condition():
    body = "可分配利润为正；累计分红不低于三年实现的年均可分配利润的30%。"
    rows = annual_average_bases(body)
    assert len(rows) == 1 and rows[0]["groups"]["basis"] == "三年实现的年均可分配利润"
    assert annual_average_bases("三年实现的可分配利润") == []


@pytest.mark.parametrize("actual,expected", [(119, "0"), (120, "0.8"), (135, "0.9"), (150, "1")])
def test_trigger_allows_partial_vesting(actual, expected):
    assert vesting_ratio(Decimal(actual), Decimal(150), Decimal(120)) == Decimal(expected)


def test_invalid_thresholds_fail():
    with pytest.raises(ValueError, match="INVALID_VESTING"):
        vesting_ratio(Decimal(100), Decimal(120), Decimal(150))


@pytest.mark.parametrize(
    "header,first,second",
    [
        ("目标值（Am） 触发值（An）", "150%", "120%"),
        ("触发值(An) 目标值(Am)", "120%", "150%"),
    ],
)
def test_threshold_year_and_column_order_come_from_literal_header(header, first, second):
    body = f"净利润增长率（A）\n{header}\n第一个归属期 2027 年 {first} {second}\n考核指标\nAn≤A<Am X=A/Am"
    table = threshold_tables(body)[0]
    row = table["rows"][0]
    assert row["year"] == "2027" and row["target"] == "150" and row["trigger"] == "120"
    assert row["status"] == "BOUND_LITERAL_ROW"
    assert table["partial_vesting_formulas"][0]["groups"]["condition"] == "An≤A<Am"
    assert body[row["start"] : row["end"]] == row["raw"]
    assert threshold_tables("没有表头：第一个归属期2027年150%120%") == []


def test_merged_total_has_income_but_no_input_value():
    units = [
        {
            "id": "H",
            "cells": [
                {"column_start": 3, "column_end": 3, "text": "实际投入金额（万元）"},
                {"column_start": 5, "column_end": 5, "text": "实际收益（万元）"},
            ],
        },
        {
            "id": "R",
            "kind": "table_row",
            "header_rows": ["H"],
            "page": 4,
            "cells": [
                {"column_start": 1, "column_end": 4, "text": "合计"},
                {"column_start": 5, "column_end": 5, "text": "85.73"},
            ],
        },
    ]
    rows = {r["column"]: r for r in table_facts(units)}
    assert rows[3]["value"] is None and rows[3]["status"] == "NO_NUMERIC_SOURCE"
    assert rows[3]["merged_cells"][0]["text"] == "合计"
    assert rows[5]["value"] == "85.73" and rows[5]["status"] == "SOURCE_NUMERIC"
    text = total_table_text(list(rows.values()))
    assert "实际投入金额：原合计行未列出" in text and "实际收益：85.73万元" in text


def navs():
    return [
        {
            "nav_date": str(date(2026, 1, 1) + timedelta(days=i)),
            "ann_date": str(date(2026, 1, 1) + timedelta(days=i)),
            "adjusted_nav": str(v),
            "updated_at": "2026-02-01T00:00:00+00:00",
        }
        for i, v in enumerate([10, 20, 10, *([15] * 19)])
    ]


def test_independent_metrics_keep_interval_and_overlapping_windows():
    rows = navs()
    result = metrics_from_frozen_navs(rows, "2026-01-01", "2026-01-22", "2026-02-02T00:00:00+00:00")
    assert result["return_pct"] == "50.0000"
    assert result["max_drawdown_pct"] == "50.0000"
    assert result["rolling_windows"] == 2 and result["positive_20d_pct"] == "50.0000"
    assert result["calendar_completeness"] == "CHECK_AGAINST_FROZEN_SOURCE_COUNTS"


def test_late_received_and_duplicate_navs_fail_without_silent_filter():
    rows = navs()
    rows[-1]["updated_at"] = "2026-02-03T00:00:00+00:00"
    with pytest.raises(ValueError, match="AFTER_FROZEN_CUTOFF"):
        metrics_from_frozen_navs(rows, "2026-01-01", "2026-01-22", "2026-02-02T00:00:00+00:00")
    rows = navs()
    with pytest.raises(ValueError, match="DUPLICATE"):
        metrics_from_frozen_navs(rows + [rows[0]], "2026-01-01", "2026-01-22", "2026-02-02T00:00:00+00:00")


def test_frozen_calendar_exclusions_are_explicit_and_missing_session_blocks():
    rows = navs()
    sessions = tuple(r["nav_date"] for i, r in enumerate(rows) if i != 1)
    result = metrics_from_frozen_navs(rows, "2026-01-01", "2026-01-22", "2026-02-02T00:00:00+00:00", sessions)
    assert result["excluded_non_session_dates"] == ["2026-01-02"]
    assert result["observations"] == 21 and result["max_drawdown_pct"] == "0.0000"
    assert result["calendar_completeness"] == "VERIFIED_FROZEN_CALENDAR"
    with pytest.raises(ValueError, match="TRADING_SESSION_NAV_MISSING"):
        metrics_from_frozen_navs(rows[:-1], "2026-01-01", "2026-01-22", "2026-02-02T00:00:00+00:00", sessions)
