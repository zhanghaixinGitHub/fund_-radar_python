"""防止季度/累计、比较基数和百分比错列；缺单元格必须明确失败。"""

from copy import deepcopy

import pytest
from app.services.fund_earnings_table_facts_v1 import review_money_row

HEADER = "本报告期 上年同期 同比 年初至报告期末"
ROW = "归母净利润（元） -10.25 20.50 -150.00% 100.75"
PAGES = [HEADER + "\n" + ROW]
SPEC = {
    "issuer": "000001",
    "document_id": "example",
    "published_date": "2023-10-28",
    "period_start": "2023-07-01",
    "period_end": "2023-09-30",
    "metric": "NET_PROFIT_ATTRIBUTABLE_TO_PARENT",
    "basis": "CONSOLIDATED_CN_GAAP",
    "currency": "CNY",
    "kind": "REPORTED_RESULT",
    "unit": "元",
    "page": 1,
    "header_quote": HEADER,
    "row_quote": ROW,
    "row_label": "归母净利润（元）",
    "cells": ["-10.25", "20.50", "-150.00%", "100.75"],
    "columns": [
        {"label": "本报告期", "period_start": "2023-07-01", "period_end": "2023-09-30", "value_kind": "AMOUNT"},
        {"label": "上年同期", "period_start": "2022-07-01", "period_end": "2022-09-30", "value_kind": "AMOUNT"},
        {"label": "同比", "period_start": "2023-07-01", "period_end": "2023-09-30", "value_kind": "PERCENT"},
        {"label": "年初至报告期末", "period_start": "2023-01-01", "period_end": "2023-09-30", "value_kind": "AMOUNT"},
    ],
    "selected_column": 0,
}
for column in SPEC["columns"]:
    column["basis"] = "CONSOLIDATED_CN_GAAP"


def test_quarter_preserves_negative_sign_and_unit():
    assert review_money_row(SPEC, PAGES)["money"][0]["cny"] == "-10.25"


def test_cumulative_selects_own_column_and_period():
    spec = {**SPEC, "period_start": "2023-01-01", "selected_column": 3}
    assert review_money_row(spec, PAGES)["money"][0]["cny"] == "100.75"


@pytest.mark.parametrize("index", [1, 3])
def test_comparative_or_cumulative_cannot_reuse_quarter_period(index):
    with pytest.raises(ValueError, match="PERIOD_MISMATCH"):
        review_money_row({**SPEC, "selected_column": index}, PAGES)


def test_percent_cannot_become_money_even_if_declared_amount():
    spec = deepcopy(SPEC)
    spec["selected_column"] = 2
    spec["columns"][2]["value_kind"] = "AMOUNT"
    with pytest.raises(ValueError, match="PERCENT_IS_NOT_MONEY"):
        review_money_row(spec, PAGES)


def test_same_period_adjusted_basis_cannot_be_substituted():
    spec = deepcopy(SPEC)
    spec["columns"][0]["basis"] = "CONSOLIDATED_CN_GAAP_RESTATED"
    with pytest.raises(ValueError, match="BASIS_MISMATCH"):
        review_money_row(spec, PAGES)


@pytest.mark.parametrize("index", [-1, 4, True])
def test_invalid_column_is_rejected(index):
    with pytest.raises(ValueError, match="OUT_OF_RANGE"):
        review_money_row({**SPEC, "selected_column": index}, PAGES)


def test_missing_cell_does_not_shift_following_values():
    row = ROW.replace("20.50", "—")
    with pytest.raises(ValueError, match="MISSING_OR_UNPARSED"):
        review_money_row({**SPEC, "row_quote": row}, [HEADER + "\n" + row])


def test_wrong_cell_sequence_rejected():
    with pytest.raises(ValueError, match="SEQUENCE_MISMATCH"):
        review_money_row({**SPEC, "cells": ["10.25", "20.50", "-150.00%", "100.75"]}, PAGES)


def test_malformed_grouping_cannot_silently_drop_commas():
    row = ROW.replace("20.50", "2,0.50")
    with pytest.raises(ValueError, match="MISSING_OR_UNPARSED"):
        review_money_row({**SPEC, "row_quote": row}, [HEADER + "\n" + row])


def test_row_unit_cannot_be_borrowed_from_another_section():
    with pytest.raises(ValueError, match="INLINE_UNIT"):
        review_money_row({**SPEC, "unit": "万元"}, PAGES)


def test_header_below_row_or_duplicate_row_is_not_accepted():
    with pytest.raises(ValueError, match="HEADER_NOT_BEFORE"):
        review_money_row(SPEC, [ROW + "\n" + HEADER])
    with pytest.raises(ValueError, match="NOT_UNIQUE"):
        review_money_row(SPEC, [PAGES[0] + "\n" + ROW])
