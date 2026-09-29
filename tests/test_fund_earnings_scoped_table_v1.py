"""检验表格单位串表、缺列、负号及不适用值被误当金额的风险。"""

from copy import deepcopy

import pytest
from app.services.fund_earnings_scoped_table_v1 import review_scoped_money_row


def fixture():
    unit = "单位：元 币种：人民币"
    header = "项目 本报告期 上年同期 增减(%)"
    row = "归属于上市公司股东的净利润 77,190,888.11 -232,376,308.45 不适用"
    scope = f"{unit}\n{header}\n{row}"
    spec = {
        "issuer": "600258",
        "period_start": "2023-01-01",
        "period_end": "2023-03-31",
        "metric": "NET_PROFIT_ATTRIBUTABLE_TO_PARENT",
        "basis": "CONSOLIDATED_CN_GAAP",
        "currency": "CNY",
        "unit": "元",
        "kind": "REPORTED_RESULT",
        "published_date": "2023-04-28",
        "document_id": "example",
        "page": 1,
        "scope_quote": scope,
        "unit_quote": unit,
        "header_quote": header,
        "row_quote": row,
        "row_label": "归属于上市公司股东的净利润",
        "cells": ["77190888.11", "-232376308.45", None],
        "selected_column": 0,
        "columns": [
            {
                "label": "本报告期",
                "value_kind": "AMOUNT",
                "period_start": "2023-01-01",
                "period_end": "2023-03-31",
                "basis": "CONSOLIDATED_CN_GAAP",
            },
            {
                "label": "上年同期",
                "value_kind": "AMOUNT",
                "period_start": "2022-01-01",
                "period_end": "2022-03-31",
                "basis": "CONSOLIDATED_CN_GAAP",
            },
            {"label": "增减(%)", "value_kind": "PERCENT"},
        ],
    }
    return spec, [scope]


def test_preserves_negative_comparative_and_na_without_zero_fill():
    spec, pages = fixture()
    fact = review_scoped_money_row(spec, pages)
    assert fact["money"][0]["cny"] == "77190888.11"
    assert fact["column_proof"]["all_cells"] == ["77190888.11", "-232376308.45", None]
    assert fact["available_at"] == "2023-04-29T08:00:00+08:00"


@pytest.mark.parametrize("index", [2, -1, 3, True])
def test_na_and_invalid_column_rejected(index):
    spec, pages = fixture()
    spec["selected_column"] = index
    with pytest.raises(ValueError):
        review_scoped_money_row(spec, pages)


def test_wrong_period_does_not_select_comparative():
    spec, pages = fixture()
    spec["selected_column"] = 1
    with pytest.raises(ValueError, match="PERIOD_OR_BASIS"):
        review_scoped_money_row(spec, pages)


@pytest.mark.parametrize("field,value", [("unit", "万元"), ("basis", "RESTATED"), ("page", 2)])
def test_wrong_scope_rejected(field, value):
    spec, pages = fixture()
    spec[field] = value
    with pytest.raises(ValueError):
        review_scoped_money_row(spec, pages)


@pytest.mark.parametrize("old,new", [("不适用", "--"), ("77,190,888.11", "7,7190,888.11"), (" -232", "-232")])
def test_unparsed_or_missing_separator_rejected(old, new):
    spec, pages = fixture()
    spec["row_quote"] = spec["row_quote"].replace(old, new)
    spec["scope_quote"] = pages[0].replace(old, new)
    with pytest.raises(ValueError):
        review_scoped_money_row(spec, [spec["scope_quote"]])


def test_extra_table_unit_rejected():
    spec, pages = fixture()
    spec["scope_quote"] = pages[0].replace(spec["header_quote"], spec["header_quote"] + "\n单位：万元")
    with pytest.raises(ValueError, match="UNIT_BOUNDARY"):
        review_scoped_money_row(spec, [spec["scope_quote"]])


def test_unit_on_previous_page_cannot_be_borrowed():
    spec, pages = fixture()
    with pytest.raises(ValueError, match="ANCHOR_NOT_UNIQUE"):
        review_scoped_money_row(spec, [pages[0].replace(spec["unit_quote"], ""), spec["unit_quote"]])


def test_duplicate_scope_and_dropped_cell_rejected():
    spec, pages = fixture()
    with pytest.raises(ValueError, match="ANCHOR_NOT_UNIQUE"):
        review_scoped_money_row(spec, [pages[0] + pages[0]])
    spec["cells"] = spec["cells"][:2]
    with pytest.raises(ValueError, match="CELL_SEQUENCE"):
        review_scoped_money_row(spec, pages)


def test_bare_percent_column_is_not_money():
    spec, pages = fixture()
    changed = deepcopy(spec)
    changed["row_quote"] = changed["row_quote"].replace("不适用", "7.40")
    changed["scope_quote"] = pages[0].replace("不适用", "7.40")
    changed["cells"][2] = "7.40"
    changed["selected_column"] = 2
    with pytest.raises(ValueError, match="NOT_MONEY"):
        review_scoped_money_row(changed, [changed["scope_quote"]])
