"""防止快报阶段、金额单位和前期比较值在后续处理时被误改。"""

from copy import deepcopy

import pytest
from app.services.fund_earnings_preliminary_v1 import review_preliminary
from tests.test_fund_earnings_scoped_table_v1 import fixture as scoped_fixture


def fixture():
    spec, _ = scoped_fixture()
    spec.update(
        {
            "period_start": "2022-01-01",
            "period_end": "2022-12-31",
            "kind": "PRELIMINARY_RESULT",
            "basis": "CONSOLIDATED_STANDARD_UNSPECIFIED",
            "source_title": "2022年度业绩快报",
            "title_page": 1,
            "audit_quote": "已经内部审计部门审计，未经会计师事务所审计",
            "audit_page": 1,
            "basis_quote": "表内数据为合并报表数据",
            "basis_page": 1,
            "unit": "亿元",
            "unit_quote": "单位：亿元",
            "row_quote": "归属于上市公司股东的净利润 58.28 60.05 -2.95%",
            "cells": ["58.28", "60.05", "-2.95%"],
        }
    )
    spec["columns"][0].update({k: spec[k] for k in ("period_start", "period_end", "basis")})
    spec["scope_quote"] = "\n".join([spec["unit_quote"], spec["header_quote"], spec["row_quote"]])
    pages = ["\n".join([spec["source_title"], spec["audit_quote"], spec["basis_quote"], spec["scope_quote"]])]
    return spec, pages


def test_preserves_preliminary_status_coarse_precision_and_comparative():
    spec, pages = fixture()
    result = review_preliminary(spec, pages)
    assert result["money"][0]["cny"] == "5828000000.00"
    assert result["reported_amount_resolution_cny"] == "1000000.00"
    assert result["kind"] == "PRELIMINARY_RESULT"
    assert not result["accounting_standard_verified"]
    assert not result["comparative_is_independent_earlier_disclosure"]
    assert result["available_at"] == "2023-04-29T08:00:00+08:00"


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "REPORTED_RESULT"),
        ("source_title", "2022年年度报告"),
        ("source_title", "2021年度业绩快报"),
        ("period_end", "2022-09-30"),
        ("published_date", "2022-12-31"),
        ("basis", "CONSOLIDATED_CN_GAAP"),
        ("metric", "ADJUSTED_NET_PROFIT_ATTRIBUTABLE_TO_PARENT"),
        ("selected_column", 1),
        ("audit_quote", "已经内部审计部门审计"),
        ("basis_quote", "母公司报表数据"),
    ],
)
def test_misleading_metadata_rejected(field, value):
    spec, pages = fixture()
    spec[field] = value
    with pytest.raises(ValueError):
        review_preliminary(spec, pages)


def test_unaudited_text_must_exist_in_original():
    spec, pages = fixture()
    with pytest.raises(ValueError):
        review_preliminary(spec, [pages[0].replace(spec["audit_quote"], "缺失")])


def test_does_not_mutate_frozen_spec():
    spec, pages = fixture()
    before = deepcopy(spec)
    review_preliminary(spec, pages)
    assert spec == before
