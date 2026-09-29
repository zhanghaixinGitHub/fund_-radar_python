"""核对重述基数、单位精度和同日顺序边界；合成原文不读取研究标签。"""

from copy import deepcopy

import pytest
from app.services.fund_earnings_dual_comparative_v1 import (
    growth_rounding_check,
    review_dual_comparative,
    same_day_order_status,
)


def fixture():
    """使用公告相同布局的合成页面，暴露列角色和单位误配。"""
    header = "项目 本报告期 上年同期 增减变动幅度 重述调整数据 法定披露数据 重述调整数据 法定披露数据"
    row = "归属于上市公司股东的净利润 210.15 211.12 115.20 -0.46% 82.43%"
    scope = "单位：人民币亿元\n" + header + "\n" + row
    audit = "本公告数据未经会计师事务所审计。"
    restatement = (
        "重述调整数据：为公司按照企业会计准则对同一控制下企业合并的要求，对2015年年度报告数据进行重述调整后的数据。"
    )
    pages = ["2016年度业绩快报公告\n" + audit + "\n" + scope, restatement]
    spec = {
        "document_id": "123",
        "issuer": "600900",
        "year": 2016,
        "published_date": "2017-01-21",
        "kind": "PRELIMINARY_RESULT",
        "page": 1,
        "title_page": 1,
        "audit_page": 1,
        "audit_quote": audit,
        "restatement_page": 2,
        "header_quote": header,
        "row_quote": row,
        "scope_quote": scope,
        "cells": ["210.15", "211.12", "115.20", "-0.46%", "82.43%"],
        "column_roles": ["CURRENT", "PRIOR_RESTATED", "PRIOR_STATUTORY", "YOY_RESTATED", "YOY_STATUTORY"],
    }
    return spec, pages


def test_keeps_two_opposite_growth_bases_and_unknown_current_standard():
    spec, pages = fixture()
    result = review_dual_comparative(spec, pages)
    assert result["reported_percent"] == {"PRIOR_RESTATED": "-0.46", "PRIOR_STATUTORY": "82.43"}
    assert result["amounts_in_source_unit"]["PRIOR_STATUTORY"] == "115.20"
    assert result["currency"] == "CNY" and result["unit"] == "亿元"
    assert not result["accounting_standard_verified"]
    assert not result["comparatives_are_independent_earlier_disclosures"]
    assert result["available_at"] == "2017-01-22T08:00:00+08:00"


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "REPORTED_RESULT"),
        ("published_date", "2016-12-30"),
        ("issuer", "6009000"),
        ("header_quote", "项目 本报告期 上年同期"),
        ("audit_quote", "已审计"),
        ("scope_quote", "单位：人民币万元"),
        ("cells", ["210.15", "115.20", "211.12", "-0.46%", "82.43%"]),
        ("column_roles", ["CURRENT", "PRIOR_STATUTORY", "PRIOR_RESTATED", "YOY_RESTATED", "YOY_STATUTORY"]),
        ("restatement_page", 1),
    ],
)
def test_rejects_changed_scope_or_meaning(field, value):
    spec, pages = fixture()
    spec[field] = value
    with pytest.raises(ValueError):
        review_dual_comparative(spec, pages)


def test_rounding_can_explain_reported_rate_but_cannot_explain_wrong_basis():
    assert growth_rounding_check("210.15", "115.20", "82.43")["is_rounding_consistency_only"]
    with pytest.raises(ValueError, match="PERCENT_INCONSISTENT"):
        growth_rounding_check("210.15", "211.12", "82.43")


@pytest.mark.parametrize(
    "current,prior,percent",
    [("-1.00", "2.00", "-150.00"), ("1.00", "0.00", "0.00"), ("1.0", "2.00", "-50.00"), ("1.00", "2.00", "-50%")],
)
def test_unsupported_numeric_shapes_stop(current, prior, percent):
    with pytest.raises(ValueError):
        growth_rounding_check(current, prior, percent)


def test_same_day_ids_never_create_chronological_change():
    spec, pages = fixture()
    a = review_dual_comparative(spec, pages)
    b = {**deepcopy(a), "document_id": "456", "kind": "FORECAST"}
    assert not same_day_order_status(a, b)["eligible_for_sequential_change"]
    b["published_date"] = "2017-01-22"
    with pytest.raises(ValueError):
        same_day_order_status(a, b)


def test_forecast_preserves_unknown_currency_and_reversed_prior_column_order():
    spec, pages = fixture()
    spec.update(
        kind="FORECAST",
        period_page=1,
        rate_page=1,
        header_quote="项目 本报告期 上年同期 法定披露数据 重述调整数据",
        row_quote="归属于上市公司股东的净利润(万元) 2,101,542.46 1,151,997.64 2,111,238.45",
        cells=["2,101,542.46", "1,151,997.64", "2,111,238.45"],
        column_roles=["CURRENT", "PRIOR_STATUTORY", "PRIOR_RESTATED"],
        reported_percent={"PRIOR_STATUTORY": "82.43", "PRIOR_RESTATED": "-0.46"},
    )
    pages[0] = "\n".join(
        [
            "2016年年度业绩预增公告",
            "2016年1月1日至2016年12月31日。",
            spec["audit_quote"],
            spec["header_quote"],
            spec["row_quote"],
            "预计2016年实现归属于上市公司股东的净利润与上年同期（法定披露数据）相比，将增加约82.43%",
            "预计2016年实现归属于上市公司股东的净利润与上年同期（重述调整数据）相比，将减少约0.46%",
        ]
    )
    fact = review_dual_comparative(spec, pages)
    assert fact["currency"] is None
    assert fact["amounts_in_source_unit"]["PRIOR_STATUTORY"] == "1,151,997.64"
    assert not fact["training_ready"]
    spec["reported_percent"]["PRIOR_RESTATED"] = "0.46"
    with pytest.raises(ValueError):
        review_dual_comparative(spec, pages)
