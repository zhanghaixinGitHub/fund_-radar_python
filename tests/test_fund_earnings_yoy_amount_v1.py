"""防止增加额当利润、千元错当元、跨会计准则和提前使用比较数。"""

import pytest
from app.services.fund_earnings_yoy_amount_v1 import compare_increases, review_increase_range, review_reported_increase


def fixtures():
    base = dict(
        issuer="600585",
        metric="NET_PROFIT_ATTRIBUTABLE_TO_PARENT",
        period_start="2018-01-01",
        period_end="2018-12-31",
        basis="CONSOLIDATED_CN_GAAP",
        currency="CNY",
    )
    f = {
        **base,
        "document_id": "forecast",
        "published_date": "2019-01-11",
        "page": 1,
        "quote": (
            "1、经本公司财务部门初步测算，预计2018年年度实现归属于上市公司股东的净利润"
            "与上年同期（法定披露数据）相比将增加126.84亿元人民币～158.55亿元人民币，同比增加80%～100%。"
        ),
        "values": ["126.84", "158.55", "80", "100"],
        "period_page": 1,
        "period_quote": "2018年1月1日至2018年12月31日",
        "basis_page": 1,
        "basis_quote": "公司按照中国企业会计准则对2018年度经营业绩进行测算",
        "audit_page": 2,
        "audit_quote": "本次所预计的业绩未经注册会计师审计。",
    }
    fpages = [f["basis_quote"] + "。" + f["period_quote"] + "。" + f["quote"], f["audit_quote"]]
    row = "归属于上市公司股东的净利润 29,814,285 15,854,670 88.05 8,529,917"
    scope = (
        "（十一）按中国会计准则编制的会计数据\n（单位：千元）\n项目 2018年 2017年 本年比上年增减（％） 2016年\n" + row
    )
    r = {
        **base,
        "document_id": "annual",
        "published_date": "2019-03-22",
        "first_page": 1,
        "last_page": 1,
        "scope_quote": scope,
        "row_quote": row,
        "cells": ["29,814,285", "15,854,670", "88.05", "8,529,917"],
        "audit_page": 2,
        "audit_quote": "按照企业会计准则的规定编制",
    }
    return f, fpages, r, [scope, r["audit_quote"]]


def test_correct_comparison_preserves_units_roles_and_dates():
    f, fp, r, rp = fixtures()
    a, b = review_increase_range(f, fp), review_reported_increase(r, rp)
    assert b["increase_cny"] == "13959615000"
    assert b["previous_comparative_cny"] == "15854670000"
    assert b["comparative_is_not_independent_prior_disclosure"]
    assert not b["eligible_for_absolute_profit_chain"]
    result = compare_increases(a, b, "2019-03-23T08:00:00+08:00")
    assert result["relation"] == "WITHIN_FORECAST_INCREASE_RANGE"
    assert not result["training_ready"]


@pytest.mark.parametrize(
    "old,new",
    [
        ("相比将增加", "相比为"),
        ("增加126.84", "增加158.55"),
        ("亿元人民币", "万元人民币"),
        ("同比增加80%～100%", "同比增加100%～80%"),
        ("中国企业会计准则", "国际财务报告准则"),
    ],
)
def test_forecast_changed_meaning_rejected(old, new):
    f, fp, _, _ = fixtures()
    for key in ["quote", "basis_quote"]:
        f[key] = f[key].replace(old, new)
    fp = [p.replace(old, new) for p in fp]
    with pytest.raises(ValueError):
        review_increase_range(f, fp)


@pytest.mark.parametrize(
    "old,new",
    [
        ("单位：千元", "单位：元"),
        ("中国会计准则", "国际财务报告准则"),
        ("2017年", "2016年"),
        ("88.05", "88.06"),
        ("15,854,670", "0"),
        ("29,814,285", "29,858,303"),
    ],
)
def test_wrong_unit_basis_columns_or_inconsistent_percent_rejected(old, new):
    _, _, r, rp = fixtures()
    r["scope_quote"] = r["scope_quote"].replace(old, new)
    r["row_quote"] = r["row_quote"].replace(old, new)
    r["cells"] = [c.replace(old, new) for c in r["cells"]]
    rp = [p.replace(old, new) for p in rp]
    with pytest.raises(ValueError):
        review_reported_increase(r, rp)


@pytest.mark.parametrize("asof", ["2019-03-23T07:59:59+08:00", "2018-12-31T08:00:00+08:00", "2019-03-23T08:00:00"])
def test_future_report_cannot_change_earlier_snapshot(asof):
    f, fp, r, rp = fixtures()
    with pytest.raises(ValueError):
        compare_increases(review_increase_range(f, fp), review_reported_increase(r, rp), asof)


@pytest.mark.parametrize(
    "key,value",
    [
        ("issuer", "600519"),
        ("basis", "IFRS"),
        ("metric", "OTHER"),
        ("period_end", "2019-12-31"),
        ("measure", "ABSOLUTE_PROFIT"),
    ],
)
def test_incompatible_comparison_rejected(key, value):
    f, fp, r, rp = fixtures()
    a, b = review_increase_range(f, fp), review_reported_increase(r, rp)
    b[key] = value
    with pytest.raises(ValueError):
        compare_increases(a, b, "2019-03-23T08:00:00+08:00")


def test_adjacent_unit_boundary_and_repeated_anchors_rejected():
    f, fp, r, rp = fixtures()
    with pytest.raises(ValueError):
        review_increase_range(f, [fp[0] + fp[0], fp[1]])
    r["scope_quote"] = r["scope_quote"].replace("项目", "（单位：元）项目")
    rp[0] = r["scope_quote"]
    with pytest.raises(ValueError):
        review_reported_increase(r, rp)


def test_comparative_not_registered_as_old_disclosure():
    _, _, r, rp = fixtures()
    b = review_reported_increase(r, rp)
    assert b["available_at"] == "2019-03-23T08:00:00+08:00"
    assert not {"money", "values", "kind"} & b.keys()
