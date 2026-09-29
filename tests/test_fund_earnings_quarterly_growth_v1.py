"""同比百分比不能冒充利润水平；跨期、更正列、未来信息不能混入变化证据。"""

from copy import deepcopy

import pytest
from app.services.fund_earnings_quarterly_growth_v1 import (
    observe_same_period,
    review_percentage_forecast,
    review_q1_report,
)


def inputs():
    base = dict(issuer="601006", period_start="2017-01-01", period_end="2017-03-31", page=1)
    f = dict(
        base,
        document_id="forecast",
        published_date="2017-04-08",
        percent="45",
        prior_value="2,054,467,566",
        period_quote="2017年1月1日至2017年3月31日。",
        quote="经财务部门初步测算，本公司预计2017年第一季度归属于上市公司股东的净利润与上年同期相比，增长45%左右。",
        prior_quote="二、上年同期业绩情况1、归属于上市公司股东的净利润：2,054,467,566元",
    )
    fp = ["\n".join([f["period_quote"], f["quote"], f["prior_quote"], "本次所预计的业绩未经注册会计师审计。"])]
    row = "归属于上市公司股东的净利润 3,161,200,616 2,054,467,566 53.87"
    r = dict(
        base,
        document_id="report",
        published_date="2017-04-27",
        title_page=1,
        title_quote="2017年第一季度报告",
        row_quote=row,
        scope_quote="2.1主要财务数据单位：元币种：人民币\n年初至报告期末 上年初至上年报告期末 比上年同期增减（%）\n"
        + row,
        cells=["3,161,200,616", "2,054,467,566", "53.87"],
    )
    rp = [r["title_quote"] + "\n" + r["scope_quote"] + "\n本公司第一季度报告未经审计。"]
    return f, fp, r, rp


def test_observed_pair_remains_unqualified_without_basis_and_range():
    f, fp, r, rp = inputs()
    a, b = review_percentage_forecast(f, fp), review_q1_report(r, rp)
    assert a["currency"] is None and not a["current_profit_derived"]
    assert b["current_cny"] == "3161200616" and b["audit_status"] == "UNAUDITED"
    observed = observe_same_period(a, b, b["available_at"])
    assert observed["difference_from_stated_number_percentage_points"] == "8.87"
    assert not observed["qualified_numeric_comparison"] and not observed["range_exceedance_established"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("period_end", "2017-06-30"),
        ("percent", "90"),
        ("prior_value", "3,680,270,243"),
        ("published_date", "2017-03-31"),
    ],
)
def test_forecast_period_rate_and_previous_value_bound(field, value):
    f, fp, _, _ = inputs()
    f[field] = value
    with pytest.raises(ValueError):
        review_percentage_forecast(f, fp)


@pytest.mark.parametrize(
    "before,after",
    [("左右", ""), ("净利润", "净资产"), ("增长", "减少"), ("未经", "已经"), ("上年同期业绩情况", "本期业绩情况")],
)
def test_forecast_semantics_not_relaxed(before, after):
    f, fp, _, _ = inputs()
    f = {k: v.replace(before, after) if isinstance(v, str) else v for k, v in f.items()}
    fp = [p.replace(before, after) for p in fp]
    with pytest.raises(ValueError):
        review_percentage_forecast(f, fp)


@pytest.mark.parametrize("change", ["unit", "currency", "column_order", "metric", "percent", "duplicate"])
def test_report_unit_scope_column_and_arithmetic(change):
    _, _, r, rp = inputs()
    before, after = {
        "unit": ("单位：元", "单位：万元"),
        "currency": ("人民币", "美元"),
        "column_order": ("年初至报告期末 上年初至上年报告期末", "上年初至上年报告期末 年初至报告期末"),
        "metric": ("净利润", "净资产"),
        "percent": ("53.87", "53.88"),
        "duplicate": ("", ""),
    }[change]
    if change == "duplicate":
        rp[0] += "\n" + r["scope_quote"]
    else:
        r = {k: v.replace(before, after) if isinstance(v, str) else v for k, v in r.items()}
        if change == "percent":
            r["cells"][-1] = "53.88"
        rp = [p.replace(before, after) for p in rp]
    with pytest.raises(ValueError):
        review_q1_report(r, rp)


@pytest.mark.parametrize("change", ["future", "same_day", "half_year", "different_comparative"])
def test_observation_cannot_cross_period_time_or_comparative(change):
    f, fp, r, rp = inputs()
    a, b = review_percentage_forecast(f, fp), review_q1_report(r, rp)
    b = deepcopy(b)
    at = b["available_at"]
    if change == "future":
        at = "2017-04-28T07:59:59+08:00"
    elif change == "same_day":
        b["available_at"] = a["available_at"]
    elif change == "half_year":
        a["period_end"] = "2017-06-30"
    else:
        b["previous_comparative_original"] = "3,680,270,243"
    with pytest.raises(ValueError):
        observe_same_period(a, b, at)
