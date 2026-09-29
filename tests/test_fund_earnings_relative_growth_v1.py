"""同比增加额不得混入绝对利润；覆盖原文、期间、近似值和伪造字段边界。"""

import pytest
from app.services.fund_earnings_relative_growth_v1 import review_relative_growth


def fixture():
    quote = (
        "1.经本公司财务部门初步测算，预计2017年度实现归属于上市公司股东的净利润"
        "与上年同期（法定披露数据）相比，将增加97亿元左右，同比增加58%左右。"
    )
    spec = dict(
        document_id="fixture",
        issuer="600519",
        metric="NET_PROFIT_ATTRIBUTABLE_TO_PARENT",
        period_start="2017-01-01",
        period_end="2017-12-31",
        published_date="2018-01-31",
        page=1,
        quote=quote,
        unit="亿元",
        increase_amount="97",
        increase_percent="58",
        period_page=1,
        period_quote="2017年1月1日至2017年12月31日",
    )
    return spec, [spec["period_quote"] + "。\n" + quote]


def test_approximate_delta_is_never_profit_level():
    spec, pages = fixture()
    result = review_relative_growth(spec, pages)
    assert result["increase_cny"] == "9700000000"
    assert result["approximate"] and result["absolute_profit_not_established"]
    assert result["available_at"] == "2018-02-01T08:00:00+08:00"
    assert not result["eligible_for_profit_level_chain"]
    assert not {"money", "values", "kind"} & result.keys()


@pytest.mark.parametrize(
    "old,new",
    [
        ("将增加97亿元左右", "为97亿元左右"),
        ("增加97", "减少97"),
        ("97亿元左右", "97亿元"),
        ("58%左右", "58%"),
        ("97", "-97"),
        ("97", "0"),
        ("法定披露数据", "市场一致预期"),
        ("归属于上市公司股东的净利润", "营业收入"),
    ],
)
def test_unsupported_meaning_is_rejected_even_if_source_contains_it(old, new):
    spec, pages = fixture()
    spec["quote"] = spec["quote"].replace(old, new)
    pages[0] = pages[0].replace(old, new)
    with pytest.raises(ValueError):
        review_relative_growth(spec, pages)


@pytest.mark.parametrize(
    "key,value",
    [
        ("increase_amount", "264.18"),
        ("increase_percent", "57"),
        ("unit", "万元"),
        ("period_end", "2017-09-30"),
        ("period_quote", "2017年1月1日至2017年9月30日"),
        ("metric", "ADJUSTED_NET_PROFIT_ATTRIBUTABLE_TO_PARENT"),
        ("published_date", ""),
    ],
)
def test_forged_or_missing_scope_is_rejected(key, value):
    spec, pages = fixture()
    spec[key] = value
    with pytest.raises(ValueError):
        review_relative_growth(spec, pages)


def test_anchor_must_be_unique_and_present():
    spec, pages = fixture()
    for source in ["空白", pages[0] + pages[0]]:
        with pytest.raises(ValueError):
            review_relative_growth(spec, [source])


def test_adjusted_profit_remains_separate_metric():
    spec, pages = fixture()
    spec["metric"] = "ADJUSTED_NET_PROFIT_ATTRIBUTABLE_TO_PARENT"
    spec["quote"] = spec["quote"].replace("股东的净利润", "股东的扣除非经常性损益的净利润")
    pages[0] = pages[0].replace("股东的净利润", "股东的扣除非经常性损益的净利润")
    assert review_relative_growth(spec, pages)["metric"] == spec["metric"]
