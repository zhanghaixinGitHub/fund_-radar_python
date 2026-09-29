"""合成原页验证准则、指标、单位、当前列和可用时间边界；不拟合模型。"""

from copy import deepcopy

import pytest
from app.services.fund_earnings_annual_forecast_v1 import compare_verified_claims
from app.services.fund_earnings_cas_parent_v1 import BASIS, METRIC, review_cas_parent


def examples():
    base = dict(
        issuer="600938",
        year=2022,
        period_start="2022-01-01",
        period_end="2022-12-31",
        metric=METRIC,
        basis=BASIS,
        currency="CNY",
        page=1,
    )
    forecast = dict(
        base,
        kind="FORECAST",
        document_id="old",
        published_date="2023-01-20",
        unit="亿元",
        period_page=1,
        values=["1,396", "1,436"],
        quote="经本公司财务部门初步测算，按照中国企业会计准则，预计2022年全年度实现归属于母公司股东的净利润为人民币1,396亿元到1,436亿元",
    )
    actual = dict(
        base,
        kind="REPORTED_RESULT",
        document_id="new",
        published_date="2023-03-30",
        unit="百万元",
        values=["141,700"],
        table_cells=["141,700", "70,320"],
        quote="人民币百万元\n归属于母公司股东的净利润\n2022年度 2021年度\n按中国企业会计准则 141,700 70,320",
    )
    pages = [forecast["quote"] + "。2022年1月1日到2022年12月31日。本次业绩预告相关的财务数据未经会计师事务所审计。"]
    return forecast, pages, actual, [actual["quote"]]


def test_explicit_cas_parent_pair_and_units():
    f, fp, a, ap = examples()
    older, newer = [review_cas_parent(s, p) for s, p in ((f, fp), (a, ap))]
    assert [x["cny"] for x in older["money"]] == ["139600000000", "143600000000"]
    assert newer["money"][0]["cny"] == "141700000000"
    for c in (older, newer):
        c.update(source_identity_verified=True, revision_issues=[])
        assert not c["unchanged_consolidated_entity_population_verified"]
    result = compare_verified_claims(older, newer, newer["available_at"])
    assert result["relation"] == "OVERLAPS_PREVIOUS_RANGE"
    assert result["midpoint_change_cny"] == "100000000"
    assert result["percent_change"] is None
    with pytest.raises(ValueError, match="CLAIM_NOT_AVAILABLE"):
        compare_verified_claims(older, newer, "2023-03-31T07:59:59+08:00")


@pytest.mark.parametrize(
    "field,value",
    [
        ("year", 2021),
        ("period_end", "2022-09-30"),
        ("metric", "NET_ASSETS"),
        ("basis", "IFRS"),
        ("currency", "USD"),
        ("unit", "万元"),
        ("kind", "PRELIMINARY_RESULT"),
        ("values", ["693", "733"]),
    ],
)
def test_forecast_rejects_changed_scope(field, value):
    f, fp, _, _ = examples()
    f[field] = value
    with pytest.raises(ValueError):
        review_cas_parent(f, fp)


@pytest.mark.parametrize(
    "old,new",
    [
        ("按照中国企业会计准则", "按照国际财务报告准则"),
        ("归属于母公司股东的净利润", "母公司净利润"),
        ("归属于母公司股东的净利润", "归属于母公司股东的净资产"),
        ("净利润为", "净利润增加额为"),
    ],
)
def test_forecast_requires_explicit_wording_even_when_source_matches(old, new):
    f, fp, _, _ = examples()
    f["quote"] = f["quote"].replace(old, new)
    fp = [p.replace(old, new) for p in fp]
    with pytest.raises(ValueError):
        review_cas_parent(f, fp)


@pytest.mark.parametrize(
    "change", ["prior_column", "wrong_cells", "wrong_unit", "ifrs_row", "net_assets", "reversed_years"]
)
def test_actual_table_rejects_wrong_selection(change):
    _, _, a, ap = examples()
    a = deepcopy(a)
    if change == "prior_column":
        a["values"] = ["70,320"]
    elif change == "wrong_cells":
        a["table_cells"] = ["141,700", "70,321"]
    elif change == "wrong_unit":
        a["unit"] = "亿元"
    else:
        old, new = {
            "ifrs_row": ("按中国企业会计准则", "按国际财务报告准则"),
            "net_assets": ("净利润", "净资产"),
            "reversed_years": ("2022年度 2021年度", "2021年度 2022年度"),
        }[change]
        a["quote"] = a["quote"].replace(old, new)
        ap = [p.replace(old, new) for p in ap]
    with pytest.raises(ValueError):
        review_cas_parent(a, ap)


def test_duplicate_anchor_and_missing_period_rejected():
    f, fp, a, ap = examples()
    with pytest.raises(ValueError):
        review_cas_parent(a, [ap[0] * 2])
    with pytest.raises(ValueError):
        review_cas_parent(f, [fp[0].replace("2022年1月1日到2022年12月31日", "")])
