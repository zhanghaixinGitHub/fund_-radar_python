"""半年准则事实的负例：错期间、准则、列、单位和审阅状态均不得通过。"""

from copy import deepcopy

import pytest
from app.services.fund_earnings_annual_forecast_v1 import compare_verified_claims
from app.services.fund_earnings_cas_interim_v1 import review_cas_interim
from app.services.fund_earnings_cas_parent_v1 import BASIS, METRIC


def fixtures():
    """合成两份有明确期间与准则的独立披露；不读取基金标签或网络。"""
    quote = (
        "按照中国企业会计准则，中国海洋石油有限公司（以下简称“公司”或“本公司”）"
        "2022年中期归属于母公司股东的净利润预计为人民币705亿元到725亿元"
    )
    common = dict(
        year=2022,
        issuer="600938",
        period_start="2022-01-01",
        period_end="2022-06-30",
        metric=METRIC,
        basis=BASIS,
        currency="CNY",
        page=1,
    )
    a = dict(
        common,
        document_id="old",
        published_date="2022-07-15",
        kind="FORECAST",
        unit="亿元",
        values=["705", "725"],
        legal_name="中国海洋石油有限公司",
        quote=quote,
        period_page=1,
        audit_page=1,
    )
    ap = [quote + "\n2022年1月1日到2022年6月30日\n本次预计的业绩未经审计。"]
    table = (
        "人民币百万元\n归属于母公司股东的净利润\n截至2022年6月30日止6个月期间\n"
        "截至2021年6月30日止6个月期间\n按中国企业会计准则 71,887 33,329"
    )
    b = dict(
        common,
        document_id="new",
        published_date="2022-08-26",
        kind="REPORTED_RESULT",
        unit="百万元",
        values=["71,887"],
        table_cells=["71,887", "33,329"],
        quote=table,
        audit_page=2,
        review_page=2,
    )
    bp = [
        table,
        "本半年度报告中的财务报告未经审计。\n自2022年1月1日至6月30日止期间的合并及公司利润表、"
        "股东权益变动表和现金流量表\n我们没有实施审计，因而不发表审计意见。",
    ]
    return a, ap, b, bp


def test_interim_same_basis_comparison_preserves_review_not_audit():
    a, ap, b, bp = fixtures()
    old, new = [review_cas_interim(s, p) for s, p in ((a, ap), (b, bp))]
    assert new["audit_status"] == "INTERIM_REVIEWED_NOT_AUDITED"
    assert new["money"][0]["cny"] == "71887000000"
    assert not new["comparative_is_independent_earlier_disclosure"]
    for f in (old, new):
        f.update(source_identity_verified=True, revision_issues=[])
    result = compare_verified_claims(old, new, new["available_at"])
    assert result["relation"] == "OVERLAPS_PREVIOUS_RANGE"
    assert not result["is_market_consensus_surprise"]
    with pytest.raises(ValueError, match="CLAIM_NOT_AVAILABLE"):
        compare_verified_claims(old, new, "2022-08-27T07:59:59+08:00")


@pytest.mark.parametrize(
    "field,value",
    [
        ("period_end", "2022-12-31"),
        ("period_start", "2022-04-01"),
        ("currency", "HKD"),
        ("metric", "NET_ASSETS"),
        ("basis", "IFRS"),
        ("unit", "万元"),
        ("values", ["33,329"]),
        ("table_cells", ["33,329", "71,887"]),
        ("kind", "PRELIMINARY_RESULT"),
    ],
)
def test_rejects_wrong_scope_or_column(field, value):
    _, _, spec, pages = fixtures()
    spec[field] = value
    with pytest.raises(ValueError):
        review_cas_interim(spec, pages)


@pytest.mark.parametrize(
    "old,new",
    [
        ("按中国企业会计准则", "按国际/香港财务报告准则"),
        ("归属于母公司股东的净利润", "归属于母公司股东的净资产"),
        ("止6个月期间", "止3个月期间"),
        ("我们没有实施审计，因而不发表审计意见。", "已经审计。"),
    ],
)
def test_changed_source_text_cannot_pass_even_if_spec_is_changed(old, new):
    _, _, spec, pages = fixtures()
    spec["quote"] = spec["quote"].replace(old, new)
    pages = [p.replace(old, new) for p in pages]
    with pytest.raises(ValueError):
        review_cas_interim(spec, pages)


def test_forecast_cannot_be_increase_amount_or_drop_rmb_or_reverse_range():
    spec, pages, _, _ = fixtures()
    for replacement in ("同比增加额", "扣除非经常性损益后的净利润"):
        bad = deepcopy(spec)
        bad["quote"] = bad["quote"].replace("净利润", replacement)
        with pytest.raises(ValueError):
            review_cas_interim(bad, [pages[0].replace(spec["quote"], bad["quote"])])
    spec["values"] = ["725", "705"]
    with pytest.raises(ValueError):
        review_cas_interim(spec, pages)
