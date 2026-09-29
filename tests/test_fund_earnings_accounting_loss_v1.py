"""明确净亏损、会计括号、百万元和多期间列的边界验证。"""

import pytest
from app.services.fund_earnings_accounting_loss_v1 import review_explicit_profit_range, review_parenthesized_loss_row


def fixture():
    unit = "单位：百万元 币种：人民币"
    header = "项目 2022年 2021年 增加/减少(%) 2020年"
    row = "归属于上市公司股东的净亏损 (32,682) (12,103) 170.03 (10,842)"
    spec = dict(
        issuer="600029",
        period_start="2022-01-01",
        period_end="2022-12-31",
        metric="NET_PROFIT_ATTRIBUTABLE_TO_PARENT",
        basis="CONSOLIDATED_CN_GAAP",
        currency="CNY",
        unit="百万元",
        kind="REPORTED_RESULT",
        published_date="2023-03-29",
        document_id="example",
        page=1,
        unit_quote=unit,
        header_quote=header,
        row_quote=row,
        row_label="归属于上市公司股东的净亏损",
        cells=["-32682", "-12103", "170.03", "-10842"],
        selected_column=0,
        columns=[
            dict(
                label="2022年",
                value_kind="AMOUNT",
                period_start="2022-01-01",
                period_end="2022-12-31",
                basis="CONSOLIDATED_CN_GAAP",
            ),
            dict(
                label="2021年",
                value_kind="AMOUNT",
                period_start="2021-01-01",
                period_end="2021-12-31",
                basis="CONSOLIDATED_CN_GAAP",
            ),
            dict(label="增加/减少(%)", value_kind="PERCENT"),
            dict(
                label="2020年",
                value_kind="AMOUNT",
                period_start="2020-01-01",
                period_end="2020-12-31",
                basis="CONSOLIDATED_CN_GAAP",
            ),
        ],
    )
    spec["scope_quote"] = "\n".join([unit, header, row])
    return spec, [spec["scope_quote"]]


def test_loss_sign_million_unit_original_tokens_and_precision_retained():
    spec, pages = fixture()
    result = review_parenthesized_loss_row(spec, pages)
    assert result["money"][0]["cny"] == "-32682000000"
    assert result["reported_magnitudes"][0]["value"] == "32682"
    assert result["column_proof"]["original_cell_tokens"][0] == "(32,682)"
    assert result["available_at"] == "2023-03-30T08:00:00+08:00"
    assert not result["training_ready"]


@pytest.mark.parametrize(
    "changed", ["32,682", "-32,682", "(-32,682)", "(32,682%）", "（32,682)", "(32,682", "(3,2682)", "不适用", "--"]
)
def test_unproven_or_malformed_parentheses_rejected(changed):
    spec, pages = fixture()
    spec["row_quote"] = spec["row_quote"].replace("(32,682)", changed)
    spec["scope_quote"] = pages[0].replace("(32,682)", changed)
    with pytest.raises(ValueError):
        review_parenthesized_loss_row(spec, [spec["scope_quote"]])


@pytest.mark.parametrize(
    "key,value", [("selected_column", 2), ("selected_column", 1), ("unit", "万元"), ("basis", "IFRS"), ("page", 2)]
)
def test_percent_comparative_unit_and_basis_rejected(key, value):
    spec, pages = fixture()
    spec[key] = value
    with pytest.raises(ValueError):
        review_parenthesized_loss_row(spec, pages)


def test_ordinary_profit_row_cannot_assume_parentheses_are_negative():
    spec, pages = fixture()
    for key in ["row_label", "row_quote", "scope_quote"]:
        spec[key] = spec[key].replace("净亏损", "净利润")
    with pytest.raises(ValueError, match="NET_LOSS_LABEL"):
        review_parenthesized_loss_row(spec, [spec["scope_quote"]])


def test_extra_unit_and_dropped_cell_rejected():
    spec, pages = fixture()
    spec["cells"].pop()
    with pytest.raises(ValueError, match="CELL_SEQUENCE"):
        review_parenthesized_loss_row(spec, pages)
    spec, pages = fixture()
    spec["scope_quote"] = pages[0].replace(spec["row_quote"], "单位：万元\n" + spec["row_quote"])
    with pytest.raises(ValueError, match="UNIT_BOUNDARY"):
        review_parenthesized_loss_row(spec, [spec["scope_quote"]])


def test_unknown_cell_preserves_position_and_cannot_be_selected():
    spec, pages = fixture()
    spec["row_quote"] = spec["row_quote"].replace("170.03", "不适用")
    spec["scope_quote"] = pages[0].replace("170.03", "不适用")
    spec["cells"][2] = None
    fact = review_parenthesized_loss_row(spec, [spec["scope_quote"]])
    assert fact["column_proof"]["all_cells"][2] is None
    spec["selected_column"] = 2
    with pytest.raises(ValueError):
        review_parenthesized_loss_row(spec, [spec["scope_quote"]])


def test_single_hyphen_profit_range_keeps_both_endpoints_positive():
    spec, _ = fixture()
    spec.update(
        kind="FORECAST", unit="万元", quote="盈利：197,339.91 万元-222,339.91 万元", values=["197339.91", "222339.91"]
    )
    fact = review_explicit_profit_range(spec, [spec["quote"]])
    assert [v["cny"] for v in fact["money"]] == ["1973399100.00", "2223399100.00"]
    assert fact["anchor"]["text"] == "盈利：197,339.91万元-222,339.91万元"


@pytest.mark.parametrize(
    "quote",
    [
        "盈利：100万元--200万元",
        "盈利：100万元到-200万元",
        "亏损：100万元-200万元",
        "盈利：100万元-200亿元",
        "盈利：200万元-100万元",
        "盈利：100万元-201万元",
        "盈利：1,00万元-200万元",
        "预计100万元-200万元",
    ],
)
def test_profit_polarity_unit_endpoint_and_number_conflicts_rejected(quote):
    spec, _ = fixture()
    spec.update(kind="FORECAST", unit="万元", quote=quote, values=["100", "200"])
    with pytest.raises(ValueError):
        review_explicit_profit_range(spec, [quote])
