"""窗口与语义边界测试，避免重复取数或把同比百分比换成金额/百分点。"""

import pytest
from app.services.fund_earnings_history_v2 import identity, merge_windows, review_yoy


def window(stock="000001", lo="2022-01-01", hi="2022-01-15", fund="A"):
    return {
        "stock": stock,
        "start": lo,
        "end": hi,
        "fund_code": fund,
        "report_sha256": "report" + fund,
        "target_records": 10,
    }


def yoy(**changes):
    value = {
        "comparison": "YEAR_ON_YEAR",
        "unit": "PERCENT",
        "metric": "PARENT_NET_PROFIT",
        "basis": "CONSOLIDATED_PARENT",
        "period_start": "2022-01-01",
        "period_end": "2022-06-30",
        "direction_word": "增长",
        "base_state": "POSITIVE",
        "page": 1,
        "quote": "净利润比上年同期增长76.15%",
        "values": ["76.15"],
    }
    value.update(changes)
    return review_yoy(value, [value["quote"]])


def test_shared_window_deduplicates_request_not_fund_provenance():
    result = merge_windows([window(), window(fund="B")])
    assert len(result) == 1
    assert len(result[0]["sources"]) == 2


def test_adjacent_merge_preserves_dates_and_distinct_company():
    values = [window(), window(lo="2022-01-16", hi="2022-01-30"), window(stock="000002")]
    result = merge_windows(values)
    assert len(result) == 2 and result[0]["end"] == "2022-01-30"


def test_one_day_gap_is_not_filled_by_merge():
    assert len(merge_windows([window(), window(lo="2022-01-17", hi="2022-01-30")])) == 2


def test_chinese_year_and_quarter_alias_not_fuzzy():
    row = {"secCode": "000001", "title_plain": "平安银行2022年一季度报告"}
    assert identity(row, ["证券代码：000001平安银行2022年第一季度报告"])["passed"]
    assert not identity(row, ["证券代码：000001平安银行2022年第三季度报告"])["passed"]
    row["title_plain"] = "2021年度报告"
    assert identity(row, ["证券代码：000001二〇二一年度报告"])["passed"]


def test_wrong_company_or_summary_not_equivalent():
    row = {"secCode": "000001", "title_plain": "2021年度报告摘要"}
    assert not identity(row, ["证券代码：0000022021年度报告摘要"])["passed"]
    assert not identity(row, ["证券代码：0000012021年度报告"])["passed"]


def test_correction_marker_not_silently_discarded():
    row = {"secCode": "000001", "title_plain": "2021年度报告（更正后）"}
    assert not identity(row, ["证券代码：0000012021年度报告"])["passed"]


@pytest.mark.parametrize("state", ["NEGATIVE", "ZERO", "UNKNOWN"])
def test_nonpositive_or_unknown_base_has_no_inferred_signed_growth(state):
    assert yoy(base_state=state)["signed_percent"] is None


def test_decline_remains_decline_and_not_money():
    result = yoy(direction_word="下降", quote="净利润同比下降30.5%", values=["30.5"])
    assert result["signed_percent"] == ["-30.5"]
    assert not result["derived_from_money"]


def test_yoy_must_be_explicit_percent():
    with pytest.raises(ValueError, match="ANCHOR_INVALID"):
        yoy(quote="净利润增长76.15万元")
    with pytest.raises(ValueError, match="PERCENTAGE_POINT"):
        yoy(quote="净利润增长76.15%百分点")


def test_wrong_comparison_or_amount_not_accepted():
    with pytest.raises(ValueError, match="NOT_YEAR"):
        yoy(comparison="QUARTER_ON_QUARTER")
    with pytest.raises(ValueError, match="VALUE_NOT_IN_ANCHOR"):
        yoy(values=["76.1"])
