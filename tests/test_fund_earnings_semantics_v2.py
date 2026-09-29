"""覆盖真实遗漏的预增/预减标题及财报表头口径，防止金额和同比列错位。"""

import pytest
from app.services.fund_earnings_semantics_v2 import earnings_title_kind, review_table_yoy


@pytest.mark.parametrize("word", ["预增", "预减", "预盈", "预亏", "扭亏"])
def test_forecast_aliases_have_no_stock_direction(word):
    assert earnings_title_kind("2022年半年度业绩" + word + "公告") == "FORECAST"


@pytest.mark.parametrize(
    "title",
    [
        "关于召开业绩说明会公告",
        "关于收到业绩承诺补偿款的公告",
        "独立董事业绩考核意见",
        "证券公司关于上市公司2021年持续督导年度报告书",
    ],
)
def test_non_earnings_disclosure_excluded(title):
    assert earnings_title_kind(title) is None


def table(**changes):
    spec = {
        "page": 1,
        "kind": "REPORTED_RESULT",
        "comparison": "YEAR_ON_YEAR",
        "unit": "PERCENT",
        "period_start": "2022-01-01",
        "period_end": "2022-03-31",
        "metric": "PARENT_NET_PROFIT",
        "basis": "CONSOLIDATED_PARENT",
        "header": "本报告期比上年同期增减变动幅度(%)",
        "row_quote": "归母净利润155,144,879.59 98.85",
        "row_values": ["155144879.59", "98.85"],
        "value_index": 1,
        "value": "98.85",
    }
    spec.update(changes)
    return review_table_yoy(spec, [spec["header"] + "\n" + spec["row_quote"]])


def test_yoy_under_table_header_is_not_money():
    result = table()
    assert result["reported_value"] == "98.85" and not result["derived_from_money"]


def test_wrong_amount_column_fails():
    with pytest.raises(ValueError, match="COLUMN_MISMATCH"):
        table(value_index=0)


def test_duplicate_amount_or_bad_units_fail():
    with pytest.raises(ValueError, match="ROW_VALUES_MISMATCH"):
        table(row_values=["155144879.59", "99.85"])
    with pytest.raises(ValueError, match="NOT_YOY_PERCENT_HEADER"):
        table(header="本报告期比上年度末变化(%)")
    with pytest.raises(ValueError, match="PERCENTAGE_POINTS"):
        table(row_quote="归母净利润155,144,879.59 增加98.85个百分点")
