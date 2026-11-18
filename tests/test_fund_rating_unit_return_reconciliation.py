import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from app.services.fund_rating_qdii_disclosure import verified_unit_total_return


def setup_case():
    params = {"api_name": "fund_div", "ts_code": "008764.OF"}
    meta = {
        "div": {"url": "https://api.tushare.pro", "params": params},
        "adj": {
            "url": "https://api.tushare.pro",
            "params": {
                "api_name": "fund_adj",
                "ts_code": "008764.OF",
                "start_date": "20250930",
                "end_date": "20261030",
            },
        },
    }
    values = {
        "div": {"code": 0, "data": {"items": []}},
        "adj": {"code": 0, "data": {"fields": ["ts_code", "trade_date", "adj_factor"], "items": [], "has_more": False}},
    }
    point = {
        "x": int(datetime(2026, 9, 29, tzinfo=timezone(timedelta(hours=8))).timestamp() * 1000),
        "y": 1.5472,
        "unitMoney": "",
    }
    chart = "var Data_netWorthTrend=" + json.dumps([point]) + ";"

    class Source:
        sources = meta
        raw = {"chart": chart.encode()}

        def json(self, key):
            return values[key]

    Source.sources["split"] = {
        "url": "https://fundf10.eastmoney.com/fhsp_008764.html",
        "method": "GET",
        "status": 200,
        "params": None,
        "received_at": "2026-11-17T13:00:00+08:00",
    }
    Source.raw["split"] = (
        "<title>天弘越南(008764)基金分红送配</title><table><tr><th>年份</th>"
        "<th>拆分折算日</th><th>拆分类型</th><th>拆分折算比例</th></tr>"
        "<tr><td>暂无拆分信息!</td></tr></table>"
    ).encode()

    snapshot = {
        "rows": [{"nav_date": "2026-09-29", "unit_nav": "1.5472", "accumulated_nav": "1.5472", "adjusted_nav": None}],
        "unit_return_reconciliation": {
            "method": "DUAL_SOURCE_UNIT_NAV_NO_DISTRIBUTION_OR_SPLIT",
            "dividend_source_ref": "div",
            "adjustment_source_ref": "adj",
            "split_source_ref": "split",
        },
    }
    inputs = {"fund_code": "008764", "as_of_date": "2026-10-30", "nav_chart_ref": "chart"}
    return Source(), inputs, snapshot, values, point


def test_null_preserved_with_verified_independent_unit_return():
    source, inputs, snapshot, _, _ = setup_case()
    result = verified_unit_total_return(source, inputs, snapshot, {"2026-09-29": Decimal("1.5472")})
    assert result["original_adjusted_nav_null_dates"] == ["2026-09-29"]
    assert result["adjusted_nav_imputed"] is False
    assert snapshot["rows"][0]["adjusted_nav"] is None


@pytest.mark.parametrize(
    "failure", ["cash", "factor", "chart_split", "split_history", "crosscheck", "cumulative", "scope"]
)
def test_events_and_value_conflicts_cannot_be_treated_as_no_events(failure):
    source, inputs, snapshot, values, point = setup_case()
    if failure == "cash":
        values["div"]["data"]["items"] = [["008764.OF", "20260929", ".2"]]
    elif failure == "factor":
        values["adj"]["data"]["items"] = [["008764.OF", "20260929", "2"]]
    elif failure == "chart_split":
        point["unitMoney"] = "基金份额折算"
        source.raw["chart"] = ("var Data_netWorthTrend=" + json.dumps([point]) + ";").encode()
    elif failure == "split_history":
        source.raw["split"] = source.raw["split"].replace("暂无拆分信息!".encode(), "2026年存在拆分".encode())
    elif failure == "crosscheck":
        snapshot["rows"][0]["unit_nav"] = "1.6"
    elif failure == "cumulative":
        snapshot["rows"][0]["accumulated_nav"] = "1.6"
    else:
        source.sources["adj"]["params"]["end_date"] = "20260930"
    with pytest.raises(ValueError):
        verified_unit_total_return(source, inputs, snapshot, {"2026-09-29": Decimal("1.5472")})
