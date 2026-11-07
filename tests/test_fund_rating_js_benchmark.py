"""官网短窗基准必须核实际产品、份额净值、原精度和全日历；均为隔离样例。"""

from copy import deepcopy

import pytest
from app.services.fund_rating_js_benchmark import compile_history


@pytest.fixture
def specimen():
    dates = ["2026-09-28", "2026-09-29", "2026-09-30"]
    expression = "中证隔离指数收益率×95%＋银行活期存款利率（税后）×5%"
    raw = {
        "protocol": {"fund_code": "999999", "as_of": dates[-1], "index_code": "fixture.CSI",
                     "issuer_percent_decimal_places": 2, "issuer_level_half_quantum": ".00005",
                     "index_close_half_quantum": ".00005", "frozen_at": "2026-10-01"},
        "identity": {"error_no": "0", "results": [{"product_code": "999999", "product_id": "fixture",
                                                   "product_name": "隔离指数联接基金"}]},
        "index": {"code": 0, "data": {"fields": ["ts_code", "trade_date", "close"],
                                     "items": [["fixture.CSI", day, "100"] for day in dates]}},
        "window": {"error_no": "0", "results": [{"date": day.replace("-", ""), "index_name": expression,
                                                  "nav_yield": "0.00", "hs300_yield": "0.00"} for day in dates]},
    }
    metadata = {"window": {"url": "https://www.jsfund.cn/servlet/json", "request_params": {
        "funcNo": "741032", "product_code": "999999", "product_id": "fixture",
        "start_date": "20260928", "end_date": "20260930"}}}

    class Source:
        sources = metadata

        def json(self, digest):
            return raw[digest]

        def pages(self, _digest):
            return ["业绩比较基准" + expression + "。"]

        def plain(self, _digest):
            return "活期存款0.350.35"

    inputs = {"fund_code": "999999", "legal_name": "隔离指数联接基金", "as_of_date": dates[-1],
              "verified_at": "2026-10-02", "index": {"code": "fixture.CSI"},
              "target_refs": {"summary": "summary", "prospectus": "prospectus"},
              "js_benchmark_reconstruction": {"protocol_hash": "protocol", "annual_cash_rate": ".0035",
                  "rate_source": "rate", "product_identity": "identity", "product_id": "fixture",
                  "index_daily": "index", "windows": ["window"]}}
    history = {"dates": dates, "rows": [{"nav_date": day, "adjusted_nav": "1"} for day in dates]}
    return Source(), inputs, history, raw, metadata


def test_js_real_product_and_share_nav_are_checked_before_constructed_reference(specimen):
    source, inputs, history, raw, _ = specimen
    result = compile_history(source, inputs, deepcopy(history))
    assert result["publisher_cross_check"]["published_valuation_dates"] == 3
    assert result["publisher_cross_check"]["own_share_observations_checked"] == 3
    raw["identity"]["results"][0]["product_code"] = "999998"
    with pytest.raises(ValueError, match="PRODUCT_IDENTITY"):
        compile_history(source, inputs, deepcopy(history))


@pytest.mark.parametrize("change,error", [
    ("own_nav", "OWN_SHARE_NAV_CONFLICT"), ("benchmark", "PRECISION_CONFLICT"),
    ("precision", "PRECISION_CHANGED"), ("expression", "CONTRACT_EXPRESSION"),
    ("request", "REQUEST_SCOPE"), ("missing", "DAILY_COVERAGE"),
])
def test_js_source_conflicts_are_not_imputed_or_tolerance_fitted(specimen, change, error):
    source, inputs, history, raw, metadata = specimen
    if change == "own_nav":
        raw["window"]["results"][1]["nav_yield"] = "1.00"
    elif change == "benchmark":
        raw["window"]["results"][1]["hs300_yield"] = "1.00"
    elif change == "precision":
        raw["window"]["results"][1]["hs300_yield"] = "0.001"
    elif change == "expression":
        raw["window"]["results"][0]["index_name"] = "其他指数×95%＋银行活期存款利率（税后）×5%"
    elif change == "request":
        metadata["window"]["request_params"]["product_code"] = "999998"
    else:
        raw["window"]["results"].pop(1)
    with pytest.raises(ValueError, match=error):
        compile_history(source, inputs, deepcopy(history))
