"""联合信息要求时间与来源均完整，不因少量来源存在而伪造其余市场值。"""

from copy import deepcopy

import pytest
from app.services import direction_1d_sprint_joint_market_data_v2 as s


@pytest.fixture
def sources():
    quote = {
        "open": 100,
        "close": 101,
        "high": 102,
        "low": 99,
        "volume": 100,
        "available": True,
        "unavailable_reason": None,
    }
    return {
        "futures": {"2026-09-16": {"intraday_return_pct": 1, "close_location": 0.5, "zero_range": False}},
        "dollar": {"2026-09-16": deepcopy(quote)},
        "credit": {"HYG": {"2026-09-16": deepcopy(quote)}, "IEF": {"2026-09-16": deepcopy(quote)}},
    }


def test_joint_feature_order_retains_each_original_source_formula(sources):
    v = s.features("2026-09-16", "2026-09-17", sources)
    dollar = s.dollar.features("2026-09-16", "2026-09-17", sources["dollar"])
    pair = s.credit.features("2026-09-16", "2026-09-17", sources["credit"])
    assert v["available"] and v["values"] == [
        1,
        0.5,
        dollar["intraday_log_pct"],
        pair["relative_log_pct"],
        pair["treasury_log_pct"],
    ]


@pytest.mark.parametrize("missing", ["futures", "dollar", "credit", "HYG", "IEF"])
def test_any_missing_component_marks_whole_joint_input_unavailable(sources, missing):
    if missing in ["HYG", "IEF"]:
        sources["credit"].pop(missing)
    else:
        sources.pop(missing)
    v = s.features("2026-09-16", "2026-09-17", sources)
    assert v["available"] is False and v["values"] == [0.0] * 5


def test_future_if_cannot_replace_t_day_value(sources):
    sources["futures"] = {"2026-09-17": sources["futures"]["2026-09-16"]}
    assert s.features("2026-09-16", "2026-09-17", sources)["reason"] == "MISSING_T_IF"


def test_extend_does_not_mutate_or_remove_original_market_inputs(sources):
    market = {"available": True, "original_price": 1}
    result = s.extend(market, "2026-09-16", "2026-09-17", sources)
    assert market == {"available": True, "original_price": 1}
    assert s.original_row({"market": result})["market"] == market
