"""个股缺报价不等于基金净值缺基准；关键输入仍必须真实齐备。"""

from copy import deepcopy

import pytest
from app.services.prediction_v2_baseline import baseline_ready


def source():
    return {
        "window": {"base_nav_date": "2026-11-13"},
        "nav": [{"nav_date": "2026-11-13"}],
        "market": {"ETF": {"date": "2026-11-13", "change_pct": -1.2}},
        "time_context": {
            "baseline_complete": False,
            "baseline_date": "2026-11-13",
            "latest_nav_date": "2026-11-13",
            "nav_status": "READY",
            "stocks": {"HALTED": {"date": "2026-10-30", "status": "STALE"}},
            "markets": {"ETF": {"date": "2026-11-13", "status": "READY"}},
        },
    }


def test_partial_stock_gap_preserved_without_blocking_identified_baseline():
    data = source()
    original = deepcopy(data)
    assert baseline_ready(data)
    assert data == original


@pytest.mark.parametrize("gap", ["nav", "market", "value", "empty", "identity"])
def test_critical_baseline_gap_remains_blocked(gap):
    data = source()
    if gap == "nav":
        data["nav"][0]["nav_date"] = "2026-11-12"
    elif gap == "market":
        data["market"]["ETF"]["date"] = "2026-11-12"
    elif gap == "value":
        data["market"]["ETF"]["change_pct"] = None
    elif gap == "empty":
        data["market"] = data["time_context"]["markets"] = {}
    else:
        data["time_context"]["markets"]["OTHER"] = {"status": "READY"}
    assert not baseline_ready(data)
