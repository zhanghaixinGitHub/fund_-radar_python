"""历史行情身份、日期边界和原始价格口径，不将接口历史行冒充盘中证据。"""

import json
from copy import deepcopy

import pytest
from app.integrations import tushare_sprint_asia as s


def payload():
    return {"code": 0, "data": {"fields": s.FIELDS, "items": [["N225", "20250106", 101.0, 99.0, 100.0]]}}


def parse(value):
    return s.parse_nikkei_history(json.dumps(value).encode(), "20250101", "20251231")


def test_raw_open_gap_not_close_return():
    out = parse(payload())["2025-01-06"]
    assert out["opening_gap_pct"] == pytest.approx(1.0)
    assert out["close"] == 99 and out["first_publication_verified"] is False


@pytest.mark.parametrize(
    "index,value",
    [(0, "KS11"), (1, "20260106"), (1, "20250105"), (2, 0), (2, True), (3, float("nan")), (4, float("inf"))],
)
def test_invalid_row_rejected(index, value):
    x = payload()
    x["data"]["items"][0][index] = value
    with pytest.raises(ValueError):
        parse(x)


def test_duplicate_empty_swapped_fields_and_provider_failure_rejected():
    candidates = []
    x = payload()
    x["data"]["items"] *= 2
    candidates.append(x)
    x = payload()
    x["data"]["items"] = []
    candidates.append(x)
    x = deepcopy(payload())
    x["data"]["fields"][2:4] = ["close", "open"]
    candidates.append(x)
    x = payload()
    x["code"] = 40203
    candidates.append(x)
    for x in candidates:
        with pytest.raises(ValueError):
            parse(x)
