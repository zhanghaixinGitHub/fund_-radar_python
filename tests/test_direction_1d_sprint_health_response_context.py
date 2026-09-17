"""个体反应只读已成熟旧标签，未来变化不得影响既有样本；边界用小型合成数据验证。"""

from copy import deepcopy
from datetime import date, timedelta

import pytest
from app.services import direction_1d_sprint_health_response_context as s


def rows(n=130):
    return [
        {
            "code": "001021",
            "u": str(date(2023, 1, 1) + timedelta(days=i)),
            "mature": str(date(2023, 1, 2) + timedelta(days=i)),
            "y": i % 2,
            "market": {"china_health": {"available": True, "intraday_log_pct": 1.0 if i % 2 else -1.0}},
        }
        for i in range(n)
    ]


def test_future_or_same_day_labels_not_read():
    values = rows()
    cutoff = "2023-04-01"
    old = s.context(values, cutoff)
    changed = deepcopy(values)
    for row in changed:
        if row["u"] >= cutoff or row["mature"] >= cutoff:
            row["y"] = "not read"
    assert s.context(changed, cutoff) == old
    assert old["max_mature"] < cutoff and old["max_u"] < cutoff


def test_126_cap_and_positive_response():
    value = s.context(rows(), "2023-06-01")
    assert value["count"] == 126 and 0 < value["beta"] <= 3 and value["available"]


def test_minimum_is_fixed_63():
    assert not s.context(rows(62), "2023-06-01")["available"]
    assert s.context(rows(63), "2023-06-01")["available"]


def test_invalid_source_observations_not_implicitly_zero():
    values = rows(63)
    values[0]["market"]["china_health"]["available"] = False
    value = s.context(values, "2023-06-01")
    assert value["count"] == 62 and not value["available"] and value["beta"] == 0


def test_current_missing_feature_stays_missing():
    prior = s.context(rows(), "2023-06-01")
    assert not s.feature({"available": False, "intraday_log_pct": 0}, prior, "2023-06-02")["available"]


def test_future_context_rejected():
    prior = s.context(rows(), "2023-06-01")
    with pytest.raises(ValueError, match="CONTEXT_INVALID"):
        s.feature({"available": True, "intraday_log_pct": 1}, prior, "2023-05-31")


def test_bad_maturity_rejected():
    prior = s.context(rows(), "2023-06-01")
    prior["max_mature"] = "2023-06-01"
    with pytest.raises(ValueError, match="NOT_MATURE"):
        s.feature({"available": True, "intraday_log_pct": 1}, prior, "2023-06-02")


def test_duplicate_or_mixed_funds_rejected():
    values = rows(63)
    with pytest.raises(ValueError, match="DUPLICATE"):
        s.context(values + [values[-1]], "2023-06-01")
    values[-1]["code"] = "other"
    with pytest.raises(ValueError, match="MULTIPLE_FUNDS"):
        s.context(values, "2023-06-01")
