"""成熟边界先于标签访问；验证个体差异、原题保留所需缺失语义及固定窗口。"""

from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pytest
from app.services import direction_1d_sprint_fund_moneyflow_context as c


def sample(n=140):
    rows, points = [], {}
    for i in range(n):
        t = str(date(2023, 1, 1) + timedelta(days=i))
        u = str(date(2023, 1, 2) + timedelta(days=i))
        flow = 0.05 if i % 2 else -0.05
        rows.append({"code": "fund", "t": t, "u": u, "mature": u, "y": i % 2})
        points[t] = {"date": t, "available": True, "large_imbalance": flow, "net_fraction": -flow}
    return rows, points


def test_fixed_window_and_response_sign():
    rows, points = sample()
    prior = c.context(rows, "2024-01-01", points)
    assert prior["count"] == 126 and prior["available_fraction"] == 1
    assert prior["beta"][0] > 0 and prior["beta"][1] < 0
    c.validate(prior, "2024-01-01")


def test_target_future_and_unmature_labels_never_read():
    rows, points = sample(8)
    cutoff = "2023-02-01"
    baseline = c.context(rows, cutoff, points)
    poison = [
        {"code": "future", "t": "DO_NOT_READ", "u": cutoff, "mature": "2023-01-01"},
        {"code": "future", "t": "DO_NOT_READ", "u": "2024-01-01", "mature": "2023-01-01"},
        {"code": "future", "t": "DO_NOT_READ", "u": "2023-01-15", "mature": cutoff},
    ]
    assert c.context(rows + poison, cutoff, points) == baseline


def test_distinct_fund_reactions_change_features_without_current_label():
    rows, points = sample()
    other = [r | {"code": "other", "y": 1 - r["y"]} for r in rows]
    p, q = c.context(rows, "2024-01-01", points), c.context(other, "2024-01-01", points)
    assert np.allclose(p["beta"], -np.asarray(q["beta"]))
    point = {"date": "2023-12-29", "available": True, "large_imbalance": 0.1, "net_fraction": -0.1}
    assert c.extra_features(p, "2024-01-01", point) != c.extra_features(q, "2024-01-01", point)


def test_missing_history_preserves_explicit_zero_count():
    prior = c.context([], "2024-01-01", {})
    c.validate(prior, "2024-01-01")
    assert prior["count"] == 0 and prior["beta"] == [0.0, 0.0]
    assert c.extra_features(prior, "2024-01-01", None) is None


def test_constant_flow_returns_zero_not_nan():
    rows, points = sample(10)
    for point in points.values():
        point.update(large_imbalance=0.1, net_fraction=0.2)
    prior = c.context(rows, "2024-01-01", points)
    assert np.allclose(prior["beta"], [0, 0])


def test_missing_flow_not_added_as_zero_sample():
    rows, points = sample(8)
    points.pop(rows[0]["t"])
    points[rows[1]["t"]]["available"] = False
    assert c.context(rows, "2024-01-01", points)["count"] == 6


def test_duplicate_date_rejected():
    rows, points = sample(8)
    with pytest.raises(ValueError, match="DUPLICATE_TARGET_DATE"):
        c.context(rows + [rows[-1]], "2024-01-01", points)


def test_multiple_funds_rejected():
    rows, points = sample(8)
    rows[0]["code"] = "other"
    with pytest.raises(ValueError, match="MULTIPLE_FUNDS"):
        c.context(rows, "2024-01-01", points)


def test_old_label_is_required_to_be_binary():
    rows, points = sample(8)
    rows[0]["y"] = 7
    with pytest.raises(ValueError, match="INVALID_MATURE_LABEL"):
        c.context(rows, "2024-01-01", points)


def test_source_date_must_match_old_question():
    rows, points = sample(8)
    points[rows[0]["t"]]["date"] = "2022-12-31"
    with pytest.raises(ValueError, match="SOURCE_ALIGNMENT"):
        c.context(rows, "2024-01-01", points)


@pytest.mark.parametrize(
    "key,value,reason",
    [
        ("max_mature", "2024-01-01", "NOT_MATURE"),
        ("count", 127, "COUNT_INVALID"),
        ("beta", [4.0, 0.0], "BETA_INVALID"),
        ("available_fraction", 2.0, "FRACTION_CHANGED"),
    ],
)
def test_context_mutation_rejected(key, value, reason):
    rows, points = sample(8)
    prior = deepcopy(c.context(rows, "2024-01-01", points))
    prior[key] = value
    with pytest.raises(ValueError, match=reason):
        c.validate(prior, "2024-01-01")
