"""验证个体上下文的封闭公式、基金隔离和历史可见性。"""

from datetime import date, timedelta

import numpy as np
import pytest
from app.services import direction_1d_sprint_sector_joint_context as s


def rows(count=200):
    output = []
    for i in range(count):
        day = date(2023, 1, 1) + timedelta(days=i)
        move = 1.0 if i % 2 else -1.0
        x, z = [0.0] * 32, [0.0] * 17
        x[23] = 0.03
        x[7] = 2 * move / 100 + x[23]
        z[12:17] = [move * k for k in range(1, 6)]
        output.append({"code": "fund", "u": str(day), "mature": str(day + timedelta(days=1)), "x": x, "z": z})
    return output


def test_fixed_covariance_formula_and_last126_dates():
    prior = s.context(rows(), "2024-01-01")
    assert prior["count"] == 126 and prior["available_fraction"] == 1
    assert prior["beta"] == pytest.approx([k / 55.25 for k in range(1, 6)])
    assert prior["max_u"] == rows()[-1]["u"]


def test_no_history_is_kept_as_zero_not_dropped():
    prior = s.context(rows(), "2023-01-01")
    assert prior["count"] == 0 and prior["max_mature"] is None
    assert s.extra_features(rows()[0], prior) == [0.0] * 11


def test_future_inputs_and_any_direction_labels_are_ignored():
    values = rows()
    prior = s.context(values, "2023-05-01")
    for row in values:
        row["y"] = "unused poisoned direction"
        if row["mature"] >= "2023-05-01" or row["u"] >= "2023-05-01":
            row["x"], row["z"] = [float("nan")] * 32, [float("nan")] * 17
    assert s.context(values, "2023-05-01") == prior


def test_other_funds_and_duplicate_dates_are_rejected():
    values = rows()
    with pytest.raises(ValueError, match="MULTIPLE_FUNDS"):
        s.context(values + [values[-1] | {"code": "other"}], "2024-01-01")
    with pytest.raises(ValueError, match="DUPLICATE_DATE"):
        s.context(values + [values[-1]], "2024-01-01")


def test_zero_market_variance_and_extreme_sector_joint_are_bounded():
    values = rows()
    for row in values:
        row["x"][7] *= 1000
    assert s.context(values, "2024-01-01")["beta"] == [3.0] * 5
    for row in values:
        row["z"] = [0.0] * 17
    assert s.context(values, "2024-01-01")["beta"] == [0.0] * 5


def test_feature_products_use_correct_market_units():
    value = rows()[-1]
    prior = s.context(rows(), "2024-01-01")
    value = value | {"u": "2024-01-02"}
    extra = s.extra_features(value, prior)
    np.testing.assert_allclose(extra[5:10], np.asarray(prior["beta"]) * np.arange(1, 6))


@pytest.mark.parametrize(
    "change",
    [
        {"count": True},
        {"count": 127},
        {"beta": [float("nan")] * 5},
        {"available_fraction": 0.5},
        {"max_mature": "2024-01-01"},
        {"max_u": "2024-01-01"},
    ],
)
def test_invalid_context_values_or_future_dates_are_rejected(change):
    prior = s.context(rows(), "2024-01-01")
    with pytest.raises(ValueError):
        s.validate_context(prior | change, "2024-01-01")


def test_common_market_move_does_not_change_residual_exposure():
    values = rows()
    before = s.context(values, "2024-01-01")
    for i, row in enumerate(values):
        change = 0.01 if i % 2 else -0.02
        row["x"][7] += change
        row["x"][23] += change
    assert s.context(values, "2024-01-01")["beta"] == pytest.approx(before["beta"])


def test_joint_solution_matches_regularized_normal_equation():
    values = rows(126)
    for i, row in enumerate(values):
        row["z"][12:17] = [np.sin((i + 1) * (j + 1)) for j in range(5)]
        row["x"][23] = 0.02
        row["x"][7] = 0.02 + sum((j + 1) * v for j, v in enumerate(row["z"][12:17])) / 100
    result = s.context(values, "2024-01-01")
    factors = np.asarray([r["z"][12:17] for r in values])
    funds = np.asarray([(r["x"][7] - r["x"][23]) * 100 for r in values])
    centered = factors - factors.mean(axis=0)
    covariance = centered.T @ centered / len(values)
    cross = centered.T @ (funds - funds.mean()) / len(values)
    # 本合成例的系数未触及±3裁剪，乘回2即去除126/(126+126)收缩。
    np.testing.assert_allclose(
        (covariance + 0.25 * np.eye(5)) @ (np.asarray(result["beta"]) * 2), cross, rtol=1e-12, atol=1e-12
    )
