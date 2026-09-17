"""八维同输入容量实验：成熟边界、按日计数、家族权重及缺源回退，不使用真实基金训练数据。"""

from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pytest
from app.services import direction_1d_sprint_market_joint_capacity as s


@pytest.fixture
def synthetic(monkeypatch):
    rows = []
    for i in range(504):
        day = date(2022, 1, 1) + timedelta(days=i)
        sign = 1 if i % 3 else -1
        for code in ["A", "B"]:
            rows.append(
                {
                    "code": code,
                    "family": code,
                    "group": "CN_EQUITY",
                    "t": str(day - timedelta(days=1)),
                    "u": str(day),
                    "mature": str(day),
                    "y": int(sign > 0),
                    "market": {
                        "available": True,
                        "raw": [sign, -sign, sign * 0.2],
                        "joint_market": {
                            "available": True,
                            "values": [sign * 0.5, sign * 0.1, sign * 0.01, -sign * 0.3, sign * 0.4],
                            "new_us_dates": [str(day - timedelta(days=1))],
                            "if_date": str(day - timedelta(days=1)),
                            "reason": None,
                        },
                    },
                }
            )
    monkeypatch.setattr(s.original, "training_rows", lambda values, _: values)
    monkeypatch.setattr(s.baseline, "market_vector", lambda value: value["raw"])
    monkeypatch.setattr(s, "active", lambda: None)
    return rows


@pytest.mark.parametrize("candidate", s.CANDIDATES)
def test_actual_fit_preserves_rows_weights_and_tree_counts_dates(synthetic, candidate):
    head = s.fit(synthetic, candidate, "2025-01-01")
    model = s.validate_head(head, candidate, "2025-01-01")
    assert head["fit_dates"] == 504 and head["fit_rows"] == 1008
    assert head["fit_hash"] == s.base.digest([s.data.original_row(r) for r in synthetic])
    assert head["weight_hash"] == s.base.digest(s.regression.weights(synthetic).tolist())
    assert head["estimator_training_units"] == (1008 if candidate == s.CANDIDATES[0] else 504)
    if candidate == s.CANDIDATES[1]:
        for predictors in model._predictors:
            nodes = predictors[0].nodes
            assert min(nodes[nodes["is_leaf"] == 1]["count"]) >= 63
    values = [r["market"] for r in synthetic[:4]]
    answers = s.candidate_answers(values, candidate, head, [{"prediction": 0}] * len(values))
    assert all(a["prediction"] == int(a["research_score"] >= 0.5) for a in answers)


def test_date_aggregation_retains_family_weighted_loss_difference():
    rows = [{"u": "d", "family": f, "group": "g", "y": y} for f, y in [("A", 1), ("A", 0), ("B", 0)]]
    weights = s.regression.weights(rows)
    x, y, w, days = s.date_arrays(rows, np.ones((3, 8)), weights)
    assert len(x) == 1 and days == ["d"] and y[0] == 0.25 and w.sum() == weights.sum()
    # 两个预测值的损失差保持相等；汇总只去掉与预测无关的日内标签方差。
    raw = np.array([r["y"] for r in rows])
    delta = np.sum(weights * ((raw - 0.2) ** 2 - (raw - 0.8) ** 2))
    assert delta == pytest.approx(np.sum(w * ((y - 0.2) ** 2 - (y - 0.8) ** 2)))


def test_different_fund_inputs_cannot_be_collapsed_to_one_date():
    rows = [{"u": "d", "group": "g", "y": y} for y in [0, 1]]
    with pytest.raises(ValueError, match="DIFFERENT_INPUTS_SAME_DATE"):
        s.date_arrays(rows, np.array([[0.0] * 8, [1.0] * 8]), np.ones(2))


@pytest.mark.parametrize("kind", ["immature", "missing", "nonbinary"])
def test_invalid_training_stops_before_any_estimator_fit(synthetic, kind):
    if kind == "immature":
        synthetic[-1]["mature"] = "2025-01-01"
    elif kind == "missing":
        synthetic[-1]["market"]["joint_market"]["available"] = False
    else:
        synthetic[-1]["y"] = 2
    with pytest.raises(ValueError):
        s.fit(synthetic, s.CANDIDATES[1], "2025-01-01")


@pytest.mark.parametrize("candidate", s.CANDIDATES)
def test_missing_source_uses_exact_frozen_answer(candidate):
    values = [
        {"available": False, "joint_market": {"available": True}},
        {"available": True, "joint_market": {"available": False}},
    ]
    fallback = [{"prediction": 0, "route": "SPX"}, {"prediction": 1, "route": "SIGN"}]
    assert s.candidate_answers(values, candidate, None, deepcopy(fallback)) == fallback


def test_scaling_keeps_continuous_sign_and_uses_only_supplied_training_scale():
    x = [[0.2, -0.4, 0.6, 0.8, 1, 0.1, 0.2, 0.3]]
    assert np.allclose(s.transform(x, [2] * 8), np.array(x) / 2)
    with pytest.raises(ValueError, match="TRANSFORM_INVALID"):
        s.transform(x, [1] * 7)


@pytest.mark.parametrize("value", [True, float("nan"), 21.0])
def test_invalid_added_market_values_rejected(synthetic, value):
    market = synthetic[0]["market"]
    market["joint_market"]["values"][0] = value
    with pytest.raises(ValueError, match="FEATURE_RANGE_INVALID"):
        s.vector(market)


def test_identity_preserves_labels_inputs_and_all_proof_except_observation_clock():
    rows = [{"market": {"old": 1, "joint_market": {"values": [1] * 5}}, "y": 1}]
    proof = {"at": "first", "source": "fixed"}
    expected = s.question_identity(rows, proof)
    assert s.verify_identity(rows, proof | {"at": "second"}, expected) == expected
    rows[0]["y"] = 0
    with pytest.raises(ValueError, match="QUESTION_IDENTITY_CHANGED"):
        s.verify_identity(rows, proof, expected)
