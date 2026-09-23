"""受控适配器与选择协议测试；不把人工候选胜出声称为真实研究改进。"""

from copy import deepcopy

import pytest
from app.services.prediction_contract import PredictionFailure
from app.services.prediction_models import FEATURES, infer_package, validate_package
from app.services.prediction_selection import choose_candidate, evaluate_answers


def package():
    return {
        "adapter": "LOGISTIC_STANDARDIZED_V1",
        "recipeVersion": "L1",
        "horizonId": "T5_V1",
        "targetDefinitionId": "NEXT_EXECUTABLE_CASH_REINVESTED_DIRECTION_V1",
        "assetGroup": "ALL",
        "featureSchemaVersion": "NAV_TOTAL_RETURN_V1",
        "features": list(FEATURES),
        "featureUnits": ["RATIO"] * 6 + ["SESSIONS"],
        "missingPolicy": "FAIL_REQUIRED",
        "threshold": 0.5,
        "labelEndMax": "2024-01-01T00:00:00+00:00",
        "trainedAt": "2024-01-02T00:00:00+00:00",
        "codeVersion": "V1",
        "dependencies": {},
        "evidenceLevel": "DEVELOPMENT_ONLY",
        "parameters": {"mean": [1.0] * 7, "scale": [2.0] * 7, "coefficients": [0.2] * 7, "intercept": 0},
    }


def three_package(horizon="T5_V1"):
    """受控三态包；旧package保留用于二分类恢复回归，避免用改标签的旧包冒充新模型。"""
    from app.services.prediction_direction import TARGET, direction_fields
    p = package()
    p.update(adapter="LOGISTIC_MULTICLASS_V2", recipeVersion="TOTAL_RETURN_LOGISTIC_THREE_STATE_V2",
             horizonId=horizon, targetDefinitionId=TARGET, **direction_fields(horizon))
    p["parameters"] = {"mean": [1.0] * 7, "scale": [2.0] * 7,
                       "classes": ["DOWN", "FLAT", "UP"],
                       "coefficients": [[-.2] * 7, [0.0] * 7, [.2] * 7], "intercepts": [0, -.2, 0]}
    return p


def test_linear_formula_and_exact_threshold():
    import math

    p = package()
    validate_package(p)
    result = infer_package(p, dict.fromkeys(FEATURES, 3.0))
    assert result["score"] == pytest.approx(1 / (1 + math.exp(-1.4)))
    assert infer_package(p, dict.fromkeys(FEATURES, 1.0))["direction"] == "NON_UP"


def test_tree_has_dedicated_adapter_and_future_labels_rejected():
    p = package()
    p.update(
        adapter="DECISION_TREE_V1",
        parameters={"tree": {"feature": "return_5d", "threshold": 0, "left": {"score": 0.2}, "right": {"score": 0.8}}},
    )
    validate_package(p)
    assert infer_package(p, dict.fromkeys(FEATURES, 1.0))["direction"] == "UP"
    p["labelEndMax"] = "2024-01-03T00:00:00+00:00"
    with pytest.raises(PredictionFailure, match="训练标签"):
        validate_package(p)
    p["adapter"] = "UNSUPPORTED_PICKLE"
    with pytest.raises(PredictionFailure):
        validate_package(p)


def test_family_date_denominator_and_failed_coverage():
    planned = [
        {"key": str(i), "fund": str(i), "date": "2024-01-01", "family": "a" if i < 2 else "b", "actual": "UP"}
        for i in range(4)
    ]
    metrics = evaluate_answers(
        planned, {"0": {"direction": "UP"}, "1": {"direction": "UP"}, "2": {"direction": "DOWN"}}
    )
    assert metrics["primaryScore"] == 0.5
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["coverage"] == 0.75
    assert metrics["dateCount"] == 1


def candidate(mid, score=0.55, cost=1, label="2024-01-01"):
    return {
        "modelId": mid,
        "metrics": {"primaryScore": score, "coverage": 1},
        "protocolHash": "p",
        "sampleHash": "s",
        "runnable": True,
        "timeValid": True,
        "cost": cost,
        "recipe": "L1",
        "labelEndMax": label,
        "activationRevision": 1,
    }


def test_candidate_strictly_better_activates_without_80_percent():
    result = choose_candidate(candidate("old"), [candidate("new", 0.58)], protocol_hash="p")
    assert result["decision"] == "ACTIVATE" and result["winner"] == "new"


def test_newer_but_worse_kept_as_shadow_and_tie_breaks():
    old = candidate("old")
    result = choose_candidate(old, [candidate("new", 0.50, label="2025-01-01")], protocol_hash="p")
    assert result["decision"] == "KEEP_CURRENT"
    assert result["shadows"] == ["new"]
    assert choose_candidate(old, [candidate("cheap", cost=0.5)], protocol_hash="p")["winner"] == "old"
    assert choose_candidate(old, [candidate("recent", label="2025-01-01")], protocol_hash="p")["winner"] == "old"
    bad = deepcopy(old)
    bad.update(modelId="incomparable", sampleHash="other")
    with pytest.raises(PredictionFailure):
        choose_candidate(old, [bad], protocol_hash="p")
