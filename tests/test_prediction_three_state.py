"""三分类的数值恢复、冻结规则和拒绝旧包；人工样本仅用于工程验收。"""

from copy import deepcopy
from datetime import datetime

import pytest
from app.services.prediction_contract import PredictionFailure, legacy_policy, policy_scope, prediction_policy
from app.services.prediction_direction import CLASSES, classify, classify_return, validate_identity, validate_rule
from app.services.prediction_models import FEATURES, baseline_package, infer_package, route_key, validate_package
from app.services.prediction_research import fit_package
from app.services.prediction_selection import direction_not_worse, evaluate_answers


@pytest.mark.parametrize(
    "value,expected", [("-.0031", "DOWN"), ("-.003", "FLAT"), ("0", "FLAT"), (".003", "FLAT"), (".0030000001", "UP")]
)
def test_decimal_closed_band(value, expected):
    assert classify_return(value, ".003") == expected


@pytest.mark.parametrize("threshold", ["0", "-1", "NaN", "Infinity", "bad"])
def test_invalid_threshold_fails(threshold):
    with pytest.raises(PredictionFailure):
        classify_return("0", threshold)


def training_samples():
    return [
        {
            "key": str(i),
            "actual": CLASSES[i % 3],
            "features": dict.fromkeys(FEATURES, (i % 3 - 1) * 0.2 + i * 0.0001),
            "knowledgeCutoff": "2023-01-01T00:00:00+08:00",
            "labelAvailableAt": "2023-06-01T00:00:00+08:00",
        }
        for i in range(90)
    ]


def test_all_class_parameters_restore_and_missing_class_is_not_faked():
    package = fit_package(training_samples(), "T5_V1", datetime.fromisoformat("2024-01-01T00:00:00+08:00"))
    validate_package(package)
    assert package["restoreMaxDiff"] < 1e-12
    assert tuple(package["parameters"]["classes"]) == CLASSES
    assert len(package["parameters"]["coefficients"]) == 3
    assert {infer_package(package, s["features"])["direction"] for s in training_samples()} == set(CLASSES)
    with pytest.raises(PredictionFailure):
        fit_package(
            [s for s in training_samples() if s["actual"] != "FLAT"],
            "T5_V1",
            datetime.fromisoformat("2024-01-01T00:00:00+08:00"),
        )


def test_baselines_frozen_identity_and_legacy_target_are_separate():
    original = deepcopy(prediction_policy())
    key = route_key("T5_V1")
    package = baseline_package("T5_V1")
    for horizon in ("T5_V1", "T20_V1", "M6_V1"):
        p = baseline_package(horizon)
        validate_package(p)
        assert infer_package(p, {"momentum": "0.001"})["direction"] == "FLAT"
        assert infer_package(p, {"momentum": "-0.1"})["direction"] == "DOWN"
        assert infer_package(p, {"momentum": "0.1"})["direction"] == "UP"
    changed = deepcopy(original)
    changed["direction"]["thresholds"]["T5_V1"] = "0.02"
    with policy_scope(changed):
        assert route_key("T5_V1") != key
        with pytest.raises(PredictionFailure):
            validate_identity(package, "T5_V1")
        assert classify(".01", "T5_V1") == "FLAT"
        assert classify(".01", "T5_V1", original) == "UP"
    assert route_key("T5_V1") == key
    with policy_scope(legacy_policy()):
        legacy = baseline_package("T5_V1")
        assert infer_package(legacy, {"momentum": 0})["direction"] == "NON_UP"
    legacy.update(targetDefinitionId=package["targetDefinitionId"])
    with pytest.raises(PredictionFailure):
        validate_package(legacy)
    broken = deepcopy(original["direction"])
    broken["thresholds"].pop("M6_V1")
    with pytest.raises(PredictionFailure):
        validate_rule(broken)


def test_majority_flat_does_not_win_and_empty_class_is_unknown():
    planned = [
        {
            "key": str(i),
            "actual": "FLAT" if i < 80 else "UP" if i < 90 else "DOWN",
            "date": "2024-01-01",
            "family": str(i),
            "fund": str(i),
        }
        for i in range(100)
    ]
    perfect = evaluate_answers(planned, {p["key"]: {"direction": p["actual"]} for p in planned})
    flat = evaluate_answers(planned, {p["key"]: {"direction": "FLAT"} for p in planned})
    assert flat["accuracy"] == 0.8 and flat["balancedAccuracy"] == pytest.approx(1 / 3)
    assert not direction_not_worse(flat, flat, 1e-8)
    assert direction_not_worse(perfect, flat, 1e-8)
    empty = evaluate_answers([], {})
    assert empty["classRecall"]["FLAT"] is None
    assert empty["balancedAccuracy"] is None
    with pytest.raises(PredictionFailure):
        evaluate_answers(planned, {"0": {"direction": "NON_UP"}})


def test_equal_highest_scores_are_deterministic_and_do_not_invent_confidence_band():
    from tests.test_prediction_models import three_package

    package = three_package()
    package["parameters"]["coefficients"] = [[0.0] * len(FEATURES) for _ in CLASSES]
    package["parameters"]["intercepts"] = [0.0, 0.0, 0.0]
    assert infer_package(package, dict.fromkeys(FEATURES, 0))["direction"] == "FLAT"
    package["parameters"]["intercepts"] = [0.0, 0.0, 0.00001]
    assert infer_package(package, dict.fromkeys(FEATURES, 0))["direction"] == "UP"
