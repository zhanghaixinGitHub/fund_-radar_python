"""合成输入验证仓位权重口径、冻结身份及日期保护。"""

import copy

import pytest
from app.services import fund_002112_input_audit as audit


def fixture():
    rows = [
        {"fund_code": code, "family": code, "target": "2022-06-01", "x": [0.0] * 13 + [0.9] + [0.0] * 6}
        for code in sorted(audit.COHORT)
    ]
    exam = {**rows[0], "fund_code": "002112", "target": "2023-04-03"}
    fold = {
        "name": "2023Q2",
        "start": "2023-04-01",
        "train_ids": [[r["fund_code"], r["target"]] for r in rows],
        "weights": [1.0] * 11,
        "train_hash": audit.digest(rows),
        "expected_dates": [exam["target"]],
    }
    return fold, {(r["fund_code"], r["target"]): r for r in rows + [exam]}


def test_actual_weights_are_not_raw_row_counts():
    assert audit.weighted_median([20, 90, 90], [10, 1, 1]) == 20
    assert audit.weighted_median([20, 90], [1, 1]) == 20
    result = audit.audit_fold(*fixture())
    assert result["by_fund"]["002112"]["learning_weight_share"] == pytest.approx(1 / 11)
    assert result["overall"]["stock_bins"][">=80%"]["learning_weight_share"] == 1


@pytest.mark.parametrize("kind", ["weight", "duplicate", "identity", "future", "allocation"])
def test_reject_changed_fold(kind):
    fold, index = copy.deepcopy(fixture())
    if kind == "weight":
        fold["weights"][0] = 0
    if kind == "duplicate":
        fold["train_ids"][1] = fold["train_ids"][0]
    if kind == "identity":
        fold["train_hash"] = "0" * 64
    if kind in ("future", "allocation"):
        row = index[tuple(fold["train_ids"][0])]
        if kind == "future":
            fold["start"] = "2022-01-01"
        else:
            row["x"][13] = float("nan")
            fold["train_hash"] = "invalid"
    with pytest.raises(ValueError):
        audit.audit_fold(fold, index)


@pytest.mark.parametrize("values,weights", [([], []), ([1], [0]), ([1], [-1]), ([1], [float("nan")])])
def test_invalid_statistics_not_imputed(values, weights):
    with pytest.raises(ValueError):
        audit.weighted_median(values, weights)
