"""字段删除、可得时间与选择边界；不执行真实模型拟合。"""

import copy

import numpy as np
import pytest

from scripts import fund_002112_one_day_ablation_v1 as a


def row():
    return {"groups": {"N": list(range(8)), "NE": list(range(8, 15))}, "E": list(range(15, 35))}


@pytest.mark.parametrize(
    "candidate,removed", [("FULL", []), ("NO_F15", [29]), ("NO_F20", [34]), ("NO_F15_F20", [29, 34])]
)
def test_only_expected_columns_removed(candidate, removed):
    r = row()
    before = copy.deepcopy(r)
    got = a.matrix([r], candidate)[0]
    assert got.tolist() == [float(x) for x in range(35) if x not in removed]
    assert r == before and got[:15].tolist() == list(range(15))


def test_original_missing_values_remain_missing():
    r = row()
    r["E"][2] = None
    assert np.isnan(a.matrix([r], "NO_F15_F20")[0, 17])
    assert r["E"][2] is None


def test_preprocessing_uses_training_only_and_saved_state():
    left, right = row(), row()
    left["E"][2] = None
    right["E"][2] = 4
    _, state = a.transform([left, right], "NO_F15")
    old_statistics = state["imputer"].statistics_.copy()
    future = row()
    future["E"][2] = 100000
    a.transform([future], "NO_F15", state)
    np.testing.assert_array_equal(old_statistics, state["imputer"].statistics_)
    assert state["imputer"].statistics_[17] == 4


@pytest.mark.parametrize("field", ["label", "nav"])
def test_equality_at_cutoff_cannot_train(field):
    rows = [
        {"target": "2025-01-02", "base": "2024-12-31", "label_mature_at": "2025-01-03T07:00:00+08:00"},
        {"target": "2025-01-06", "base": "2025-01-03"},
    ]
    nav = {d: {"available_at": "2025-01-03T07:00:00+08:00"} for d in ["2024-12-31", "2025-01-02"]}
    update = {"training": [0], "evaluate": [1], "cutoff": "2025-01-03T08:00:00+08:00"}
    a.validate_training(rows, update, nav)
    if field == "label":
        rows[0]["label_mature_at"] = update["cutoff"]
    else:
        nav["2025-01-02"]["available_at"] = update["cutoff"]
    with pytest.raises(ValueError, match="IMMATURE"):
        a.validate_training(rows, update, nav)


def test_deterministic_selection_uses_only_2025_metrics():
    scores = {g: {"correct": 130, "brier": 0.5} for g in a.CANDIDATES}
    assert a.choose(scores) == "NO_F15_F20"
    scores["NO_F15"]["correct"] = 131
    assert a.choose(scores) == "NO_F15"
    scores["NO_F20"] = {"correct": 131, "brier": 0.49}
    assert a.choose(scores) == "NO_F20"


def test_comparison_requires_same_order_and_full_dates():
    with pytest.raises(ValueError, match="COMMON_DATES_REQUIRED"):
        a.comparison([{"target": "a"}], [{"target": "b"}], {})


def test_accuracy_alone_cannot_pass_support_check():
    baseline = {
        "correct": 100,
        "brier": 0.5,
        "recall": {"DOWN": 0.6},
        "quarters": {str(q): {"correct": 25} for q in range(1, 5)},
    }
    candidate = copy.deepcopy(baseline)
    candidate["correct"] = 110
    assert a.support(candidate, baseline)["passed"]
    candidate["recall"]["DOWN"] = 0.5
    assert not a.support(candidate, baseline)["passed"]


def test_cannot_change_root_to_reset_budget(tmp_path):
    with pytest.raises(ValueError, match="FIXED_ROOT_REQUIRED"):
        a.execute(tmp_path)
