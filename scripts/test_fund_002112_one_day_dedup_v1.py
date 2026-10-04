"""去重研究的列契约、缺失语义和固定选择；单测不拟合预测模型。"""

import copy

import numpy as np
import pytest

from scripts import fund_002112_one_day_dedup_v1 as a
from scripts import fund_002112_one_day_dedup_verify_v1 as verify


def row():
    return {"groups": {"N": list(range(8)), "NE": list(range(8, 15))}, "E": list(range(15, 35))}


@pytest.mark.parametrize(
    "candidate,drop",
    [
        ("DEDUP", [28]),
        ("DEDUP_NO_COUNT", [27, 28]),
        ("DEDUP_NO_EARNINGS_WEIGHT", [24, 25, 28]),
        ("DEDUP_NO_COUNT_EARNINGS_WEIGHT", [24, 25, 27, 28]),
    ],
)
def test_column_contract_preserves_nav_f12_f15_f20(candidate, drop):
    r = row()
    original = copy.deepcopy(r)
    assert a.matrix([r], candidate)[0].tolist() == [float(i) for i in range(35) if i not in drop]
    assert r == original and {26, 29, 34}.issubset(a.kept(candidate))
    assert set(drop) == verify.EXPECTED_DROPS[candidate]


def test_deleted_missing_indicator_does_not_survive():
    left, right = row(), row()
    left["E"][13] = None  # 被删F14的缺失指示也必须删除。
    left["E"][2] = None  # 保留字段的缺失仍需要表达。
    _, state = a.transform([left, right], "DEDUP")
    assert state["imputer"].indicator_.features_.tolist() == [17]
    assert np.isnan(a.matrix([left], "DEDUP")[0, 17])


def test_future_rows_cannot_refit_preprocessing():
    train = row()
    train["E"][2] = None
    _, state = a.transform([train], "DEDUP")
    old_mean = state["scaler"].mean_.copy()
    future = row()
    future["E"][2] = 100000
    a.transform([future], "DEDUP", state)
    np.testing.assert_array_equal(state["scaler"].mean_, old_mean)
    assert train["E"][2] is None


def test_duplicate_screen_includes_missingness():
    left, right = row(), row()
    left["E"][11] = left["E"][13] = None
    right["E"][11] = right["E"][13] = 0.2
    assert a.duplicate_evidence([left, right])["rows"] == 2
    left["E"][13] = 0
    with pytest.raises(ValueError, match="NOT_IDENTICAL"):
        a.duplicate_evidence([left, right])


def test_constant_columns_cannot_establish_variable_duplication():
    r = row()
    r["E"][11] = r["E"][13] = None
    with pytest.raises(ValueError, match="NO_VARIABLE"):
        a.duplicate_evidence([r, r])


def test_fixed_tiebreaker_ignores_diagnostic_scores():
    scores = {g: {"correct": 120, "brier": 0.5} for g in a.CANDIDATES}
    assert a.choose(scores) == "DEDUP_NO_COUNT_EARNINGS_WEIGHT"
    scores["DEDUP"]["correct"] = 121
    scores["2026_BEST"] = {"correct": 999, "brier": 0}
    assert a.choose(scores) == "DEDUP"
    scores["DEDUP_NO_COUNT"] = {"correct": 121, "brier": 0.49}
    assert a.choose(scores) == "DEDUP_NO_COUNT"


@pytest.mark.parametrize("action", [a.prepare, a.freeze, a.execute])
def test_no_root_change_to_reset_budget(tmp_path, action):
    with pytest.raises(ValueError, match="FIXED_ROOT_REQUIRED"):
        action(tmp_path)


def test_fit_reservation_preserves_failed_attempt_and_budget(tmp_path):
    for i in range(80):
        a.c.reserve(tmp_path, "fit", str(i))
    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        a.c.reserve(tmp_path, "fit", "new")
    with pytest.raises(ValueError, match="ALREADY_RESERVED"):
        a.c.reserve(tmp_path, "fit", "0")
    assert len(a.c.lines(tmp_path / "fit-ledger.jsonl")) == 80


def test_feature_names_match_frozen_business_meaning():
    fields = a.prior.features.FIELDS
    assert [fields[i] for i in (9, 10, 11, 12, 13)] == [
        "new_forecast_weight",
        "new_realized_earnings_weight",
        "material_change_weight",
        "qualified_event_count",
        "direct_disclosed_weight",
    ]


def test_independent_score_checks_quarters_and_missing_class():
    ps = [
        {"target": "2025-01-02", "predicted": "UP", "probabilities": [0, 0, 1]},
        {"target": "2025-04-01", "predicted": "DOWN", "probabilities": [1, 0, 0]},
    ]
    truth = {"2025-01-02": "UP", "2025-04-01": "DOWN"}
    scores = {
        "correct": 2,
        "accuracy": 1,
        "brier": 0,
        "recall": {"UP": 1, "DOWN": 1, "FLAT": None},
        "quarters": {"1": {"correct": 1}, "2": {"correct": 1}},
        "versus_full_events": {"extra_correct": 0, "wrong_to_right": 0, "right_to_wrong": 0},
    }
    verify.verify_score(scores, ps, ps, truth)
    scores["quarters"]["1"]["correct"] = 2
    with pytest.raises(AssertionError):
        verify.verify_score(scores, ps, ps, truth)
