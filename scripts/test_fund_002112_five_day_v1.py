"""五日目标的时间边界、重叠评价与固定准入测试；测试不执行真实模型拟合。"""

import copy
from datetime import date, timedelta

import pytest

from scripts import fund_002112_five_day_v1 as f


def sample():
    sessions = ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13"]
    nav = {d: {"unit_nav": str(i + 1), "available_at": d + "T23:00:00+08:00"} for i, d in enumerate(sessions)}
    row = {
        "base": sessions[0],
        "target": sessions[1],
        "as_of": sessions[0] + "T08:00:00+08:00",
        "label_mature_at": nav[sessions[1]]["available_at"],
        "groups": {"N": [0] * 8, "NE": [0] * 7},
        "E": [None] * 20,
        "ED": [None] * 20,
        "trigger": False,
        "trigger_decay": False,
    }
    return row, sessions, nav


def test_five_sessions_cross_holiday_preserve_input_and_source():
    row, sessions, nav = sample()
    before = copy.deepcopy(row)
    rows, pending = f.make_rows([row], sessions, nav, "2026-10-13")
    assert rows[0]["target"] == "2026-10-12"
    assert rows[0]["as_of"] == before["as_of"]
    assert rows[0]["groups"] == before["groups"]
    assert rows[0]["E"] == before["E"]
    assert rows[0]["source_row_sha256"] == f.c.io.digest(before)
    assert rows[0]["label_mature_at"] == "2026-10-12T23:00:00+08:00"
    assert row == before and not pending


def test_tail_is_retained_but_neither_prediction_nor_known_answer():
    row, sessions, nav = sample()
    rows, pending = f.make_rows([row], sessions, nav)
    assert not rows
    assert len(pending) == 1
    assert pending[0]["label_mature_at"] is None
    assert pending[0]["state"] == "OUTSIDE_FROZEN_HISTORY_NOT_A_PREDICTION"


def test_late_base_version_delays_five_day_answer():
    row, sessions, nav = sample()
    nav[row["base"]]["available_at"] = "2026-10-22T08:00:00+08:00"
    rows, _ = f.make_rows([row], sessions, nav, "2026-10-13")
    assert rows[0]["label_mature_at"] == "2026-10-22T08:00:00+08:00"


@pytest.mark.parametrize("problem", ["missing", "zero", "negative"])
def test_invalid_label_nav_is_not_removed_or_imputed(problem):
    row, sessions, nav = sample()
    if problem == "missing":
        del nav["2026-10-12"]
    else:
        nav["2026-10-12"]["unit_nav"] = "0" if problem == "zero" else "-1"
    with pytest.raises(ValueError, match="LABEL_NAV_MISSING_OR_INVALID"):
        f.make_rows([row], sessions, nav, "2026-10-13")


def test_unknown_calendar_cannot_guess_natural_day_endpoint():
    row, sessions, nav = sample()
    with pytest.raises(ValueError, match="CALENDAR_END_UNKNOWN"):
        f.make_rows([row], sessions[:5], nav)


def test_maturity_equal_to_update_is_excluded():
    rows = [
        {"target": "2024-12-20", "label_mature_at": "2024-12-24T08:00:00+08:00"},
        {"target": "2024-12-23", "label_mature_at": "2024-12-25T08:00:00+08:00"},
        {
            "target": "2025-01-02",
            "base": "2024-12-25",
            "as_of": "2024-12-25T08:00:00+08:00",
            "label_mature_at": "2025-01-03T08:00:00+08:00",
        },
    ]
    assert f.train.calendar(rows, "2025")[0]["training"] == [0]


def test_selective_abstention_retains_full_denominator_and_down_days():
    truth = {"a": "UP", "b": "DOWN", "c": "DOWN"}
    preds = [
        {"target": "a", "predicted": "UP", "probabilities": [0.3, 0, 0.7]},
        {"target": "b", "predicted": "UP", "probabilities": [0.49, 0, 0.51]},
        {"target": "c", "predicted": "UP", "probabilities": [0.3, 0, 0.7]},
    ]
    s = f.selective_metrics(preds, truth)
    assert s["coverage"] == 2 / 3 and s["judged_accuracy"] == 0.5
    assert s["correct_over_all_dates"] == 1 / 3
    assert s["down_not_caught_including_abstention"] == 2
    assert s["abstained_targets"] == ["b"]


def test_block_interval_reproducible_and_zero_gain_not_positive():
    assert f.forward.bootstrap_difference([0] * 243) == [0, 0]
    values = ([1] * 20 + [-1] * 20) * 4
    assert f.forward.bootstrap_difference(values) == f.forward.bootstrap_difference(values)
    with pytest.raises(ValueError, match="WAIT_FOR_120"):
        f.forward.bootstrap_difference([0] * 119)


def fake_scores():
    base = {
        "days": 243,
        "correct": 130,
        "quarters": {str(q): {"correct": 30} for q in range(1, 5)},
        "recall": {"DOWN": 0.6},
        "brier": 0.5,
        "paired_block_ci95": [0, 0],
        "all_nonoverlap_phases": {str(q): {"extra_correct": 0} for q in range(5)},
    }
    good = {**copy.deepcopy(base), "correct": 142, "brier": 0.48, "paired_block_ci95": [0.001, 0.1]}
    return {"B0": base, "B1": good, "B2": copy.deepcopy(base), "B3": copy.deepcopy(base), "always_up": {"correct": 132}}


@pytest.mark.parametrize("gate", ["ci", "quarters", "recall", "brier", "phases", "always_up", "coverage", "gain"])
def test_each_promotion_gate_cannot_be_ignored(gate):
    scores = fake_scores()
    assert f.select(scores)["selected"] == "B1"
    one = scores["B1"]
    if gate == "ci":
        one["paired_block_ci95"] = [-0.001, 0.1]
    elif gate == "quarters":
        one["quarters"]["1"]["correct"] = one["quarters"]["2"]["correct"] = 29
    elif gate == "recall":
        one["recall"]["DOWN"] = 0.54
    elif gate == "brier":
        one["brier"] = 0.5001
    elif gate == "phases":
        one["all_nonoverlap_phases"]["0"]["extra_correct"] = -1
        one["all_nonoverlap_phases"]["1"]["extra_correct"] = -1
    elif gate == "always_up":
        scores["always_up"]["correct"] = 142
    elif gate == "coverage":
        one["days"] = 242
    else:
        one["correct"] = 134
    decision = f.select(scores)
    assert decision["selected"] == "B0" and not decision["future_observation_eligible"]


def test_nonoverlap_phases_all_reported():
    days = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(125)]
    rows = [
        {"target": d, "base": d, "base_session_index": i, "groups": {"N": [0] * 8}, "trigger": False}
        for i, d in enumerate(days)
    ]
    nav = {d: {"unit_nav": "1"} for d in days}
    preds = [{"target": d, "probabilities": [0, 1, 0], "predicted": "FLAT"} for d in days]
    scores = f.score({g: copy.deepcopy(preds) for g in f.GROUPS}, rows, nav)
    assert set(scores["B0"]["all_nonoverlap_phases"]) == set("01234")
    assert sum(v["days"] for v in scores["B0"]["all_nonoverlap_phases"].values()) == 125
    assert all(v["days"] == 25 for v in scores["B0"]["all_nonoverlap_phases"].values())


def test_experiment_cannot_move_to_reset_budget(tmp_path):
    with pytest.raises(ValueError, match="FIXED_EXPERIMENT_ROOT_REQUIRED"):
        f.execute(tmp_path)


def test_attempt_reservation_counts_failures_and_rejects_duplicate(tmp_path):
    f.c.reserve(tmp_path, "fit", "attempt-0")
    with pytest.raises(ValueError, match="ALREADY_RESERVED"):
        f.c.reserve(tmp_path, "fit", "attempt-0")
    for i in range(1, 80):
        f.c.reserve(tmp_path, "fit", f"attempt-{i}")
    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        f.c.reserve(tmp_path, "fit", "attempt-80")
