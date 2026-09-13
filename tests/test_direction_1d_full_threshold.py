"""完整样本与阈值实验的时间隔离、未来答案拒绝、可重复选择及冻结文件保护。"""

from copy import deepcopy
from datetime import datetime, time, timedelta

import numpy as np
import pytest
from app.services import direction_1d_full_threshold as study
from app.services.direction_1d_protocol import ZONE, calendar, digest


def raw(t="2024-09-30", u="2024-10-08"):
    return {
        "fund_code": "synthetic",
        "family": "synthetic",
        "group": "CN_EQUITY",
        "t": t,
        "u": u,
        "kind": "HISTORICAL_RECONSTRUCTION",
        "mature_at": "2024-10-25T08:00:00+08:00",
        "x": [0.0] * 7,
        "y": 1,
        "input_hash": "original_input",
        "label_hash": "original_answer",
    }


def test_holiday_nav_availability_preserves_original_evidence():
    r = raw()
    before = deepcopy(r)
    updated = study.redate([r])[0]
    assert r == before
    assert updated["mature_at"] == "2024-10-09T08:00:00+08:00"
    assert updated["nav_available_at_assumed"] == "2024-10-08T08:00:00+08:00"
    assert updated["legacy_mature_at"] == before["mature_at"]
    assert all(updated[k] == before[k] for k in ("x", "y", "input_hash", "label_hash"))
    assert "NOT_FIRST_PUBLICATION_PROOF" in updated["availability_version"]


@pytest.mark.parametrize(
    "t,u",
    [
        ("2024-12-31", "2025-01-02"),
        ("2026-09-11", "2026-09-14"),
        ("2024-09-30", "2024-10-09"),
        ("2024-09-30", "2024-09-30"),
    ],
)
def test_protected_year_or_wrong_target_rejected(t, u):
    with pytest.raises(ValueError, match="SOURCE_OR_NEXT_SESSION"):
        study.redate([raw(t, u)])


@pytest.mark.parametrize("case", ["duplicate", "forward"])
def test_no_duplicates_or_genuine_predictions_in_history(case):
    rows = [raw(), raw()] if case == "duplicate" else [{**raw(), "kind": "REAL_FORECAST"}]
    with pytest.raises(ValueError, match="SOURCE_OR_NEXT_SESSION"):
        study.redate(rows)


def ready(cutoff):
    row = {**raw(), "mature_at": cutoff.isoformat(), "nav_available_at_assumed": cutoff.isoformat()}
    row.update(
        {k: {"available_at_assumed": cutoff.isoformat()} for k in ("market_input", "specific_input", "activity_input")}
    )
    return row


@pytest.mark.parametrize(
    "field", ["mature_at", "nav_available_at_assumed", "market_input", "specific_input", "activity_input"]
)
def test_each_future_input_or_label_blocks_fit(field):
    cutoff = datetime(2024, 10, 9, 8, tzinfo=ZONE)
    row = ready(cutoff)
    study.check_available([row], cutoff, fit=True)
    late = (cutoff + timedelta(microseconds=1)).isoformat()
    if field.endswith("input"):
        row[field]["available_at_assumed"] = late
    else:
        row[field] = late
    with pytest.raises(ValueError, match="DATA_NOT_AVAILABLE"):
        study.check_available([row], cutoff, fit=True)


def test_question_does_not_require_its_future_answer_at_prediction():
    cutoff = datetime(2024, 10, 8, 8, tzinfo=ZONE)
    row = ready(cutoff)
    row["mature_at"] = (cutoff + timedelta(days=1)).isoformat()
    study.check_available([row], cutoff, fit=False)


def validation_universe():
    days = [d for d in calendar()[0] if d.year == 2023][:92]
    rows = []
    for i in range(1, 91):
        for code in ("synthetic_a", "synthetic_b"):
            rows.append(
                {
                    "fund_code": code,
                    "family": code,
                    "t": str(days[i - 1]),
                    "u": str(days[i]),
                    "y": i % 2,
                    "mature_at": datetime.combine(days[i + 1], time(8), ZONE).isoformat(),
                }
            )
    return rows, datetime.combine(days[-1], time(), ZONE)


def test_validation_keeps_same_target_dates_together_and_only_mature_answers():
    rows, cutoff = validation_universe()
    folds = study.choose_validation(rows, cutoff)
    assert [len(f) for f in folds] == [42, 42, 42]
    sets = [{rows[i]["u"] for i in f} for f in folds]
    assert all(len(s) == 21 for s in sets)
    assert max(sets[0]) < min(sets[1]) and max(sets[1]) < min(sets[2])
    assert not sets[0] & sets[1] and not sets[1] & sets[2]
    assert all(datetime.fromisoformat(rows[i]["mature_at"]) <= cutoff for f in folds for i in f)
    assert len({i for f in folds for i in f}) == 126
    assert rows[-1]["u"] not in sets[-1]


def test_too_few_earlier_validation_days_rejected():
    rows, cutoff = validation_universe()
    with pytest.raises(ValueError, match="VALIDATION_TOO_SHORT"):
        study.choose_validation(rows[:124], cutoff)


@pytest.mark.parametrize("same_index", [True, False])
def test_fit_and_validation_cannot_share_any_target_day(same_index):
    cutoff = datetime(2024, 10, 9, 8, tzinfo=ZONE)
    rows = [ready(cutoff), {**ready(cutoff), "fund_code": "different"}]
    with pytest.raises(ValueError, match="FIT_EXAM_OVERLAP"):
        study.job(rows, [0], [0 if same_index else 1], cutoff)


def threshold_rows(labels):
    return [
        {
            "family": "synthetic",
            "fund_code": "synthetic",
            "t": f"2023-01-{i + 1:02}",
            "u": f"2023-01-{i + 2:02}",
            "y": y,
        }
        for i, y in enumerate(labels)
    ]


@pytest.mark.parametrize(
    "scores,expected",
    [([0.20, 0.30, 0.46, 0.48], 0.45), ([0.46, 0.49, 0.51, 0.52], 0.50), ([0.51, 0.53, 0.70, 0.80], 0.55)],
)
def test_threshold_selected_only_by_validation_accuracy(scores, expected):
    rows = threshold_rows([0, 0, 1, 1])
    a = study.select_threshold(rows, scores)
    assert a["threshold"] == expected
    assert a == study.select_threshold(rows, scores)


def test_equal_validation_accuracy_prefers_point_five():
    assert study.select_threshold(threshold_rows([1, 1]), [0.8, 0.9])["threshold"] == 0.5


def test_tie_at_equal_distance_prefers_lower_threshold():
    assert study.select_threshold(threshold_rows([1, 0]), [0.47, 0.52])["threshold"] == 0.45


def test_threshold_comparison_is_strict_and_can_prefer_higher_cutoff():
    result = study.select_threshold(threshold_rows([0, 0, 1]), [0.45, 0.50, 0.55])
    assert result["weighted_accuracies"]["0.5"] == 1.0
    assert result["threshold"] == 0.5


@pytest.mark.parametrize("scores", [[np.nan], [np.inf], [-0.1], [1.1], []])
def test_invalid_or_missing_scores_do_not_produce_threshold(scores):
    with pytest.raises(ValueError, match="VALIDATION_SCORES_INVALID"):
        study.select_threshold(threshold_rows([1]), scores)


def test_duplicate_share_class_does_not_change_threshold_selection():
    rows = threshold_rows([0, 0, 1, 1])
    scores = [0.2, 0.3, 0.46, 0.48]
    duplicated = [*rows, *[{**r, "fund_code": "synthetic_other_share"} for r in rows]]
    assert (
        study.select_threshold(rows, scores)["threshold"] == study.select_threshold(duplicated, scores * 2)["threshold"]
    )


def minimal_spec(tmp_path):
    study.write_new(tmp_path / "input.json", {"original": True})
    spec = {
        "fingerprint": study.fingerprint(),
        "recipe": study.RECIPE,
        "tree_recipe": study.tree.TREE_RECIPE,
        "activity_recipe": study.activity.FEATURE_RECIPE,
        "model_released": False,
        "protected_years_excluded": [2025, 2026],
        "max_main_fits": 32,
        "max_replay_fits": 32,
        "source": {"source_expires_at": "2099-01-01T00:00:00+08:00"},
        "input_files": {"input.json": study.file_hash(tmp_path / "input.json")},
    }
    return spec


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_main_fits", 34),
        ("max_replay_fits", 64),
        ("model_released", True),
        ("protected_years_excluded", []),
        ("recipe", {}),
        ("fingerprint", {}),
    ],
)
def test_semantic_tampering_rejected_even_with_rehashed_spec(tmp_path, field, value):
    spec = minimal_spec(tmp_path)
    spec[field] = value
    study.write_new(tmp_path / "study.json", spec)
    study.write_new(tmp_path / "study-receipt.json", {"hash": digest(spec)})
    with pytest.raises(ValueError, match="CHANGED"):
        study.verify_inputs(tmp_path)


def test_input_content_tampering_and_repeat_write_rejected(tmp_path):
    spec = minimal_spec(tmp_path)
    study.write_new(tmp_path / "study.json", spec)
    study.write_new(tmp_path / "study-receipt.json", {"hash": digest(spec)})
    study.verify_inputs(tmp_path)
    with pytest.raises(FileExistsError):
        study.write_new(tmp_path / "study.json", {})
    (tmp_path / "input.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="INPUT_CHANGED"):
        study.verify_inputs(tmp_path)


def test_selection_rejects_answer_after_outer_cutoff_before_scoring(monkeypatch):
    rows, cutoff = validation_universe()
    folds = study.choose_validation(rows, cutoff)
    schedule = {"2024Q1": {"train_as_of": cutoff.isoformat()}}
    for k, indexes in enumerate(folds, 1):
        schedule[f"2024Q1-V{k}"] = {"exam_indexes": indexes}
    rows[folds[-1][-1]]["mature_at"] = (cutoff + timedelta(seconds=1)).isoformat()
    monkeypatch.setattr(study, "QUARTERS", ("2024Q1",))
    with pytest.raises(ValueError, match="VALIDATION_ANSWER_TOO_LATE"):
        study.selections(rows, schedule, {})
