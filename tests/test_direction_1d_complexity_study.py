"""复杂度实验的数值恢复、未来答案隔离、原模型保护与选择规则回归。"""

from copy import deepcopy
from datetime import datetime, time

import numpy as np
import pytest
from app.services import direction_1d_complexity_study as study
from app.services.direction_1d_protocol import ZONE, calendar, digest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def question(i, label):
    return {
        "fund_code": "synthetic",
        "family": "synthetic",
        "t": f"2023-01-{i + 1:02}",
        "u": f"2023-01-{i + 2:02}",
        "y": label,
    }


@pytest.mark.parametrize("winner", study.BRANCHES)
def test_selection_uses_same_earlier_questions(winner):
    rows = [question(i, y) for i, y in enumerate([0, 0, 1, 1])]
    scores = {b: [0.9, 0.9, 0.1, 0.1] for b in study.BRANCHES}
    scores[winner] = [0.1, 0.1, 0.9, 0.9]
    assert study.select_branch(rows, scores)["branch"] == winner


def test_equal_accuracy_keeps_existing_model_and_strict_half_threshold():
    rows = [question(0, 0), question(1, 1)]
    scores = {b: [0.5, 0.51] for b in study.BRANCHES}
    report = study.select_branch(rows, scores)
    assert report["branch"] == study.REFERENCE
    assert report["reports"][study.REFERENCE]["accuracy"] == 1


@pytest.mark.parametrize("scores", [[0.2], [0.2, float("nan")], [0.2, 1.01], [[0.2], [0.8]]])
def test_missing_or_bad_scores_cannot_drop_hard_questions(scores):
    with pytest.raises(ValueError, match="SCORES_INVALID"):
        study.score_summary([question(0, 0), question(1, 1)], scores)


def test_single_class_diagnostic_does_not_invent_other_recall():
    report = study.score_summary([question(0, 1)], [0.8])
    assert report["up_recall"] == 1
    assert report["non_up_recall"] is None


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(20260912)
    days = [d for d in calendar()[0] if 2021 <= d.year <= 2023]
    values = rng.normal(size=(540, 12))
    rows = []
    for i, x in enumerate(values):
        t, u, mature = days[i : i + 3]
        at = datetime.combine(t, time(18), ZONE).isoformat()
        rows.append(
            {
                "fund_code": "synthetic",
                "family": "synthetic",
                "t": str(t),
                "u": str(u),
                "group": "CN_EQUITY",
                "x": x[:7].tolist(),
                "y": int(x[0] * x[7] > 0),
                "mature_at": datetime.combine(mature, time(8), ZONE).isoformat(),
                "nav_available_at_assumed": datetime.combine(u, time(8), ZONE).isoformat(),
                "market_input": {"x": x[7:9].tolist(), "available_at_assumed": at},
                "specific_input": {"x": x[9:11].tolist(), "available_at_assumed": at},
                "activity_input": {"x": x[11:].tolist(), "available_at_assumed": at},
            }
        )
    fit_rows, checked = rows[:500], rows[510:]
    w = study.weights(fit_rows)
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(values[:500], sample_weight=w)
        estimator = HistGradientBoostingClassifier(**study.tree.TREE_RECIPE).fit(
            scaler.transform(values[:500]), [r["y"] for r in fit_rows], sample_weight=w
        )
    control = {
        "kind": "RESEARCH_ONLY",
        "candidate": study.activity.CANDIDATE,
        "target_definition": "UNIT_NAV_DIRECTION_V1",
        "model_released": False,
        "features": study.activity.FEATURES,
        "feature_recipe": study.activity.FEATURE_RECIPE,
        "recipe": study.tree.TREE_RECIPE,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "baseline": float(estimator._baseline_prediction[0, 0]),
        "trees": study.tree.export_trees(estimator),
        "fit_hash": digest(fit_rows),
        "weight_hash": digest(w.tolist()),
        "fit_count": 500,
        "distinct_dates": 500,
        "classes": {str(c): sum(r["y"] == c for r in fit_rows) for c in (0, 1)},
        "train_as_of": datetime.combine(days[505], time(), ZONE).isoformat(),
    }
    old = deepcopy(control)
    models = {b: study.fit(fit_rows, checked, control, {"cohort_id": "synthetic"}, b) for b in study.RECIPES}
    assert control == old
    return fit_rows, checked, control, models


@pytest.mark.parametrize("branch", study.RECIPES)
def test_both_depths_restore_native_predictions_and_preserve_scales(fitted, branch):
    rows, checked, control, models = fitted
    m = models[branch]
    assert max(m["restore_max_score_diff"].values()) <= 1e-12
    assert m["mean"] == control["mean"] and m["scale"] == control["scale"]
    assert m["weight_hash"] == control["weight_hash"]
    assert np.isfinite(study.predict(m, rows + checked)).all()
    assert max(n["depth"] for tree in m["trees"] for n in tree) <= study.RECIPES[branch]["max_depth"]


def test_inference_never_reads_answers(fitted):
    _, checked, _, models = fitted
    without = [{k: v for k, v in r.items() if k not in ("y", "mature_at")} for r in checked]
    for model in models.values():
        np.testing.assert_array_equal(study.predict(model, checked), study.predict(model, without))


@pytest.mark.parametrize("mutation", ["scale", "rounds", "feature", "cycle", "recipe", "released", "nan"])
def test_corrupt_model_fails_closed(fitted, mutation):
    model = deepcopy(fitted[3]["DEPTH3"])
    if mutation == "scale":
        model["scale"][0] = 0
    if mutation == "rounds":
        model["trees"].pop()
    if mutation == "recipe":
        model["recipe"]["max_depth"] = 9
    if mutation == "released":
        model["model_released"] = True
    if mutation == "nan":
        model["baseline"] = float("nan")
    if mutation in ("feature", "cycle"):
        n = next(n for nodes in model["trees"] for n in nodes if not n["leaf"])
        n["feature" if mutation == "feature" else "left"] = 12 if mutation == "feature" else 0
    with pytest.raises(ValueError, match="COMPLEXITY_MODEL"):
        study.predict(model, fitted[1])


def test_future_answer_rejected_before_any_fit(fitted, monkeypatch):
    rows, checked, control, _ = deepcopy(fitted)
    rows[-1]["mature_at"] = "2026-09-14T08:00:00+08:00"
    control["fit_hash"] = digest(rows)

    def forbidden(*args, **kwargs):
        pytest.fail("future answer reached estimator.fit")

    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", forbidden)
    with pytest.raises(ValueError, match="DATA_NOT_AVAILABLE"):
        study.fit(rows, checked, control, {"cohort_id": "synthetic"}, "DEPTH3")


def test_changed_members_rejected_before_fit(fitted):
    rows, checked, control, _ = fitted
    with pytest.raises(ValueError, match="FIT_MEMBERS"):
        study.fit(rows[:-1], checked, control, {"cohort_id": "synthetic"}, "DEPTH1")


def test_selection_rejects_answers_later_than_quarter(monkeypatch):
    row = {**question(0, 1), "mature_at": "2026-09-14T08:00:00+08:00"}
    schedule = {q: {"train_as_of": "2024-01-01T00:00:00+08:00"} for q in study.QUARTERS}
    schedule.update({q + f"-V{k}": {"exam_indexes": [0]} for q in study.QUARTERS for k in (1, 2, 3)})
    models = {k: {b: {} for b in study.BRANCHES} for k in schedule}
    monkeypatch.setattr(study, "predict", lambda m, r: np.full(len(r), 0.8))
    with pytest.raises(ValueError, match="SELECTION_ANSWER_TOO_LATE"):
        study.selections([row], schedule, models)


@pytest.mark.parametrize("year", [2025, 2026])
def test_protected_year_rejected_before_schedule_use(year):
    schedule = {q + s: {} for q in study.QUARTERS for s in ("", "-V1", "-V2", "-V3")}
    row = {**question(0, 0), "u": f"{year}-01-02"}
    with pytest.raises(ValueError, match="PROTECTED_YEAR"):
        study.validate_schedule([row], schedule, [])


def test_missing_time_windows_rejected():
    with pytest.raises(ValueError, match="SCHEDULE"):
        study.validate_schedule([question(0, 0)], {}, [])
