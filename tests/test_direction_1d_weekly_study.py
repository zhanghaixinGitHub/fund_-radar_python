"""周日截点、成熟后再训练、504日窗口及真实两套树JSON恢复；仅合成数据拟合。"""

from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pytest
from app.services import direction_1d_weekly_study as study
from app.services.direction_1d_protocol import ZONE, calendar, digest


@pytest.mark.parametrize(
    "target,expected",
    [
        ("2024-01-02", "2023-12-31"),
        ("2024-02-19", "2024-02-18"),
        ("2024-10-08", "2024-10-06"),
        ("2024-12-31", "2024-12-29"),
    ],
)
def test_target_week_handles_holiday_and_year_boundary(target, expected):
    assert study.week_cutoff(target).isoformat() == expected + "T12:00:00+08:00"


@pytest.mark.parametrize("target", ["2023-12-29", "2025-01-02", "2026-09-14", "2024-01-07", "2024-10-01"])
def test_protected_or_non_session_exam_rejected(target):
    with pytest.raises(ValueError, match="EXAM_DATE_INVALID"):
        study.week_cutoff(target)


def test_maturity_and_input_arrival_both_gate_weekly_fit():
    cutoff = study.week_cutoff("2024-01-08")
    raw = [
        {
            "fund_code": str(i),
            "group": "CN_EQUITY",
            "t": "2024-01-04",
            "mature_at": (cutoff + timedelta(seconds=1 if i == 1 else 0)).isoformat(),
        }
        for i in range(3)
    ]
    available = {(r["fund_code"], r["t"]): cutoff + timedelta(seconds=1 if i == 2 else 0) for i, r in enumerate(raw)}
    assert study.select_week(raw, available, cutoff) == [raw[0]]


def test_504_trading_days_excludes_old_rows_and_keeps_family_duplicates():
    days = calendar()[0][:600]
    raw = [
        {
            "fund_code": str(i),
            "group": "CN_EQUITY",
            "family": "same",
            "t": str(d),
            "mature_at": "2023-12-01T08:00:00+08:00",
        }
        for d in days
        for i in range(2)
    ]
    cutoff = study.week_cutoff("2024-01-02")
    available = {(r["fund_code"], r["t"]): cutoff for r in raw}
    result = study.select_week(raw, available, cutoff)
    assert len(result) == 504 * 2
    assert {r["t"] for r in result} == {str(d) for d in days[-504:]}
    assert np.allclose(study.original.weights(result), 0.5)


@pytest.mark.parametrize("field", ["mature_at", "nav", "market_input", "specific_input", "activity_input"])
def test_each_information_timestamp_must_be_known(field):
    cutoff = study.week_cutoff("2024-01-08")
    row = {"fund_code": "synthetic", "t": "2024-01-04", "u": "2024-01-05", "mature_at": cutoff.isoformat()}
    row.update(
        {k: {"available_at_assumed": cutoff.isoformat()} for k in ("market_input", "specific_input", "activity_input")}
    )
    available = {("synthetic", row["t"]): cutoff}
    study.ensure_known([row], available, cutoff)
    late = cutoff + timedelta(microseconds=1)
    if field == "nav":
        available["synthetic", row["t"]] = late
    elif field == "mature_at":
        row[field] = late.isoformat()
    else:
        row[field]["available_at_assumed"] = late.isoformat()
    with pytest.raises(ValueError, match="FIT_NOT_AVAILABLE"):
        study.ensure_known([row], available, cutoff)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(34)
    values = rng.normal(size=(500, 12))
    rows = []
    for d, x in zip(calendar()[0][61:561], values, strict=True):
        stamp = str(d) + "T18:00:00+08:00"
        rows.append(
            {
                "fund_code": "synthetic",
                "family": "synthetic",
                "group": "CN_EQUITY",
                "t": str(d),
                "mature_at": (datetime.combine(d, datetime.min.time(), ZONE) + timedelta(days=5)).isoformat(),
                "x": x[:7].tolist(),
                "y": int(x[0] + x[-1] > 0),
                "market_input": {"x": x[7:9].tolist(), "available_at_assumed": stamp},
                "specific_input": {"x": x[9:11].tolist(), "available_at_assumed": stamp},
                "activity_input": {"x": [float(x[-1])], "available_at_assumed": stamp},
            }
        )
    control_rows = study.activity.control_rows(rows)
    week = {
        "control_fit_hash": digest(control_rows),
        "fit_hash": digest(rows),
        "weight_hash": digest(study.original.weights(rows).tolist()),
        "train_as_of": "2023-12-31T12:00:00+08:00",
    }
    spec = {"cohort_id": "SYNTHETIC_ONLY"}
    control = study.fit_control(control_rows, week, spec, rows[:40])
    model = study.activity.fit(
        rows,
        control,
        {**spec, "fit_hashes": {"week": week["fit_hash"]}, "weight_hashes": {"week": week["weight_hash"]}},
        "week",
        rows[:40],
    )
    return rows, week, spec, control, model


def test_both_weekly_models_restore_and_share_scale(fitted, tmp_path):
    rows, week, spec, control, model = fitted
    study.verify_model(control, study.activity.control_rows(rows), week, spec)
    study.verify_model(model, rows, week, spec, control=control)
    assert model["mean"][:11] == control["mean"] and model["scale"][:11] == control["scale"]
    for i, (m, predictor) in enumerate(((control, study.tree.predict), (model, study.activity.predict))):
        assert max(m["restore_max_score_diff"].values()) <= 1e-12
        study.write_new(tmp_path / f"model{i}.json", m)
        assert np.array_equal(predictor(m, rows), predictor(study.read(tmp_path / f"model{i}.json"), rows))


@pytest.mark.parametrize("case", ["scale", "weight", "cutoff", "control", "restore", "recipe"])
def test_tampered_model_cannot_verify(fitted, case):
    rows, week, spec, control, source = fitted
    model = deepcopy(source)
    if case == "scale":
        model["mean"][0] += 1
    elif case == "weight":
        model["weight_hash"] = "wrong"
    elif case == "cutoff":
        model["train_as_of"] = "2024-12-31T12:00:00+08:00"
    elif case == "control":
        model["control_model_hash"] = "wrong"
    elif case == "restore":
        model["restore_max_score_diff"]["fit"] = float("nan")
    else:
        model["recipe"] = {**model["recipe"], "max_depth": 3}
    with pytest.raises(ValueError):
        study.verify_model(model, rows, week, spec, control=control)


@pytest.mark.parametrize("case", ["fit", "late", "small"])
def test_invalid_control_fit_rejected_before_classifier(fitted, monkeypatch, case):
    a, b, spec, _, _ = fitted
    rows, week = study.activity.control_rows(deepcopy(a)), deepcopy(b)
    monkeypatch.setattr(study.HistGradientBoostingClassifier, "fit", lambda *a, **k: pytest.fail("不得拟合"))
    if case == "fit":
        rows.pop()
    else:
        if case == "late":
            rows[0]["mature_at"] = "2024-01-02T08:00:00+08:00"
        else:
            rows = rows[:30]
        week["control_fit_hash"] = digest(rows)
        week["weight_hash"] = digest(study.original.weights(rows).tolist())
    with pytest.raises(ValueError):
        study.fit_control(rows, week, spec, [])


@pytest.mark.parametrize("replay", [False, True])
def test_budget_cannot_be_restarted(tmp_path, monkeypatch, replay):
    (tmp_path / ("replay" if replay else "main")).mkdir()
    monkeypatch.setattr(study, "verify_inputs", lambda _: {})
    monkeypatch.setattr(study, "verify", lambda _: {})
    monkeypatch.setattr(study, "fit_control", lambda *a: pytest.fail("不得重复预算"))
    with pytest.raises(FileExistsError):
        study.run(tmp_path, replay=replay)


def test_predictions_preserve_all_old_conclusions_and_threshold():
    old = [
        {
            "fund_code": "synthetic",
            "t": "2023-12-29",
            "u": "2024-01-02",
            "y": 1,
            "kind": "HISTORICAL_RECONSTRUCTION",
            "scores": {"TREE11": 0.7},
            "directions": {"TREE11": 1},
        }
    ]
    before = deepcopy(old)
    scores = {c: {("synthetic", "2023-12-29"): 0.5 if i == 0 else 0.6} for i, c in enumerate(study.CANDIDATES)}
    result = study.expected_predictions(old, scores)[0]
    assert old == before and result["kind"] == old[0]["kind"]
    assert result["scores"]["TREE11"] == 0.7
    assert result["directions"][study.CANDIDATES[0]] == 0
    assert result["directions"][study.CANDIDATES[1]] == 1
    assert result["weekly_cutoff"] == "2023-12-31T12:00:00+08:00"


@pytest.mark.parametrize("case", ["input", "budget", "recipe", "code", "expired"])
def test_input_seal_and_frozen_budget_must_match(tmp_path, monkeypatch, case):
    monkeypatch.setattr(study, "fingerprint", lambda: {"code": "known"})
    study.write_new(tmp_path / "schedule.json", {"2023-12-31": {}})
    spec = {
        "fingerprint": study.fingerprint(),
        "candidates": list(study.CANDIDATES),
        "pairs": [list(p) for p in study.PAIRS],
        "schedule_recipe": deepcopy(study.SCHEDULE_RECIPE),
        "tree_recipe": study.tree.TREE_RECIPE,
        "activity_recipe": study.activity.FEATURE_RECIPE,
        "model_released": False,
        "source": {"source_expires_at": (datetime.now(ZONE) + timedelta(days=1)).isoformat()},
        "input_files": {"schedule.json": study.file_hash(tmp_path / "schedule.json")},
        "weeks": ["2023-12-31"],
        "max_main_fits": 2,
        "max_replay_fits": 2,
    }
    if case == "budget":
        spec["max_main_fits"] += 2
    elif case == "recipe":
        spec["schedule_recipe"]["threshold"] = 0.6
    elif case == "code":
        spec["fingerprint"] = {"code": "wrong"}
    elif case == "expired":
        spec["source"]["source_expires_at"] = (datetime.now(ZONE) - timedelta(days=1)).isoformat()
    else:
        (tmp_path / "schedule.json").write_text("{}", encoding="utf-8")
    study.write_new(tmp_path / "study.json", spec)
    study.write_new(tmp_path / "study-receipt.json", {"hash": digest(spec)})
    with pytest.raises(ValueError):
        study.verify_inputs(tmp_path)
