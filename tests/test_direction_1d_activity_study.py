"""成交活跃度窗口与12维树的边界；仅合成数据一次工程拟合，无真实基金试选。"""

from copy import deepcopy
from datetime import date, datetime, timedelta

import numpy as np
import pytest
from app.services import direction_1d_activity_study as study
from app.services.direction_1d_protocol import ZONE, calendar, digest


@pytest.fixture
def sample_window():
    days = calendar()[0][10:32]
    series = {
        str(d): {
            "index_code": study.acquisition.INDEX,
            "date": str(d),
            "close": "100",
            "amount": "100" if i < 20 else ("200" if i == 20 else "9999"),
        }
        for i, d in enumerate(days)
    }
    rows = [{"fund_code": "synthetic", "t": str(days[20]), "u": str(days[21]), "y": 1}]
    return rows, series, days


def test_denominator_excludes_t_and_never_reads_u(sample_window):
    rows, series, days = sample_window
    original = deepcopy(rows)
    result = study.attach(rows, series)
    extra = result[0]["activity_input"]
    assert extra["x"] == [1.0]
    assert extra["window_dates"] == [str(d) for d in days[:21]]
    assert extra["available_at_assumed"] == str(days[20]) + "T18:00:00+08:00"
    assert (date.fromisoformat(rows[0]["t"]) - days[0]).days > 20
    series[str(days[21])]["amount"] = "1"
    assert study.attach(rows, series) == result
    assert rows == original and study.control_rows(result) == original


def test_common_unit_scaling_does_not_change_ratio(sample_window):
    rows, series, _ = sample_window
    before = study.attach(rows, series)[0]["activity_input"]["x"]
    for row in series.values():
        row["amount"] = str(float(row["amount"]) * 1000)
    assert study.attach(rows, series)[0]["activity_input"]["x"] == before


def test_missing_intermediate_session_is_not_skipped(sample_window):
    rows, series, days = sample_window
    del series[str(days[7])]
    with pytest.raises(ValueError, match="WINDOW_INCOMPLETE"):
        study.attach(rows, series)


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "Infinity"])
def test_nonpositive_or_nonfinite_amount_stops_feature(sample_window, value):
    rows, series, days = sample_window
    series[str(days[3])]["amount"] = value
    with pytest.raises(ValueError, match="WINDOW_VALUE_INVALID"):
        study.attach(rows, series)


@pytest.mark.parametrize("case", ["code", "date"])
def test_window_identity_cannot_be_replaced(sample_window, case):
    rows, series, days = sample_window
    series[str(days[3])]["index_code" if case == "code" else "date"] = "wrong"
    with pytest.raises(ValueError, match="WINDOW_IDENTITY_INVALID"):
        study.attach(rows, series)


@pytest.mark.parametrize("t", ["2024-01-07", "2025-01-02", "2026-09-11", "2021-01-04"])
def test_protected_or_short_window_rejected(sample_window, t):
    rows, series, _ = sample_window
    rows[0]["t"] = t
    with pytest.raises(ValueError, match="WINDOW_DATE_INVALID"):
        study.attach(rows, series)


def test_same_day_deadline_rejects_close_input(sample_window, tmp_path, monkeypatch):
    rows, series, _ = sample_window
    (tmp_path / "main").mkdir()
    (tmp_path / "base").mkdir()
    rows[0]["u"] = rows[0]["t"]
    study.write_new(tmp_path / "base/common-exam.json", rows)
    study.write_new(
        tmp_path / "main/models.json",
        {q: {"fit_hash": digest([]), "train_as_of": "2024-01-01T08:00:00+08:00"} for q in (*study.QUARTERS, "FINAL")},
    )
    monkeypatch.setattr(study.sector, "selected_fit", lambda *a: [])
    with pytest.raises(ValueError, match="EXAM_NOT_AVAILABLE"):
        study.inputs(tmp_path, series)


@pytest.fixture(scope="module")
def fitted():
    # 12维合成数据中标签仅依赖末维，验证第12维分裂能保存和恢复；无真实数据读取。
    rng = np.random.default_rng(12)
    values = rng.normal(size=(500, 12))
    values[:, -1] *= 0.01
    rows = [
        {
            "fund_code": "synthetic",
            "family": "synthetic",
            "group": "CN_EQUITY",
            "t": str(d),
            "mature_at": datetime.combine(d + timedelta(days=3), datetime.min.time(), ZONE).isoformat(),
            "x": x[:7].tolist(),
            "market_input": {"x": x[7:9].tolist()},
            "specific_input": {"x": x[9:11].tolist()},
            "activity_input": {
                "x": [float(x[-1])],
                "available_at_assumed": datetime.combine(d, datetime.min.time(), ZONE).isoformat(),
            },
            "y": int(x[-1] > 0),
        }
        for d, x in zip(calendar()[0][61:561], values, strict=True)
    ]
    w = study.weights(rows)
    control = {
        "candidate": study.tree.CANDIDATE,
        "kind": "RESEARCH_ONLY",
        "target_definition": "UNIT_NAV_DIRECTION_V1",
        "train_as_of": "2024-01-01T08:00:00+08:00",
        "model_released": False,
        "features": study.sector.feature_names("SPECIFIC11"),
        "recipe": study.tree.TREE_RECIPE,
        "mean": values[:, :11].mean(axis=0).tolist(),
        "scale": values[:, :11].std(axis=0).tolist(),
        "baseline": 0.0,
        "trees": [
            [
                {
                    "leaf": True,
                    "value": 0.0,
                    "feature": 0,
                    "threshold": 0.0,
                    "left": 0,
                    "right": 0,
                    "depth": 0,
                    "count": 500,
                }
            ]
            for _ in range(100)
        ],
        "fit_hash": digest(study.control_rows(rows)),
        "fit_count": len(rows),
        "weight_hash": digest(w.tolist()),
        "distinct_dates": len(rows),
        "classes": {str(y): sum(r["y"] == y for r in rows) for y in (0, 1)},
    }
    spec = {
        "cohort_id": "SYNTHETIC_ONLY",
        "fit_hashes": {"FINAL": digest(rows)},
        "weight_hashes": {"FINAL": digest(w.tolist())},
    }
    model = study.fit(rows, control, spec, "FINAL", rows[:40])
    return rows, control, spec, model


def test_twelfth_dimension_json_round_trip_matches_estimator(fitted, tmp_path):
    rows, control, _, model = fitted
    assert max(model["restore_max_score_diff"].values()) <= 1e-12
    assert any(n["feature"] == 11 and not n["leaf"] for tree in model["trees"] for n in tree)
    assert model["mean"][:11] == control["mean"] and model["scale"][:11] == control["scale"]
    assert model["weight_hash"] == control["weight_hash"] and len(model["features"]) == 12
    study.write_new(tmp_path / "model.json", model)
    assert np.array_equal(study.predict(model, rows), study.predict(study.read(tmp_path / "model.json"), rows))
    assert not len(study.predict(model, []))


@pytest.mark.parametrize("case", ["rounds", "dimension", "nan", "feature", "cycle", "recipe", "target", "release"])
def test_invalid_model_is_never_scored(fitted, case):
    rows, _, _, source = fitted
    model = deepcopy(source)
    if case == "rounds":
        model["trees"].pop()
    elif case == "dimension":
        model["scale"].pop()
    elif case == "nan":
        model["trees"][0][0]["value"] = float("nan")
    elif case == "recipe":
        model["feature_recipe"]["unit"] = "PERCENT"
    elif case == "target":
        model["target_definition"] = "TOTAL_RETURN"
    elif case == "release":
        model["model_released"] = True
    else:
        node = next(n for tree in model["trees"] for n in tree if not n["leaf"])
        node["feature" if case == "feature" else "left"] = 12 if case == "feature" else 0
    with pytest.raises(ValueError, match="ACTIVITY_MODEL_"):
        study.predict(model, rows[:2])


@pytest.mark.parametrize("case", ["fit", "weight", "input_late", "answer_late"])
def test_changed_or_unavailable_fit_cannot_train(fitted, monkeypatch, case):
    a, b, c, _ = fitted
    rows, control, spec = deepcopy(a), deepcopy(b), deepcopy(c)
    monkeypatch.setattr(study.HistGradientBoostingClassifier, "fit", lambda *a, **k: pytest.fail("应在拟合之前拒绝"))
    if case == "fit":
        rows.pop()
    elif case == "weight":
        control["weight_hash"] = "bad"
    else:
        if case == "input_late":
            rows[0]["activity_input"]["available_at_assumed"] = "2024-02-01T08:00:00+08:00"
        else:
            rows[0]["mature_at"] = "2024-02-01T08:00:00+08:00"
        spec["fit_hashes"]["FINAL"] = digest(rows)
        control["fit_hash"] = digest(study.control_rows(rows))
    with pytest.raises(ValueError, match="ACTIVITY_FIT_"):
        study.fit(rows, control, spec, "FINAL", [])


def test_nonfinite_added_feature_not_imputed(fitted):
    a, _, _, model = fitted
    rows = deepcopy(a[:1])
    rows[0]["activity_input"]["x"] = [float("nan")]
    with pytest.raises(ValueError, match="ACTIVITY_FEATURE_INVALID"):
        study.predict(model, rows)


@pytest.mark.parametrize("replay", [False, True])
def test_budget_cannot_be_repeated(tmp_path, monkeypatch, replay):
    (tmp_path / ("replay" if replay else "main")).mkdir()
    monkeypatch.setattr(study, "verify_inputs", lambda _: {})
    monkeypatch.setattr(study, "verify", lambda _: {})
    monkeypatch.setattr(study, "fit", lambda *a: pytest.fail("不能重复预算"))
    with pytest.raises(FileExistsError):
        study.run(tmp_path, replay=replay)


def test_frozen_input_tampering_fails(tmp_path):
    study.write_new(tmp_path / "input.json", {"count": 5704})
    spec = {
        "fingerprint": study.fingerprint(),
        "tree_recipe": study.tree.TREE_RECIPE,
        "feature_recipe": study.FEATURE_RECIPE,
        "features": study.FEATURES,
        "candidate": study.CANDIDATE,
        "reference": study.REFERENCE,
        "model_released": False,
        "source": {"source_expires_at": (datetime.now(ZONE) + timedelta(days=1)).isoformat()},
        "input_files": {"input.json": study.file_hash(tmp_path / "input.json")},
    }
    study.write_new(tmp_path / "study.json", spec)
    study.write_new(tmp_path / "study-receipt.json", {"hash": digest(spec)})
    study.verify_inputs(tmp_path)
    (tmp_path / "input.json").write_text('{"count":5703}', encoding="utf-8")
    with pytest.raises(ValueError, match="FROZEN_INPUT_CHANGED"):
        study.verify_inputs(tmp_path)
