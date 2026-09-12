"""新一日幅度的时间/数据边界与12维树恢复；仅合成数据工程拟合，不试选真实基金。"""

from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pytest
from app.services import direction_1d_nav1_study as study
from app.services.direction_1d_protocol import ZONE, calendar, digest


def history():
    return {
        "funds": [
            {
                "fund_code": "f",
                "rows": [
                    {"date": "2024-01-05", "nav": "1.00", "ann_date": "2024-01-06", "source_hash": "fri"},
                    {"date": "2024-01-07", "nav": "99999", "ann_date": "2024-01-08", "source_hash": "not-session"},
                    {"date": "2024-01-08", "nav": "1.01", "ann_date": "2024-01-09", "source_hash": "mon"},
                    {"date": "2024-01-09", "nav": "999", "ann_date": "2024-01-10", "source_hash": "future"},
                ],
            }
        ]
    }


def test_previous_session_not_natural_day_and_never_reads_u():
    navs = study.nav_index(history())
    original = [{"fund_code": "f", "t": "2024-01-08", "u": "2024-01-09", "y": 1}]
    before = deepcopy(original)
    rows = study.attach(original, navs)
    extra = rows[0]["nav1_input"]
    assert extra["x"] == pytest.approx([0.01])
    assert extra["nav_dates"] == ["2024-01-05", "2024-01-08"]
    assert extra["available_at_assumed"] == "2024-01-09T08:00:00+08:00"
    navs["f"]["2024-01-09"]["nav"] = "0.1"
    assert study.attach(original, navs) == rows
    assert original == before and study.control_rows(rows) == before


def test_missing_previous_session_cannot_use_sunday_or_fill():
    navs = study.nav_index(history())
    del navs["f"]["2024-01-05"]
    with pytest.raises(ValueError, match="PREVIOUS_SESSION_MISSING"):
        study.attach([{"fund_code": "f", "t": "2024-01-08"}], navs)


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "Infinity"])
def test_bad_unit_nav_is_rejected(value):
    navs = study.nav_index(history())
    navs["f"]["2024-01-08"]["nav"] = value
    with pytest.raises(ValueError, match="VALUE_INVALID"):
        study.attach([{"fund_code": "f", "t": "2024-01-08"}], navs)


@pytest.mark.parametrize("case", ["fund", "date"])
def test_duplicate_identity_never_silently_replaces_rows(case):
    source = history()
    if case == "fund":
        source["funds"].append(deepcopy(source["funds"][0]))
    else:
        source["funds"][0]["rows"].append(deepcopy(source["funds"][0]["rows"][0]))
    with pytest.raises(ValueError, match="DUPLICATE_"):
        study.nav_index(source)


@pytest.mark.parametrize("t", ["2024-01-07", "2025-01-02", "2026-09-11"])
def test_non_session_or_protected_year_rejected(t):
    with pytest.raises(ValueError, match="IDENTITY_OR_DATE_INVALID"):
        study.attach([{"fund_code": "f", "t": t}], study.nav_index(history()))


def test_announcement_delay_is_retained_and_same_day_not_before_close():
    navs = study.nav_index(history())
    navs["f"]["2024-01-08"]["ann_date"] = "2024-01-11"
    row = {"fund_code": "f", "t": "2024-01-08"}
    assert study.attach([row], navs)[0]["nav1_input"]["available_at_assumed"] == "2024-01-11T08:00:00+08:00"
    navs["f"]["2024-01-08"]["ann_date"] = "2024-01-08"
    assert study.attach([row], navs)[0]["nav1_input"]["available_at_assumed"] == "2024-01-08T18:00:00+08:00"


def test_exam_late_input_stops_instead_of_dropping_question(tmp_path, monkeypatch):
    (tmp_path / "base/baseline").mkdir(parents=True)
    (tmp_path / "main").mkdir()
    source = history()
    source["funds"][0]["rows"][2]["ann_date"] = "2024-01-10"
    study.write_new(tmp_path / "base/baseline/history.json", source)
    study.write_new(tmp_path / "base/common-exam.json", [{"fund_code": "f", "t": "2024-01-08", "u": "2024-01-09"}])
    study.write_new(
        tmp_path / "main/models.json",
        {q: {"fit_hash": digest([]), "train_as_of": "2024-01-01T08:00:00+08:00"} for q in (*study.QUARTERS, "FINAL")},
    )
    monkeypatch.setattr(study.sector, "selected_fit", lambda *a: [])
    with pytest.raises(ValueError, match="EXAM_NOT_AVAILABLE"):
        study.inputs(tmp_path)


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
            "nav1_input": {
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
    with pytest.raises(ValueError, match="NAV1_MODEL_"):
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
            rows[0]["nav1_input"]["available_at_assumed"] = "2024-02-01T08:00:00+08:00"
        else:
            rows[0]["mature_at"] = "2024-02-01T08:00:00+08:00"
        spec["fit_hashes"]["FINAL"] = digest(rows)
        control["fit_hash"] = digest(study.control_rows(rows))
    with pytest.raises(ValueError, match="NAV1_FIT_"):
        study.fit(rows, control, spec, "FINAL", [])


def test_nonfinite_added_feature_not_imputed(fitted):
    a, _, _, model = fitted
    rows = deepcopy(a[:1])
    rows[0]["nav1_input"]["x"] = [float("nan")]
    with pytest.raises(ValueError, match="NAV1_FEATURE_INVALID"):
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
