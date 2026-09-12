"""检查树JSON恢复、拓扑、固定预算和只读诊断边界；仅用合成数据做一次工程拟合。"""

from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pytest
from app.services import direction_1d_tree_audit as audit
from app.services import direction_1d_tree_study as study
from app.services.direction_1d_protocol import ZONE, calendar, digest
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="module")
def fitted():
    # 不读取真实基金或考题；只检验不同数值路径能否准确保存、恢复。
    rng = np.random.default_rng(61)
    values = rng.normal(size=(500, 11))
    days = calendar()[0][61:561]
    rows = [
        {
            "fund_code": "synthetic",
            "family": "synthetic",
            "group": "CN_EQUITY",
            "t": str(d),
            "x": x[:7].tolist(),
            "market_input": {"x": x[7:9].tolist()},
            "specific_input": {"x": x[9:].tolist()},
            "y": int((x[0] > 0) != (x[9] > 0)),
        }
        for d, x in zip(days, values, strict=True)
    ]
    w = study.weights(rows)
    scaler = StandardScaler().fit(values, sample_weight=w)
    linear = {
        "fit_hash": digest(rows),
        "weight_hash": digest(w.tolist()),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "train_as_of": "2024-01-01T00:00:00+08:00",
    }
    model = study.fit(rows, linear, {"cohort_id": "SYNTHETIC_TEST_ONLY"}, rows[:40])
    return rows, model, linear


def test_numeric_tree_json_matches_estimator_and_round_trip(fitted, tmp_path):
    rows, model, _ = fitted
    assert max(model["restore_max_score_diff"].values()) <= 1e-12
    assert len(model["trees"]) == 100 and max(n["depth"] for t in model["trees"] for n in t) <= 2
    study.write_new(tmp_path / "model.json", model)
    restored = study.read(tmp_path / "model.json")
    assert np.array_equal(study.predict(model, rows), study.predict(restored, rows))
    assert not len(study.predict(restored, []))


@pytest.mark.parametrize("case", ["rounds", "scale", "nan", "feature", "cycle", "recipe", "target"])
def test_malformed_model_never_scores(fitted, case):
    rows, source, _ = fitted
    model = deepcopy(source)
    if case == "rounds":
        model["trees"].pop()
    elif case == "scale":
        model["scale"][0] = 0
    elif case == "nan":
        model["trees"][0][0]["value"] = float("nan")
    elif case == "recipe":
        model["recipe"]["max_depth"] = 3
    elif case == "target":
        model["target_definition"] = "TOTAL_RETURN"
    else:
        node = next(n for tree in model["trees"] for n in tree if not n["leaf"])
        node["feature" if case == "feature" else "left"] = 11 if case == "feature" else 0
    with pytest.raises(ValueError, match="TREE_MODEL_"):
        study.predict(model, rows[:2])


def test_changed_fit_or_weights_prevents_classifier_fit(fitted, monkeypatch):
    rows, _, linear = fitted
    monkeypatch.setattr(study.HistGradientBoostingClassifier, "fit", lambda *a, **k: pytest.fail("不能多拟合"))
    bad = {**linear, "weight_hash": "changed"}
    with pytest.raises(ValueError, match="FIT_OR_WEIGHTS_CHANGED"):
        study.fit(rows, bad, {"cohort_id": "fake"}, [])
    with pytest.raises(ValueError, match="FIT_OR_WEIGHTS_CHANGED"):
        study.fit(rows[:-1], linear, {"cohort_id": "fake"}, [])


def test_invalid_new_input_is_not_imputed(fitted):
    rows, model, _ = fitted
    row = deepcopy(rows[0])
    row["specific_input"]["x"][0] = float("nan")
    with pytest.raises(ValueError, match="SECTOR_FEATURE_INVALID"):
        study.predict(model, [row])


@pytest.mark.parametrize("replay", [False, True])
def test_completed_budget_is_not_refitted(tmp_path, monkeypatch, replay):
    (tmp_path / ("replay" if replay else "main")).mkdir()
    monkeypatch.setattr(study, "verify_inputs", lambda r: {})
    monkeypatch.setattr(study, "verify", lambda r: {})
    monkeypatch.setattr(study, "fit", lambda *a: pytest.fail("已用预算不能再次拟合"))
    with pytest.raises(FileExistsError):
        study.run(tmp_path, replay=replay)


def test_frozen_input_tampering_fails(tmp_path):
    study.write_new(tmp_path / "evidence.json", {"count": 25})
    spec = {
        "fingerprint": study.fingerprint(),
        "tree_recipe": study.TREE_RECIPE,
        "input_files": {"evidence.json": study.file_hash(tmp_path / "evidence.json")},
        "source": {"source_expires_at": (datetime.now(ZONE) + timedelta(days=1)).isoformat()},
    }
    study.write_new(tmp_path / "study.json", spec)
    study.write_new(tmp_path / "study-receipt.json", {"hash": digest(spec)})
    study.verify_inputs(tmp_path)
    (tmp_path / "evidence.json").write_text('{"count":24}', encoding="utf-8")
    with pytest.raises(ValueError, match="FROZEN_INPUT_CHANGED"):
        study.verify_inputs(tmp_path)


def test_cash_events_are_annotations_not_rewritten_labels():
    row = {"fund_code": "f", "t": "2024-01-10", "u": "2024-01-11", "y": 1}
    events = [
        {"fund_code": "f", "ex_date": "2024-01-05", "nav_ex_date": None, "ann_date": "2024-01-03"},
        {"fund_code": "f", "ex_date": None, "nav_ex_date": "2024-01-11", "ann_date": "2024-01-12"},
        {"fund_code": "other", "ex_date": "2024-01-11", "nav_ex_date": None, "ann_date": "2024-01-03"},
    ]
    flags = audit.event_flags(row, events)
    assert flags == {"target_cash_event": True, "input_cash_event": True, "target_event_announced_by_t": False}
    assert row["y"] == 1
    events[1]["ann_date"] = "2024-01-10"
    assert audit.event_flags(row, events)["target_event_announced_by_t"]


def test_nav_revision_is_rejected_before_scoring(tmp_path, monkeypatch):
    (tmp_path / "baseline").mkdir()
    study.write_new(tmp_path / "study.json", {"fund_codes": ["f"]})
    study.write_new(tmp_path / "common-exam.json", [])
    study.write_new(
        tmp_path / "baseline/history.json",
        {
            "funds": [
                {
                    "fund_code": "f",
                    "rows": [{"date": "2024-01-02", "nav": "1.2", "ann_date": "2024-01-03", "source_hash": "old"}],
                }
            ]
        },
    )
    snapshot = {
        "fund_codes": ["f"],
        "base_study_hash": study.file_hash(tmp_path / "study.json"),
        "navs": [
            {
                "fund_code": "f",
                "nav_date": "2024-01-02",
                "unit_nav": "1.2",
                "ann_date": "2024-01-03",
                "content_hash": "new",
            }
        ],
    }
    monkeypatch.setattr(audit.sector, "selected_fit", lambda *a: pytest.fail("应在读FIT前拒绝历史修订"))
    with pytest.raises(ValueError, match="FROZEN_NAV_CHANGED"):
        audit.analyze(tmp_path, snapshot)


def test_adjustment_factor_is_only_an_event_cross_check():
    snapshot = {
        "navs": [
            {"fund_code": "f", "nav_date": "2024-01-02", "unit_nav": "1", "adjusted_nav": "1"},
            {"fund_code": "f", "nav_date": "2024-01-03", "unit_nav": "0.9", "adjusted_nav": "1"},
        ],
        "dividends": [{"fund_code": "f", "ex_date": "2024-01-03", "nav_ex_date": None}],
    }
    before = deepcopy(snapshot)
    changes = audit.adjustment_changes(snapshot)
    assert len(changes) == 1 and changes[0]["cash_event_matches"]
    assert changes[0]["ratio_change"] == pytest.approx(1 / 0.9 - 1)
    assert snapshot == before
