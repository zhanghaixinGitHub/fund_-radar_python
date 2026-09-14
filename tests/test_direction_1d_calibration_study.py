"""校准专项回归：时间隔离、数值恢复、同题统计、单次预算和封存完整性。"""

from copy import deepcopy
from datetime import datetime, time

import numpy as np
import pytest
from app.services import direction_1d_calibration_study as study
from app.services.direction_1d_protocol import ZONE, calendar, canonical, digest
from app.services.direction_1d_training import read, write_new


@pytest.fixture
def earlier():
    days = [d for d in calendar()[0] if d.year == 2023][:65]
    rows, scores = [], []
    for i in range(63):
        for f in ("A", "B"):
            t, u, mature = days[i : i + 3]
            at = datetime.combine(t, time(18), ZONE).isoformat()
            rows.append(
                {
                    "fund_code": f,
                    "family": f,
                    "t": str(t),
                    "u": str(u),
                    "y": int((i + (f == "B")) % 3 != 0),
                    "mature_at": datetime.combine(mature, time(8), ZONE).isoformat(),
                    "nav_available_at_assumed": datetime.combine(u, time(8), ZONE).isoformat(),
                    **{k: {"available_at_assumed": at} for k in ("market_input", "specific_input", "activity_input")},
                }
            )
            scores.append(float(0.35 + 0.15 * np.sin(i + (f == "B"))))
    q = "2024Q1"
    meta = study.fit_metadata(
        rows, scores, "2024-01-01T00:00:00+08:00", q, {q + suffix: suffix for suffix in ("", "-V1", "-V2", "-V3")}
    )
    return {"rows": rows, "scores": scores, "metadata": meta}


def test_real_sigmoid_fit_reproduces_and_restores(earlier):
    old = deepcopy(earlier)
    a, b = study.fit(earlier, "synthetic"), study.fit(earlier, "synthetic")
    assert a == b and earlier == old
    assert a["restore_max_score_diff"] <= 1e-12
    assert a["target_days"] == 63 and a["fit_count"] == 126
    assert np.isfinite(study.predict(a, [0, 0.1, 0.5, 0.9, 1])).all()


def test_identity_and_known_logistic_mapping_do_not_require_answers(earlier):
    model = study.fit(earlier, "synthetic")
    model.update(slope=1.0, intercept=0.0)
    np.testing.assert_allclose(study.predict(model, [0.1, 0.5, 0.9]), [0.1, 0.5, 0.9], atol=1e-15)
    model.update(slope=0.0, intercept=float(np.log(3)))
    np.testing.assert_allclose(study.predict(model, [0.0, 0.5, 1.0]), [0.75, 0.75, 0.75])


@pytest.mark.parametrize(
    "scores", [[0.2], [float("nan"), 0.5], [-0.1, 0.5], [0.1, 1.1], [[0.1], [0.9]], [float("inf"), 0.5]]
)
def test_missing_nan_out_of_range_scores_rejected(scores):
    with pytest.raises(ValueError, match="SCORES_INVALID"):
        study.valid_scores(scores, 2)


@pytest.mark.parametrize(
    "change", ["future_answer", "future_input", "duplicate", "missing_day", "one_class", "protected", "hash"]
)
def test_invalid_calibration_data_never_reaches_fit(earlier, monkeypatch, change):
    data = deepcopy(earlier)
    if change == "future_answer":
        data["rows"][-1]["mature_at"] = "2024-01-02T08:00:00+08:00"
    if change == "future_input":
        data["rows"][-1]["activity_input"]["available_at_assumed"] = "2024-01-02T08:00:00+08:00"
    if change == "duplicate":
        data["rows"][1] = deepcopy(data["rows"][0])
    if change == "missing_day":
        data["rows"], data["scores"] = data["rows"][:-2], data["scores"][:-2]
    if change == "one_class":
        for r in data["rows"]:
            r["y"] = 0
    if change == "protected":
        data["rows"][-1]["u"] = "2025-01-02"
    if change == "hash":
        data["metadata"]["score_hash"] = "changed"

    def forbidden(*a, **k):
        pytest.fail("invalid or future data reached fitting")

    monkeypatch.setattr(study.LogisticRegression, "fit", forbidden)
    with pytest.raises(ValueError):
        study.fit(data, "synthetic")


@pytest.mark.parametrize(
    "field,value",
    [
        ("slope", float("nan")),
        ("intercept", float("inf")),
        ("model_released", True),
        ("iterations", 1000),
        ("restore_max_score_diff", 0.01),
        ("recipe", {}),
    ],
)
def test_bad_model_fails_closed(earlier, field, value):
    model = study.fit(earlier, "synthetic")
    model[field] = value
    with pytest.raises(ValueError, match="MODEL_INVALID"):
        study.predict(model, [0.2])


def questions(scores, answers):
    return [
        {
            "fund_code": "A",
            "family": "A",
            "t": f"2024-01-{i + 1:02}",
            "u": f"2024-01-{i + 2:02}",
            "y": y,
            "scores": {study.REFERENCE: p},
        }
        for i, (p, y) in enumerate(zip(scores, answers, strict=True))
    ]


def test_bins_keep_empty_bins_and_count_boundary_once():
    rows = questions([0.0, 0.5, 1.0], [0, 0, 1])
    probability = study.bins(rows, [0, 0.5, 1], confidence=False)
    confidence = study.bins(rows, [0, 0.5, 1], confidence=True)
    assert len(probability) == 10 and len(confidence) == 6
    assert sum(b["count"] for b in probability) == sum(b["count"] for b in confidence) == 3
    assert probability[1]["mean_score"] is None
    assert confidence[0]["observed_fraction"] == 1  # 0.5严格计非上涨，保留旧判断口径。


def test_conservative_scores_are_not_counted_as_more_correct():
    original = study.summary(questions([0.9, 0.1, 0.9, 0.1], [0, 1, 1, 0]), study.REFERENCE)
    conservative = study.summary(questions([0.51, 0.49, 0.51, 0.49], [0, 1, 1, 0]), study.REFERENCE)
    assert original["correct"] == conservative["correct"] == 2
    assert original["high_confidence"]["wrong"] == 2
    assert conservative["high_confidence"]["count"] == 0
    assert conservative["high_confidence"]["error_rate_within"] is None


@pytest.fixture
def packaged(tmp_path, earlier, monkeypatch):
    root = tmp_path / "study"
    root.mkdir()
    (root / "control/main").mkdir(parents=True)
    data, controls, exam = {}, {}, []
    for q in study.QUARTERS:
        d = deepcopy(earlier)
        d["metadata"] = study.fit_metadata(
            d["rows"], d["scores"], "2024-01-01T00:00:00+08:00", q, {q + s: s for s in ("", "-V1", "-V2", "-V3")}
        )
        data[q], controls[q] = d, {study.REFERENCE: {"synthetic": True}}
        for f in ("A", "B"):
            i = len(exam)
            exam.append(
                {
                    "fund_code": f,
                    "family": f,
                    "t": f"2024-01-{i + 1:02}",
                    "u": f"2024-01-{i + 2:02}",
                    "quarter": q,
                    "y": i % 2,
                    "raw_score": 0.4 if i % 2 else 0.6,
                }
            )
    write_new(root / "control/exam.json", exam)
    write_new(
        root / "control/main/predictions.json", [{**r, "scores": {study.REFERENCE: r["raw_score"]}} for r in exam]
    )
    spec = {
        "cohort_id": "synthetic",
        "created_at": "2026-01-01T00:00:00+08:00",
        "exam_count": len(exam),
        "fund_codes": ["A", "B"],
        "calibration_metadata": {q: d["metadata"] for q, d in data.items()},
    }
    monkeypatch.setattr(study, "verify_inputs", lambda r: spec)
    monkeypatch.setattr(study, "earlier_data", lambda r: (data, controls))
    monkeypatch.setattr(study.activity, "predict", lambda m, rows: np.asarray([r["raw_score"] for r in rows]))
    return root, data


def test_main_replay_and_zero_fit_verification_have_exact_budget(packaged, monkeypatch):
    root, _ = packaged
    main = study.run(root)
    replay = study.run(root, replay=True)
    assert main["successful_fits"] == replay["successful_fits"] == 4
    assert main["models_hash"] == replay["models_hash"]

    def forbidden(*a, **k):
        pytest.fail("verification or duplicate execution fitted again")

    monkeypatch.setattr(study, "fit", forbidden)
    assert study.verify(root)["new_fits"] == 0
    with pytest.raises(FileExistsError):
        study.run(root)
    with pytest.raises(FileExistsError):
        study.run(root, replay=True)


@pytest.mark.parametrize("change", ["extra_fit", "incomplete_replay", "prediction", "reservation", "model", "proof"])
def test_run_corruption_is_detected(packaged, change):
    root, _ = packaged
    study.run(root)
    study.run(root, replay=True)
    if change == "extra_fit":
        write_new(root / "main/extra-reserved.json", {})
    if change == "incomplete_replay":
        (root / "replay/completion.json").unlink()
    if change == "prediction":
        p = root / "main/predictions.json"
        obj = read(p)
        obj[0]["scores"][study.CANDIDATE] = 0.999
        p.write_text(canonical(obj), encoding="utf-8")
    if change == "reservation":
        p = root / "main/2024Q1-reserved.json"
        obj = read(p)
        obj["at"] = "2020-01-01T00:00:00+08:00"
        p.write_text(canonical(obj), encoding="utf-8")
    if change == "model":
        p = root / "main/2024Q1.json"
        obj = read(p)
        obj["slope"] += 1
        p.write_text(canonical(obj), encoding="utf-8")
    if change == "proof":
        p = root / "replay-proof.json"
        obj = read(p)
        obj["successful_fits"] = 5
        p.write_text(canonical(obj), encoding="utf-8")
    with pytest.raises(ValueError):
        study.verify(root)


@pytest.mark.parametrize("change", ["budget", "code", "input", "expired", "source_escape"])
def test_frozen_spec_and_source_integrity(tmp_path, monkeypatch, change):
    monkeypatch.setattr(study, "fingerprint", lambda: {"code": "fixed"})
    write_new(tmp_path / "source.json", {"input": 1})
    spec = {
        **deepcopy(study.LIMITS),
        "fingerprint": {"code": "fixed"},
        "features": study.activity.FEATURES,
        "feature_recipe": study.activity.FEATURE_RECIPE,
        "source_expires_at": "2099-01-01T00:00:00+08:00",
        "input_files": {"source.json": study.file_hash(tmp_path / "source.json")},
    }
    if change == "budget":
        spec["max_main_fits"] = 5
    if change == "code":
        spec["fingerprint"]["code"] = "changed"
    if change == "input":
        spec["input_files"]["source.json"] = "changed"
    if change == "expired":
        spec["source_expires_at"] = "2020-01-01T00:00:00+08:00"
    if change == "source_escape":
        spec["input_files"] = {"../source.json": "changed"}
    write_new(tmp_path / "study.json", spec)
    write_new(tmp_path / "study-receipt.json", {"hash": digest(spec)})
    with pytest.raises(ValueError):
        study.verify_inputs(tmp_path)
