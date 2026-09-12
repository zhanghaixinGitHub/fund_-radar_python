"""验证公平对照与信息时间，合成题目不计研究拟合或真实成绩。"""

from datetime import datetime

import pytest
from app.services import direction_1d_comparison as study
from app.services.direction_1d_protocol import ZONE, digest


def row(code="000001", group="CN_EQUITY", day="2024-01-02", y=1):
    return {
        "fund_code": code,
        "group": group,
        "family": code,
        "t": day,
        "u": "2024-01-03",
        "x": [0.1] * 7,
        "y": y,
        "actual_direction": "UP" if y else "DOWN",
        "momentum": 1,
        "kind": "HISTORICAL_RECONSTRUCTION",
    }


def test_one_shared_vector_with_type_indicators_not_separate_models():
    a, b, c = row(), row(group="CN_MIXED"), row(group="CN_BOND")
    assert study.vector(a, "POOLED_7") == study.vector(b, "POOLED_7") == [0.1] * 7
    assert study.vector(a, "POOLED_7_TYPE")[-2:] == [0.0, 0.0]
    assert study.vector(b, "POOLED_7_TYPE")[-2:] == [0.0, 1.0]
    assert study.vector(c, "POOLED_7_TYPE")[-2:] == [1.0, 0.0]
    with pytest.raises(ValueError):
        study.vector(row(group="QDII"), "POOLED_7")


def test_fit_is_exact_union_and_never_includes_exam_or_future_input(monkeypatch):
    fit_rows = [row(), row("000002", "CN_BOND")]
    exam = row("000003", day="2024-04-02")
    monkeypatch.setattr(study.original, "select_fit", lambda rows, cutoff: [r for r in rows if r["t"] < "2024-04-01"])
    cutoff = datetime(2024, 4, 1, tzinfo=ZONE)
    models = {r["group"] + "-2024Q2": {"train_as_of": cutoff.isoformat(), "fit_hash": digest([r])} for r in fit_rows}
    available = {(r["fund_code"], r["t"]): datetime(2024, 1, 3, tzinfo=ZONE) for r in fit_rows}
    assert study.exact_fit([*fit_rows, exam], models, "2024Q2", available) == fit_rows
    available["000001", "2024-01-02"] = datetime(2024, 4, 2, tzinfo=ZONE)
    with pytest.raises(ValueError, match="FIT_INPUT_NOT_AVAILABLE"):
        study.exact_fit([*fit_rows, exam], models, "2024Q2", available)


def test_common_exam_excludes_unavailable_inputs_and_duplicate_questions(monkeypatch):
    rows = [row(), row("000002")]
    predictions = [
        {"fund_code": r["fund_code"], "t": r["t"], "u": r["u"], "y": r["y"], "score": 0.6, "job": "CN_EQUITY-2024Q1"}
        for r in rows
    ]
    monkeypatch.setattr(study.original, "score", lambda model, x: 0.6)
    models = {"CN_EQUITY-2024Q1": {}}
    available = {
        ("000001", "2024-01-02"): datetime(2024, 1, 3, 8, tzinfo=ZONE),
        ("000002", "2024-01-02"): datetime(2024, 1, 3, 8, 1, tzinfo=ZONE),
    }
    valid, excluded = study.common_exam(rows, predictions, models, available)
    assert excluded == 1 and [r["fund_code"] for r in valid] == ["000001"]
    with pytest.raises(ValueError, match="DUPLICATE"):
        study.common_exam(rows, [predictions[0], predictions[0]], models, available)


def test_paired_identical_predictions_have_zero_gain_and_interval():
    rows = []
    for i in range(1, 11):
        r = row(day=f"2024-01-{i:02d}", y=i % 2)
        r.update(u=f"2024-01-{i + 1:02d}", directions={study.BASELINE: 1, "POOLED_7": 1})
        rows.append(r)
    report = study.paired(
        rows, "POOLED_7", {"bootstrap_seed": 1, "bootstrap_block_days": 5, "bootstrap_repetitions": 20}
    )
    assert report["weighted_accuracy_difference"] == report["extra_correct"] == 0
    assert report["block_bootstrap_interval_95"] == [0.0, 0.0]


def test_changed_frozen_comparison_inputs_are_rejected(tmp_path):
    (tmp_path / "baseline").mkdir()
    for name in study.BASE_FILES:
        study.write_new(tmp_path / "baseline" / name, {})
    study.write_new(tmp_path / "common-exam.json", [])
    study.write_new(
        tmp_path / "study.json",
        {
            "fingerprint": study.fingerprint(),
            "candidates": study.CANDIDATES,
            "recipe": study.RECIPE,
            "base_hashes": {name: digest({}) for name in study.BASE_FILES},
            "common_exam_hash": digest([]),
        },
    )
    study.verify_inputs(tmp_path)
    (tmp_path / "baseline/models.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="COMPARISON_BASE_CHANGED"):
        study.verify_inputs(tmp_path)
