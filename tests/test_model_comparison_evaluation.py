"""手算分数、分母、重叠时间块和阶段读取顺序。"""

from datetime import date, timedelta

import pytest
from app.services.model_comparison_evaluation import evaluate_window, metrics, paired_bootstrap
from app.services.model_comparison_protocol import settings


def test_hand_calculated_metrics_and_half_is_down():
    result = metrics([(0, 0.5), (1, 0.75)])
    assert float(result["brier_score"]) == 0.15625
    assert float(result["accuracy"]) == 1
    assert float(result["balanced_accuracy"]) == 1
    assert len(result["reliability"]["bins"]) == 5
    assert metrics([(1, 1.0)])["balanced_accuracy"] is None
    assert metrics([(0, 1.0)])["log_loss"] < 19
    assert metrics([]) is None
    with pytest.raises(ValueError):
        metrics([(1, 2.0)])


def records():
    result = []
    for fund in settings()["funds"]:
        for day in ("2024-01-02", "2024-01-03"):
            models = {
                m: {"probability": 0.6, "raw_direction": 1, "reason": None}
                for m in ("A", "B", *settings()["baselines"])
            }
            result.append({"fund_code": fund, "cutoff_date": day, "key": fund + day, "models": models})
    result[0]["models"]["B"] = {"probability": None, "reason": "TIMEOUT"}
    return result


def test_failures_stay_in_denominator_and_only_common_samples_are_paired():
    predictions = records()
    report = evaluate_window(
        predictions, {p["key"]: 1 for p in predictions}, ["2024-01-02", "2024-01-03"], settings()["bootstrap"]
    )
    b = report["models"]["B"]["coverage"]["001632"]
    assert b["planned"] == 2 and b["scored"] == 1 and b["scored_coverage"] == 0.5
    assert b["prediction_failures"] == {"TIMEOUT": 1}
    assert report["paired_count"] == 5
    assert report["conclusion"] == "DIFFERENCE_UNCLEAR"
    with pytest.raises(ValueError, match="denominator"):
        evaluate_window(predictions[:-1], {}, ["2024-01-02", "2024-01-03"], settings()["bootstrap"])


def test_calendar_blocks_never_compress_missing_dates_and_funds_resample_together():
    days = [str(date(2024, 1, 1) + timedelta(days=i)) for i in range(120)]
    errors = {f: {d: (-0.1 if n < 60 else 0.1) for n, d in enumerate(days)} for f in settings()["funds"]}
    report = paired_bootstrap(days, errors, settings()["bootstrap"])
    assert report["complete_blocks_per_fund"] == {f: 6 for f in errors}
    assert report["macro_ci"] == report["per_fund_ci"]["001632"]
    for offset in range(0, 120, 20):
        errors["001632"].pop(days[offset])
    gapped = paired_bootstrap(days, errors, settings()["bootstrap"])
    assert gapped["complete_blocks_per_fund"]["001632"] == 0
    assert gapped["eligible_for_winner"] is False
    assert gapped == paired_bootstrap(days, errors, settings()["bootstrap"])


def test_exam_answers_cannot_be_opened_without_frozen_prediction_receipt(tmp_path, monkeypatch):
    from app.services import model_comparison_runner as runner

    monkeypatch.setattr(runner, "checked_dataset", lambda folder: ({}, {}, {}))
    opened = []
    monkeypatch.setattr(runner, "load_stage", lambda *args: opened.append(args))
    with pytest.raises(FileNotFoundError):
        runner.score_frozen_predictions(tmp_path)
    assert not opened
