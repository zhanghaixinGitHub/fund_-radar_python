"""三分类训练、原文还原和生产作业路由验证；合成样本不用于准确率宣传。"""

from copy import deepcopy
from datetime import datetime, timedelta

import pytest
from app.services.direction_1d_explanation import explain_original
from app.services.direction_1d_protocol import FEATURE_VERSION, FEATURES, ZONE, digest
from app.services.direction_1d_three_state import CLASSES, POLICY, PROTOCOL, TARGET, TIE_ORDER, predict
from app.services.direction_1d_three_state_training import fit, metrics


def model():
    return {
        "protocol": PROTOCOL,
        "target_definition": TARGET,
        "horizon": 1,
        "feature_version": FEATURE_VERSION,
        "features": list(FEATURES),
        "direction_policy": POLICY,
        "class_order": list(CLASSES),
        "tie_order": list(TIE_ORDER),
        "coef": [[-2, 0, 0, 0, 0, 0, 0], [0] * 7, [2, 0, 0, 0, 0, 0, 0]],
        "intercept": [0, 1, 0],
        "mean": [0] * 7,
        "scale": [1] * 7,
    }


@pytest.mark.parametrize(("value", "expected"), [(-2, "DOWN"), (0, "FLAT"), (2, "UP")])
def test_three_real_decisions_and_explanations(value, expected):
    m = model()
    values = [value, 0, 0, 0, 0, 0, 0]
    result = predict(m, values)
    assert result["direction"] == expected
    assert sum(result["class_scores"].values()) == pytest.approx(1)
    source = {"fund_code": "008888", "features": values}
    body = {
        "protocol": PROTOCOL,
        "fund_code": "008888",
        "base_nav_date": "2026-09-23",
        "target_nav_date": "2026-09-24",
        "input": source,
        "input_hash": digest(source),
        "branches": [
            {
                "branch_id": "FIXED",
                "model_id": "id",
                "model_hash": "hash",
                "status": "AVAILABLE",
                "score": result["score"],
                "class_scores": result["class_scores"],
                "predicted_direction": expected,
            }
        ],
    }
    branch = explain_original(body, lambda *_: m)["branches"][0]
    assert branch["direction"] == expected and branch["referenceDirection"] != expected
    assert branch["intercept"] + sum(f["contribution"] for f in branch["factors"]) > 0
    body["branches"][0]["class_scores"]["FLAT"] += 0.01
    with pytest.raises(ValueError, match="RESTORE"):
        explain_original(body, lambda *_: m)


def test_ties_use_flat_then_up_and_legacy_model_rejected():
    m = model()
    m["intercept"] = [0, 0, 0]
    assert predict(m, [0] * 7)["direction"] == "FLAT"
    m["protocol"] = "DIRECTION_1D_V1"
    with pytest.raises(ValueError, match="PROTOCOL"):
        predict(m, [0] * 7)


def samples():
    rows = []
    for index in range(270):
        day = datetime(2023, 1, 1, tzinfo=ZONE) + timedelta(days=index)
        value = index % 3 - 1
        rows.append(
            {
                "fund_code": "008888",
                "family": "one",
                "group": "CN_EQUITY",
                "t": str(day.date()),
                "u": str((day + timedelta(days=1)).date()),
                "mature_at": (day + timedelta(days=2)).isoformat(),
                "x": [value, 0, 0, 0, 0, 0, 0],
                "actual_direction": CLASSES[index % 3],
            }
        )
    return rows


def test_training_replays_and_does_not_collapse_flat_with_down():
    rows = samples()
    at = datetime(2024, 1, 1, tzinfo=ZONE)
    a = fit(rows, "CN_EQUITY", "synthetic", at)
    assert a == fit(rows, "CN_EQUITY", "synthetic", at)
    assert a["restore_max_score_diff"] < 1e-12
    report = metrics(a, rows)
    assert report["correct"] == len(rows) and report["recall"]["FLAT"] == 1
    assert set(report["predicted_counts"]) == set(CLASSES)
    bad = deepcopy(rows)
    for row in bad:
        if row["actual_direction"] == "FLAT":
            row["actual_direction"] = "DOWN"
    with pytest.raises(ValueError, match="INSUFFICIENT"):
        fit(bad, "CN_EQUITY", "synthetic", at)
    rows[0]["mature_at"] = "2025-01-01T00:00:00+08:00"
    with pytest.raises(ValueError, match="IMMATURE"):
        fit(rows, "CN_EQUITY", "synthetic", at)


def test_production_jobs_get_new_idempotency_namespace(monkeypatch):
    from app.services import direction_1d_jobs as jobs

    monkeypatch.setattr(jobs.repo, "clock", lambda: datetime(2026, 9, 24, 14, tzinfo=ZONE))
    monkeypatch.setattr(jobs, "submit", lambda key, kind, body: (key, kind, body))
    key, kind, body = jobs.submit_forecast("008888")
    assert key.startswith(PROTOCOL + ":") and kind == "FORECAST" and body["protocol"] == PROTOCOL
