"""时间权重与不可覆盖研究的边界；仅合成数据一次拟合，不读取真实基金调参。"""

from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pytest
from app.services import direction_1d_recency_study as study
from app.services.direction_1d_protocol import ZONE, calendar, digest


def test_half_life_uses_sessions_and_preserves_total():
    days = calendar()[0]
    rows = [{"t": str(days[i]), "family": "one"} for i in (10, 136, 262)]
    w, info = study.recency_weights(rows)
    assert w[0] / w[1] == pytest.approx(0.5)
    assert w[1] / w[2] == pytest.approx(0.5)
    assert w.sum() == pytest.approx(study.weights(rows).sum(), abs=1e-12)
    assert info["anchor_t"] == rows[-1]["t"] and info["max_age_trading_days"] == 252
    # 原始权重相同的三行，经衰减后的有效行数变少；不能把它当独立市场日。
    assert info["effective_row_count"] < info["original_effective_row_count"]


def test_weekend_counts_one_trading_session():
    rows = [{"t": t, "family": "one"} for t in ("2024-01-05", "2024-01-08")]
    w, info = study.recency_weights(rows)
    assert info["max_age_trading_days"] == 1
    assert w[0] / w[1] == pytest.approx(2 ** (-1 / 126))


def test_family_same_date_ratios_and_row_order_are_preserved():
    rows = [
        {"t": "2024-01-02", "family": "a", "fund_code": "a1"},
        {"t": "2024-01-02", "family": "a", "fund_code": "a2"},
        {"t": "2024-01-02", "family": "b", "fund_code": "b1"},
        {"t": "2024-01-03", "family": "a", "fund_code": "a1"},
    ]
    w, _ = study.recency_weights(rows)
    old = study.weights(rows)
    assert w[0] / w[2] == pytest.approx(old[0] / old[2])
    reversed_w, _ = study.recency_weights(list(reversed(rows)))
    np.testing.assert_allclose(w, reversed_w[::-1], rtol=1e-14)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"t": "2024-01-06", "family": "a"}],
        [{"t": "2025-01-02", "family": "a"}],
        [{"t": "2026-09-11", "family": "a"}],
    ],
)
def test_invalid_or_protected_fit_date_rejected(rows):
    with pytest.raises(ValueError, match="FIT_DATE_INVALID_OR_PROTECTED"):
        study.recency_weights(rows)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_base_weight_rejected(monkeypatch, value):
    monkeypatch.setattr(study, "weights", lambda _: np.asarray([value]))
    with pytest.raises(ValueError, match="BASE_WEIGHT_INVALID"):
        study.recency_weights([{"t": "2024-01-02", "family": "one"}])


@pytest.fixture(scope="module")
def fitted():
    # 合成两个输入的条件交互；控制对象为合法常数树，不另拟合或读真实基金。
    rng = np.random.default_rng(126)
    values = rng.normal(size=(500, 11))
    days = calendar()[0][61:561]
    rows = [
        {
            "fund_code": "synthetic",
            "family": "synthetic",
            "group": "CN_EQUITY",
            "t": str(d),
            "mature_at": datetime.combine(d + timedelta(days=3), datetime.min.time(), ZONE).isoformat(),
            "x": x[:7].tolist(),
            "market_input": {"x": x[7:9].tolist()},
            "specific_input": {"x": x[9:].tolist()},
            "y": int((x[0] > 0) != (x[9] > 0)),
        }
        for d, x in zip(days, values, strict=True)
    ]
    old = study.weights(rows)
    control = {
        "candidate": study.tree.CANDIDATE,
        "kind": "RESEARCH_ONLY",
        "target_definition": "UNIT_NAV_DIRECTION_V1",
        "train_as_of": "2024-01-01T00:00:00+08:00",
        "model_released": False,
        "features": study.sector.feature_names("SPECIFIC11"),
        "recipe": study.tree.TREE_RECIPE,
        "mean": values.mean(axis=0).tolist(),
        "scale": values.std(axis=0).tolist(),
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
        "fit_hash": digest(rows),
        "fit_count": len(rows),
        "weight_hash": digest(old.tolist()),
        "distinct_dates": 500,
        "classes": {str(y): sum(r["y"] == y for r in rows) for y in (0, 1)},
    }
    info = study.recency_weights(rows)[1]
    spec = {"cohort_id": "SYNTHETIC_ONLY", "weight_audits": {"FINAL": info}}
    model = study.fit(rows, control, spec, "FINAL", rows[:40])
    return rows, control, spec, model


def test_recency_json_matches_estimator_and_preserves_control(fitted, tmp_path):
    rows, control, spec, model = fitted
    assert max(model["restore_max_score_diff"].values()) <= 1e-12
    assert model["weight_hash"] != control["weight_hash"]
    assert model["weight_audit"] == spec["weight_audits"]["FINAL"]
    assert model["mean"] == control["mean"] and model["scale"] == control["scale"]
    assert model["weight_audit"]["anchor_t"] == max(r["t"] for r in rows)
    assert model["train_as_of"][:10] > model["weight_audit"]["anchor_t"]
    study.write_new(tmp_path / "model.json", model)
    assert np.array_equal(study.predict(model, rows), study.predict(study.read(tmp_path / "model.json"), rows))
    assert control["candidate"] == study.tree.CANDIDATE


@pytest.mark.parametrize("case", ["half_life", "candidate", "release", "tree", "input"])
def test_invalid_recency_model_cannot_score(fitted, case):
    rows, _, _, source = fitted
    model, selected = deepcopy(source), deepcopy(rows[:1])
    if case == "half_life":
        model["weight_recipe"]["half_life_trading_days"] = 63
    elif case == "candidate":
        model["candidate"] = study.tree.CANDIDATE
    elif case == "release":
        model["model_released"] = True
    elif case == "tree":
        model["trees"].pop()
    else:
        selected[0]["x"][0] = float("nan")
    with pytest.raises(ValueError):
        study.predict(model, selected)


@pytest.mark.parametrize("case", ["fit", "weight", "maturity"])
def test_changed_or_unmature_fit_never_calls_estimator(fitted, monkeypatch, case):
    source, source_control, source_spec, _ = fitted
    rows, control, spec = deepcopy(source), deepcopy(source_control), deepcopy(source_spec)
    monkeypatch.setattr(study.HistGradientBoostingClassifier, "fit", lambda *a, **k: pytest.fail("必须在拟合前拒绝"))
    if case == "fit":
        rows.pop()
    elif case == "weight":
        control["weight_hash"] = "bad"
    else:
        control["train_as_of"] = rows[0]["t"] + "T00:00:00+08:00"
    with pytest.raises(ValueError, match="RECENCY_FIT_"):
        study.fit(rows, control, spec, "FINAL", [])


@pytest.mark.parametrize("replay", [False, True])
def test_budget_cannot_be_reused(tmp_path, monkeypatch, replay):
    (tmp_path / ("replay" if replay else "main")).mkdir()
    monkeypatch.setattr(study, "verify_inputs", lambda _: {})
    monkeypatch.setattr(study, "verify", lambda _: {})
    monkeypatch.setattr(study, "fit", lambda *a: pytest.fail("预算已经使用"))
    with pytest.raises(FileExistsError):
        study.run(tmp_path, replay=replay)


def test_frozen_input_cannot_be_replaced(tmp_path):
    study.write_new(tmp_path / "input.json", {"count": 25})
    spec = {
        "fingerprint": study.fingerprint(),
        "tree_recipe": study.tree.TREE_RECIPE,
        "weight_recipe": study.WEIGHT_RECIPE,
        "candidate": study.CANDIDATE,
        "reference": study.REFERENCE,
        "model_released": False,
        "source": {"source_expires_at": (datetime.now(ZONE) + timedelta(days=1)).isoformat()},
        "input_files": {"input.json": study.file_hash(tmp_path / "input.json")},
    }
    study.write_new(tmp_path / "study.json", spec)
    study.write_new(tmp_path / "study-receipt.json", {"hash": digest(spec)})
    study.verify_inputs(tmp_path)
    (tmp_path / "input.json").write_text('{"count":24}', encoding="utf-8")
    with pytest.raises(ValueError, match="FROZEN_INPUT_CHANGED"):
        study.verify_inputs(tmp_path)


def test_append_score_does_not_rewrite_old_labels_or_conclusions():
    old = [
        {
            "fund_code": "f",
            "t": "2024-01-02",
            "y": 1,
            "kind": "HISTORICAL",
            "scores": {"TREE11": 0.4, "LINEAR11": 0.6},
            "directions": {"TREE11": 0, "LINEAR11": 1},
        }
    ]
    before = deepcopy(old)
    result = study.expected_predictions(old, {("f", "2024-01-02"): 0.5})
    assert old == before and result[0]["y"] == 1
    assert result[0]["scores"]["TREE11"] == 0.4 and result[0]["directions"][study.CANDIDATE] == 0


def test_replay_requires_identical_models_and_scores(tmp_path):
    for mode in ("main", "replay"):
        (tmp_path / mode).mkdir()
        study.write_new(tmp_path / mode / "models.json", {"tree": 1})
        study.write_new(tmp_path / mode / "predictions.json", {"score": 0.5 if mode == "main" else 0.5001})
    with pytest.raises(ValueError, match="REPLAY_MISMATCH"):
        study.require_replay_outputs(tmp_path)
