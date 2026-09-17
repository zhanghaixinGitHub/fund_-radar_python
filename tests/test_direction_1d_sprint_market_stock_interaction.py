"""固定交互模型的原训练身份、时点、缺失回退和九维边界；仅人工样本拟合。"""

from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pytest
from app.services import direction_1d_sprint_market_stock_interaction as s
from sklearn.linear_model import LogisticRegression


@pytest.mark.parametrize("scale", [[1, 1, 1], [1, 0, 1, 1], [1, 1, 1, float("nan")]])
def test_bad_scale_rejected(scale):
    with pytest.raises(ValueError, match="TRANSFORM_INVALID"):
        s.transform([[1, 2, 3, 4, 5, 6]], scale, False)


def test_missing_extra_keeps_corresponding_parent_answer():
    values = [
        {"available": True, "stock_breadth": {"available": False}},
        {"available": False, "stock_breadth": {"available": True}},
    ]
    fallback = [{"prediction": 1, "route": "RAW_PARENT"}, {"prediction": 0, "route": "SPX_SIGN_FALLBACK"}]
    before = deepcopy(fallback)
    assert s.candidate_answers(values, s.CANDIDATES[0], None, fallback) == before
    assert fallback == before


@pytest.mark.parametrize("basis", [float("nan"), 21.0, True])
def test_bad_stock_value_not_silently_normalized(monkeypatch, basis):
    monkeypatch.setattr(s.baseline, "market_vector", lambda _: [1, 2, 3])
    with pytest.raises(ValueError, match="STOCK_BREADTH_INVALID"):
        s.vector(
            {
                "stock_breadth": {
                    "available": True,
                    "breadth": basis,
                    "median_pct": 0.2,
                    "iqr_pct": 2.0,
                }
            }
        )


def test_head_timing_and_schema(head):
    assert s.validate_head(head, s.CANDIDATES[0], "2026-09-16") is head["model"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("cutoff", "2026-09-15"),
        ("max_feature_date", "2026-09-14"),
        ("fit_dates", 503),
        ("signed", False),
        ("mean", [1.0] * 9),
    ],
)
def test_head_changed_recipe_or_maturity_rejected(head, field, value):
    head[field] = value
    with pytest.raises(ValueError):
        s.validate_head(head, s.CANDIDATES[0], "2026-09-16")


@pytest.fixture
def identity_input():
    rows = [
        {
            "code": "001021",
            "y": 1,
            "market": {"original": 2, "stock_breadth": {"breadth": 0.1}},
        }
    ]
    proof = {"at": "2026-09-16T09:00:00+08:00", "fresh_rows_hash": "source", "merged_rows": 1}
    return rows, proof, s.question_identity(rows, proof)


def test_only_verification_time_can_change(identity_input):
    rows, proof, expected = identity_input
    proof["at"] = "2026-09-16T09:30:00+08:00"
    assert s.verify_identity(rows, proof, expected) == expected


@pytest.mark.parametrize(
    "kind", ["source_hash", "source_count", "label", "original_feature", "basis", "extra_proof_key"]
)
def test_actual_question_or_source_changes_rejected(identity_input, kind):
    rows, proof, expected = identity_input
    if kind == "source_hash":
        proof["fresh_rows_hash"] = "changed"
    elif kind == "source_count":
        proof["merged_rows"] = 2
    elif kind == "label":
        rows[0]["y"] = 0
    elif kind == "original_feature":
        rows[0]["market"]["original"] = 3
    elif kind == "basis":
        rows[0]["market"]["stock_breadth"]["breadth"] = 0.2
    else:
        proof["new_field"] = True
    with pytest.raises(ValueError, match="QUESTION_IDENTITY_CHANGED"):
        s.verify_identity(rows, proof, expected)


def test_missing_new_feature_falls_back_to_sign_control_not_raw(monkeypatch):
    controls = {n: [{"prediction": int(n == s.CONTROLS[1])}] for n in s.CONTROLS}
    monkeypatch.setattr(s.reference, "batch", lambda *_: controls)
    result = s.batch([{"available": True, "stock_breadth": {"available": False}}], {}, [])
    assert result[s.CANDIDATES[0]] == controls[s.CONTROLS[1]]
    assert result[s.CANDIDATES[0]] != controls[s.CONTROLS[0]]


@pytest.fixture
def synthetic_training(monkeypatch):
    """只用人工504行走实际拟合函数，验证六维矩阵和样本权重；不读取基金成绩。"""
    rows = []
    for i in range(504):
        u = date(2023, 1, 2) + timedelta(days=i)
        sign = 1 if i % 3 else -1
        rows.append(
            {
                "code": "SYNTHETIC",
                "family": "SYNTHETIC",
                "group": "CN_EQUITY",
                "t": str(u - timedelta(days=1)),
                "u": str(u),
                "mature": str(u),
                "y": int(sign > 0),
                "market": {
                    "available": True,
                    "raw": [sign, -sign, sign],
                    "stock_breadth": {
                        "available": True,
                        "date": str(u - timedelta(days=1)),
                        "breadth": sign / 3,
                        "median_pct": sign / 2,
                        "iqr_pct": 1 + i % 5,
                    },
                },
            }
        )
    monkeypatch.setattr(s.original, "training_rows", lambda rs, _: rs)
    monkeypatch.setattr(s.baseline, "market_vector", lambda v: v["raw"])
    monkeypatch.setattr(s, "active", lambda: None)
    return rows


def test_real_fit_function_on_synthetic_rows_preserves_sample_identity(synthetic_training):
    rows = synthetic_training
    head = s.fit(rows, s.CANDIDATES[0], "2025-01-01")
    assert s.validate_head(head, s.CANDIDATES[0]).coef_.shape == (1, 9)
    assert head["fit_hash"] == s.base.digest([s.data.original_row(r) for r in rows])
    assert head["weight_hash"] == s.base.digest(s.regression.weights(rows).tolist())
    assert head["fit_dates"] == head["fit_rows"] == 504


@pytest.mark.parametrize("bad", ["maturity", "source_missing", "date_count"])
def test_invalid_training_window_rejected_before_fit(synthetic_training, bad):
    rows = synthetic_training
    if bad == "maturity":
        rows[-1]["mature"] = "2025-01-01"
    elif bad == "source_missing":
        rows[-1]["market"]["stock_breadth"]["available"] = False
    else:
        rows.pop()
    with pytest.raises(ValueError, match="TRAINING_MATURITY_OR_COVERAGE_INVALID"):
        s.fit(rows, s.CANDIDATES[0], "2025-01-01")


@pytest.fixture
def head():
    raw = np.array(
        [[1, 1, 1, 0.1, 0.2, 2], [-1, -1, -1, -0.1, -0.2, 1], [1, -1, 1, 0.2, 0.1, 1], [-1, 1, -1, -0.2, -0.1, 2]],
        dtype=float,
    )
    x = np.column_stack([raw, np.sign(raw[:, :3]) * raw[:, 3, None]])
    model = LogisticRegression(C=0.1, fit_intercept=False, max_iter=1500, random_state=17).fit(x, [1, 0, 1, 0])
    return {
        "model": model,
        "scale": [1.0] * 9,
        "mean": [0.0] * 9,
        "signed": True,
        "fit_dates": 504,
        "fit_end": "2026-09-14",
        "max_mature_date": "2026-09-15",
        "cutoff": "2026-09-16",
        "max_feature_date": "2026-09-10",
    }


@pytest.mark.parametrize("breadth", [-1.0, -0.2, 0.0, 0.4, 1.0])
def test_interactions_depend_on_t_breadth_and_overnight_sign_not_magnitude(monkeypatch, breadth):
    market = {
        "raw": [2.0, -0.1, 0.0],
        "stock_breadth": {"available": True, "breadth": breadth, "median_pct": 0.2, "iqr_pct": 2.0},
    }
    monkeypatch.setattr(s.baseline, "market_vector", lambda m: m["raw"])
    a = s.vector(market)
    assert a[6:] == [breadth, -breadth, 0.0]
    market["raw"] = [0.01, -100.0, 0.0]
    assert s.vector(market)[6:] == a[6:]
    x = s.transform([a], [2.0, 2.0, 2.0, 0.5, 0.5, 2.0, 0.5, 0.5, 0.5], True)[0]
    assert np.allclose(x[:3], [1.0, -1.0, 0.0])
    assert np.allclose(x[3:], [breadth / 0.5, 0.4, 1.0, breadth / 0.5, -breadth / 0.5, 0.0])


def test_original_six_inputs_identical_to_unextended_recipe(monkeypatch):
    from app.services import direction_1d_sprint_market_stock_breadth as old

    market = {
        "raw": [2.0, -0.1, 0.0],
        "stock_breadth": {"available": True, "breadth": 0.4, "median_pct": 0.2, "iqr_pct": 2.0},
    }
    monkeypatch.setattr(s.baseline, "market_vector", lambda m: m["raw"])
    assert s.vector(market)[:6] == old.vector(market)
