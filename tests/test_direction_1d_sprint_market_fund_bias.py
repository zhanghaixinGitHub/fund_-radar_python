"""基金偏置的收缩、训练成熟边界及代码感知的实际预测回读。"""

from copy import deepcopy
from datetime import date, timedelta

import joblib
import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_fund_bias as s
from app.services import direction_1d_sprint_market_fund_forward as runtime
from sklearn.linear_model import LogisticRegression

from test_direction_1d_sprint_market_fxi_interval import head as lr_head
from test_direction_1d_sprint_market_fxi_interval import paired
from test_direction_1d_sprint_market_only_forward import forecast
from test_direction_1d_sprint_market_only_forward import ready as parent_ready  # noqa: F401


@pytest.fixture(autouse=True)
def small_scope(monkeypatch):
    monkeypatch.setattr(s, "CODE_ORDER", ("001000", "002000"))


def head(signed=False):
    model = LogisticRegression(C=0.1, fit_intercept=False, max_iter=1500, random_state=17)
    model.coef_ = np.array([[0.0, 0.0, 0.0, 4.0, -4.0]])
    model.intercept_ = np.array([0.0])
    model.classes_ = np.array([0, 1])
    model.n_features_in_ = 5
    return {
        "model": model,
        "scale": [1.0] * 3,
        "mean": [0.0] * 3,
        "signed": signed,
        "code_order": list(s.CODE_ORDER),
        "bias_scale": 0.25,
        "group": "CN_EQUITY",
        "allowed_codes": list(s.CODE_ORDER),
        "training_codes": list(s.CODE_ORDER),
        "fit_dates": 504,
        "fit_end": "2026-09-14",
        "max_mature_date": "2026-09-15",
        "cutoff": "2026-09-16",
    }


def bundle():
    out = {n: {"CN_EQUITY": head(bool(i))} for i, n in enumerate(s.CANDIDATES)}
    out.update({n: {"CN_EQUITY": lr_head(bool(i))} for i, n in enumerate(s.CONTROLS[:2])})
    return out


@pytest.mark.parametrize("signed", [False, True])
def test_bias_remains_quarter_scale_and_only_own_code_changes(signed):
    values = [paired(), paired()]
    x = s.design(values, list(s.CODE_ORDER), [2, 3, 4], signed)
    np.testing.assert_allclose(x[0, :3], x[1, :3])
    np.testing.assert_allclose(x[:, 3:], [[0.25, 0], [0, 0.25]])
    h = head(signed)
    answers = s.candidate_answers(values, list(s.CODE_ORDER), s.CANDIDATES[int(signed)], h)
    assert [v["prediction"] for v in answers] == [1, 0]
    assert answers[0]["research_score"] == pytest.approx(1 / (1 + np.exp(-1)))


@pytest.mark.parametrize("signed", [False, True])
def test_real_fit_learns_different_fund_bias_only_from_mature_labels(monkeypatch, signed):
    rng = np.random.default_rng(17089)
    rows = []
    for day in range(504):
        point = rng.normal(size=3).tolist()
        for i, code in enumerate(s.CODE_ORDER):
            target = str(date(2021, 1, 1) + timedelta(days=day))
            rows.append(
                {
                    "code": code,
                    "family": code,
                    "group": "CN_EQUITY",
                    "u": target,
                    "mature": target,
                    "y": int((i == 0) != (day % 10 == 0)),
                    "market": {"features": point, "available": True, "etf_available": True, "cnya_available": True},
                }
            )
    monkeypatch.setattr(s.original, "training_rows", lambda values, cutoff: values)
    monkeypatch.setattr(s.data, "scope", lambda: [{"code": c, "group": "CN_EQUITY"} for c in s.CODE_ORDER])
    monkeypatch.setattr(s, "active", lambda: None)
    h = s.fit(rows, s.CANDIDATES[int(signed)], "2024-01-01")
    assert h["model"].coef_[0, 3] > 0 > h["model"].coef_[0, 4]
    assert h["fit_hash"] == b.digest(rows) and h["weight_hash"] == b.digest(s.regression.weights(rows).tolist())
    s.validate_head(h, s.CANDIDATES[int(signed)], "2024-01-01")
    rows[0]["mature"] = "2024-01-01"
    with pytest.raises(ValueError, match="TRAINING_MATURITY_INVALID"):
        s.fit(rows, s.CANDIDATES[int(signed)], "2024-01-01")


@pytest.mark.parametrize(
    "fault",
    ["unknown", "wrong_group", "wrong_order", "wrong_scale", "future_maturity", "nan", "wrong_C", "wrong_group_head"],
)
def test_wrong_fund_or_modified_model_cannot_silently_predict(fault):
    trained = bundle()
    h = trained[s.CANDIDATES[0]]["CN_EQUITY"]
    code = "001000"
    if fault == "unknown":
        code = "999999"
    elif fault == "wrong_group":
        h["allowed_codes"] = ["002000"]
        h["training_codes"] = ["002000"]
    elif fault == "wrong_order":
        h["code_order"].reverse()
    elif fault == "wrong_scale":
        h["bias_scale"] = 1.0
    elif fault == "future_maturity":
        h["max_mature_date"] = h["cutoff"]
    elif fault == "nan":
        h["model"].coef_[0, 3] = float("nan")
    elif fault == "wrong_C":
        h["model"].C = 1.0
    else:
        h["group"] = "CN_BOND"
    with pytest.raises(ValueError):
        s.answers(paired(), "CN_EQUITY", trained, code)


def test_missing_sources_keep_spx_fallback_and_controls_ignore_fund_bias():
    values = [paired(), paired(False)]
    before = deepcopy(values)
    heads = {n: head(bool(i)) for i, n in enumerate(s.CANDIDATES)}
    controls = [lr_head(False), lr_head(True)]
    out = s.batch(values, list(s.CODE_ORDER), heads, controls)
    original = s.reference.batch(values, controls)
    assert len(out) == 7 and values == before
    assert all(out[n] == original[n] for n in s.CONTROLS)
    assert all(out[n][1]["route"] == "SPX_SIGN_FALLBACK" for n in s.CANDIDATES)


def test_checkpoint_hash_mismatch_is_rejected_without_refit(parent_ready, monkeypatch):  # noqa: F811
    path = s.root() / "checkpoints" / f"2026-09-16-CN_EQUITY-{s.CANDIDATES[0]}.joblib"
    b.save(
        path.with_suffix(".json"),
        {"plan_hash": "P", "name": s.CANDIDATES[0], "cutoff": "2026-09-16", "group": "CN_EQUITY", "sha256": "wrong"},
    )
    joblib.dump(head(), path)
    monkeypatch.setattr(s, "fit", lambda *args: pytest.fail("must not refit"))
    with pytest.raises(ValueError, match="CHECKPOINT_CHANGED"):
        s.checkpoint([], s.CANDIDATES[0], "2026-09-16", "CN_EQUITY", "P")


@pytest.mark.parametrize("fault", ["swapped_fund_answer", "late_receipt", "code", "model", "input"])
def test_future_binds_answer_to_fund_and_actual_early_receipt(parent_ready, monkeypatch, fault):  # noqa: F811
    forecast(parent_ready)
    manifest = {"at": "2026-09-16T06:50:00+08:00", "plan_hash": "FUND_BIAS", "model_sha256": "BIAS_HASH"}
    trained = bundle()
    b.save(s.root() / "result.json", manifest)
    monkeypatch.setattr(s, "models", lambda: (manifest, trained))
    monkeypatch.setattr(s, "live_market", lambda source: paired())
    assert runtime.tick(s)["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-16/001000.json"
    initial = path.read_bytes()
    runtime.tick(s)
    assert path.read_bytes() == initial
    value = b.read(path)
    assert value["answers"][s.CANDIDATES[0]]["prediction"] == 1
    receipt = s.root() / "receipts/2026-09-16/001000.json"
    record = b.read(receipt)
    if fault == "late_receipt":
        record["readback_at"] = "2026-09-16T08:30:00+08:00"
    else:
        if fault == "swapped_fund_answer":
            value["answers"] = s.answers(paired(), "CN_EQUITY", trained, "002000")
        elif fault == "code":
            value["code"] = "002000"
        else:
            value[fault + "_hash"] = "CHANGED"
        b.save(path, value, replace=True)
        record["forecast_hash"] = b.digest(value)
    b.save(receipt, record, replace=True)
    assert runtime.report(s)["verified_forecasts"] == 0
    from scripts import direction_1d_sprint_market_fund_bias as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_market_ridge_direction"
