"""验证第四轮时间隔离、权重口径、父证据继承及不可改写的未来记录。"""

from datetime import datetime, timedelta

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_overnight as p
from app.services import direction_1d_sprint_sparse as s


def test_feature_subset_and_rule_have_explicit_meaning():
    x = list(range(32))
    assert s.vector(x, "LR8_BAL252") == [30, 31, 7, 0, 1, 3, 18, 23]
    assert s.vector(x, "LR2_BAL252") == [30, 31]
    for value, prediction in [(-0.01, 0), (0.0, 1), (0.01, 1)]:
        x[30] = value
        answer = s.answer(x, "SPX_SIGN")
        assert answer["prediction"] == prediction
        assert answer["research_score"] is None
    x[30] = float("nan")
    with pytest.raises(ValueError, match="SPARSE_INPUT_INVALID"):
        s.vector(x, "LR2_BAL252")


def test_class_balance_and_duplicate_family_weight():
    rows = [
        {"u": "2024-01-01", "family": "a", "y": 0},
        {"u": "2024-01-01", "family": "b", "y": 1},
        {"u": "2024-01-02", "family": "a", "y": 0},
    ]
    weights = s.training_weights(rows)
    assert weights[[0, 2]].sum() == pytest.approx(weights[1])
    duplicated = rows + [dict(rows[0])]
    changed = s.training_weights(duplicated)
    assert changed[0] + changed[3] == pytest.approx(weights[0])
    np.testing.assert_allclose(changed[1:3], weights[1:3])
    assert changed.sum() == pytest.approx(weights.sum())


def training_rows():
    rows = []
    first = datetime(2023, 1, 1)
    for i in range(180):
        day = (first + timedelta(days=i)).date()
        for j in range(2):
            x = [0.0] * 32
            x[30] = (i % 7 - 3) / 100
            x[31] = float(i % 11 != 0)
            rows.append(
                {
                    "u": str(day),
                    "mature": str(day + timedelta(days=1)),
                    "family": str(j),
                    "code": str(j),
                    "group": "CN_EQUITY",
                    "y": int(i % 7 > 2),
                    "x": x,
                }
            )
    return rows


def test_fit_never_reads_unmatured_labels_or_features():
    rows = training_rows()
    cutoff = "2023-06-10"
    before = s.fit(rows, "LR2_BAL252", cutoff)
    for r in rows:
        if r["mature"] >= cutoff:
            r["y"] = 1 - r["y"]
            r["x"] = [float("nan")] * 32
    after = s.fit(rows, "LR2_BAL252", cutoff)
    assert before["fit_hash"] == after["fit_hash"]
    np.testing.assert_array_equal(before["model"][-1].coef_, after["model"][-1].coef_)
    assert after["fit_end"] < cutoff


def test_tree_equity_response_to_spx_is_nondecreasing():
    trained = s.fit(training_rows(), "TREE8_BAL252", "2023-07-01")
    x = [0.0] * 32
    x[31] = 1
    scores = []
    for move in np.linspace(-0.05, 0.05, 21):
        x[30] = move
        scores.append(s.answer(x, "TREE8_BAL252", trained)["research_score"])
    assert np.all(np.diff(scores) >= -1e-12)


class ConstantModel:
    def predict_proba(self, x):
        return np.asarray([[0.2, 0.8]] * len(x))


@pytest.fixture
def ready(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 14, 12, tzinfo=b.ZONE))
    b.initialize()
    b.save(tmp_path / "scope.json", {"codes": ["001000", "002000"]})
    b.save(tmp_path / "history.json", {"funds": [{"fund_code": "001000"}]})
    original = {
        "at": "2026-09-14T20:00:00+08:00",
        "base": "2026-09-14",
        "u": "2026-09-15",
        "code": "001000",
        "group": "CN_EQUITY",
        "family": "f",
        "answers": {"ORIGINAL7": {"prediction": 0}},
    }
    b.save(tmp_path / "forward/2026-09-15/001000.json", original)
    observation = {"received_at": "2026-09-15T07:10:00+08:00", "parsed": {"status": "COMPLETE"}}
    b.save(p.root() / "live/2026-09-15/0700-response.json", observation)
    previous = {
        "at": "2026-09-15T07:11:00+08:00",
        "u": "2026-09-15",
        "x": [0.1] * 32,
        "original_hash": b.digest(original),
        "observation_hash": b.digest(observation),
        "observation_file": "live/2026-09-15/0700-response.json",
        "answers": {"TREE32_252": {"prediction": 0}},
    }
    b.save(p.root() / "forward/2026-09-15/001000.json", previous)
    b.save(
        p.root() / "receipts/2026-09-15/001000.json",
        {
            "readback_at": "2026-09-15T07:11:01+08:00",
            "status": "VERIFIED",
            "forecast_hash": b.digest(previous),
        },
    )
    b.save(s.root() / "result.json", {"winner": "LR2_BAL252"})
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 7, 15, tzinfo=b.ZONE))
    monkeypatch.setattr(p, "source", lambda: None)
    bundle = {name: {"CN_EQUITY": {"model": ConstantModel()}} for name in s.CANDIDATES if name != "SPX_SIGN"}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    return tmp_path, original, previous


def test_real_input_reuse_is_immutable_and_outcomes_are_paired(ready, monkeypatch):
    directory, original, previous = ready
    result = s.tick()
    assert result["verified_forecasts"] == 1 and result["pending"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    raw = path.read_bytes()
    assert b.read(path)["parent_hash"] == b.digest(previous)
    assert b.read(path)["answers"]["SPX_SIGN"]["research_score"] is None
    s.tick()
    assert path.read_bytes() == raw
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {
            "y": 1,
            "actual_direction": "UP",
            "forecast_hash": b.digest(original),
        },
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    report = s.report()["matched_forward_metrics"]
    assert report["LR2_BAL252"]["accuracy"] == 1
    assert report["TREE32_252"]["accuracy"] == 0
    assert report["ORIGINAL7"]["count"] == report["SPX_SIGN"]["count"]


def test_parent_after_deadline_is_rejected(ready):
    path = p.root() / "receipts/2026-09-15/001000.json"
    receipt = b.read(path) | {"readback_at": "2026-09-15T08:30:00+08:00"}
    b.save(path, receipt, replace=True)
    with pytest.raises(ValueError, match="PARENT_NOT_VERIFIED_BEFORE_CUTOFF"):
        s.tick()


def test_changed_observation_is_rejected(ready):
    path = p.root() / "live/2026-09-15/0700-response.json"
    value = b.read(path) | {"received_at": "2026-09-15T08:31:00+08:00"}
    b.save(path, value, replace=True)
    with pytest.raises(ValueError, match="PARENT_NOT_VERIFIED_BEFORE_CUTOFF"):
        s.tick()


def test_persistence_crossing_cutoff_is_retained_but_invalid(ready, monkeypatch):
    original_save = b.save

    def crossing(path, payload, **kwargs):
        original_save(path, payload, **kwargs)
        if path.parent.parent == s.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", crossing)
    report = s.tick()
    assert report["verified_forecasts"] == 0 and report["invalid_or_late"] == 1
    assert report["missing_due_predictions"] == 1


def test_no_inference_outside_forecast_window(ready, monkeypatch):
    monkeypatch.setattr(s, "models", lambda: pytest.fail("unexpected inference"))
    monkeypatch.setattr(p, "source", lambda: pytest.fail("unexpected external read"))
    for at in [datetime(2026, 9, 14, 21, tzinfo=b.ZONE), datetime(2026, 9, 17, 13, tzinfo=b.ZONE)]:
        monkeypatch.setattr(b, "now", lambda at=at: at)
        assert s.tick()["verified_forecasts"] == 0


def test_tampered_model_cannot_be_deserialized(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    b.save(s.root() / "result.json", {"code": s.fingerprint(), "model_sha256": "wrong"})
    (s.root() / "models.joblib").write_bytes(b"invalid")
    monkeypatch.setattr(s.joblib, "load", lambda path: pytest.fail("untrusted artifact loaded"))
    with pytest.raises(ValueError, match="ROUND_04_MODEL_OR_CODE_CHANGED"):
        s.models()
