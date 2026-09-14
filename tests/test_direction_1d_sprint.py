"""验证三天研究的时间隔离、评分口径、不可改写答案和模型完整性。"""

import copy
import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pytest
from app.services import direction_1d_sprint as s


class ConstantModel:
    def predict_proba(self, x):
        return np.array([[0.3, 0.7]] * len(x))


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "ROOT", tmp_path)
    monkeypatch.setattr(s, "now", lambda: datetime(2026, 9, 14, 12, tzinfo=s.ZONE))
    s.initialize()
    wanted = list(map(str, s.input_days(date(2026, 9, 14))))
    fund = {
        "fund_code": "001000",
        "family": "family-a",
        "group": "CN_EQUITY",
        "rows": [{"date": d, "nav": str(1 + i * 0.001), "ann_date": d} for i, d in enumerate(wanted)],
    }
    history = {"at": s.now().isoformat(), "expires_at": "2027-01-01T00:00:00+08:00", "funds": [fund], "errors": []}
    s.save(tmp_path / "history.json", history)
    s.save(tmp_path / "scope.json", {"codes": ["001000", "002000"]})
    (tmp_path / "round-01").mkdir()
    model_path = tmp_path / "round-01/models.joblib"
    joblib.dump({"TEST": {"CN_EQUITY": {"model": ConstantModel(), "n": 18}}}, model_path)
    s.save(
        tmp_path / "round-01/result.json",
        {
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "source_code_hash": hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest(),
        },
    )
    monkeypatch.setattr(s, "refresh", lambda: copy.deepcopy(history))
    return tmp_path, history


def test_future_label_change_does_not_change_features():
    days = s.calendar()[0][100:163]
    fund = {
        "fund_code": "1",
        "family": "f",
        "group": "CN_EQUITY",
        "rows": [{"date": str(d), "nav": str(1 + i * 0.001), "ann_date": str(d)} for i, d in enumerate(days)],
    }
    first = s.samples({"funds": [fund]})[0]
    fund["rows"][61]["nav"] = "0.5"
    changed = s.samples({"funds": [fund]})[0]
    assert first["x"] == changed["x"]
    assert first["y"] != changed["y"]
    assert len(first["x"]) == 18


def test_late_announcement_rejects_historical_input():
    days = s.calendar()[0][100:163]
    fund = {
        "fund_code": "1",
        "family": "f",
        "group": "CN_EQUITY",
        "rows": [{"date": str(d), "nav": str(1 + i * 0.001), "ann_date": "2027-01-01"} for i, d in enumerate(days)],
    }
    assert s.samples({"funds": [fund]}) == []


def test_family_day_weighting_is_not_share_count_weighting():
    rows = [{"u": "2025-01-01", "family": "same", "y": 1, "prediction": 1}] * 10
    rows += [{"u": "2025-01-01", "family": "other", "y": 0, "prediction": 1}]
    result = s.metrics(rows)
    assert result["accuracy"] == 0.5
    assert result["raw_accuracy"] == 10 / 11


def test_fit_excludes_future_mature_labels():
    days = s.calendar()[0][100:241]
    rows = [
        {"u": str(d), "mature": str(d + timedelta(days=2)), "family": "f", "x": [i / 100, *([0.1] * 17)], "y": i % 2}
        for i, d in enumerate(days)
    ]
    cutoff = str(days[-1])
    original = s.fit(rows, "LR18_252", cutoff)
    changed = [r | {"y": 1 - r["y"]} if r["mature"] >= cutoff else r for r in rows]
    replay = s.fit(changed, "LR18_252", cutoff)
    assert original["fit_end"] < cutoff
    np.testing.assert_array_equal(original["model"][-1].coef_, replay["model"][-1].coef_)


def test_tampered_evidence_is_rejected(tmp_path):
    path = tmp_path / "record.json"
    s.save(path, {"a": 1})
    path.write_text(path.read_text(encoding="utf-8").replace('"a":1', '"a":2'), encoding="utf-8")
    with pytest.raises(ValueError, match="EVIDENCE_HASH_MISMATCH"):
        s.read(path)


def test_predictions_are_immutable_and_outcomes_wait(isolated, monkeypatch):
    root, history = isolated
    monkeypatch.setattr(s, "now", lambda: datetime(2026, 9, 14, 20, tzinfo=s.ZONE))
    result = s.tick()
    path = root / "forward/2026-09-15/001000.json"
    before = path.read_bytes()
    assert result["future_forecasts"] == 1 and result["mature_outcomes"] == 0
    history["funds"][0]["rows"][-1]["nav"] = "0.8"
    s.tick()
    assert path.read_bytes() == before
    history["funds"][0]["rows"].append({"date": "2026-09-15", "nav": "1.1", "ann_date": "2026-09-15"})
    monkeypatch.setattr(s, "now", lambda: datetime(2026, 9, 15, 12, tzinfo=s.ZONE))
    assert s.tick()["mature_outcomes"] == 0
    monkeypatch.setattr(s, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=s.ZONE))
    assert s.tick()["mature_outcomes"] == 1
    assert s.read(root / "outcomes/2026-09-15/001000.json")["base_revision_detected"] is True


def test_missed_deadline_does_not_backfill(isolated, monkeypatch):
    root, _ = isolated
    monkeypatch.setattr(s, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=s.ZONE))
    result = s.tick()
    assert result["future_forecasts"] == 0
    assert result["missing_eligible_predictions"] == 1
    assert result["whole_watchlist_coverage"] == 0
    assert not list((root / "forward").glob("*/*.json"))


def test_model_tampering_fails_before_deserialization(isolated, monkeypatch):
    root, _ = isolated
    monkeypatch.setattr(s, "now", lambda: datetime(2026, 9, 14, 20, tzinfo=s.ZONE))
    (root / "round-01/models.joblib").write_bytes(b"untrusted bytes")
    monkeypatch.setattr(s.joblib, "load", lambda _: pytest.fail("untrusted model was loaded"))
    with pytest.raises(ValueError, match="MODEL_HASH_MISMATCH"):
        s.tick()


def test_checkpoint_does_not_call_provider(isolated, monkeypatch):
    monkeypatch.setattr(s, "now", lambda: datetime(2026, 9, 17, 13, tzinfo=s.ZONE))
    monkeypatch.setattr(s, "refresh", lambda: pytest.fail("provider called after deadline"))
    result = s.tick()
    assert result["phase"] == "CHECKPOINT_REACHED"
    assert result["missing_eligible_predictions"] == 3
    assert result["model_released"] is False
