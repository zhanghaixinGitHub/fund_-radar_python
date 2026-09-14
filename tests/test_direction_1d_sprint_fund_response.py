"""验证个体反应训练的成熟日边界、共享先验及第五轮不可回写的提前答案。"""

from datetime import datetime, timedelta

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_fund_response as f
from app.services import direction_1d_sprint_overnight as o
from app.services import direction_1d_sprint_sparse as s


def sample_rows():
    result = []
    for i in range(180):
        day = (datetime(2023, 1, 1) + timedelta(days=i)).date()
        for code in ("a", "b"):
            x = [0.0] * 32
            x[30] = (i % 7 - 3) / 100
            x[31] = float(i % 11 != 0)
            y = int(i % 7 > 2)
            result.append(
                {
                    "code": code,
                    "family": code,
                    "group": "CN_EQUITY",
                    "u": str(day),
                    "mature": str(day + timedelta(days=1)),
                    "x": x,
                    "y": y if code == "a" else 1 - y,
                }
            )
    return result


def test_individual_response_and_blend_are_distinct():
    bundle = f.fit_bundle(sample_rows(), "2023-07-01")
    row = sample_rows()[-2]
    row["x"][30] = 0.05
    a = f.answers(bundle, row)
    other = f.answers(bundle, row | {"code": "b"})
    assert a["FUND_LR2_BAL504"]["research_score"] > other["FUND_LR2_BAL504"]["research_score"]
    gp, fp = [a[n]["research_score"] for n in ("GROUP_LR2_BAL504", "FUND_LR2_BAL504")]
    weight = 180 / (180 + 252)
    assert a["BLEND_LR2_BAL504"]["research_score"] == pytest.approx(weight * fp + (1 - weight) * gp)


def test_unmatured_labels_cannot_change_model_or_conditional_prior():
    rows = sample_rows()
    cutoff = "2023-06-10"
    before = f.fit_bundle(rows, cutoff)
    for r in rows:
        if r["mature"] >= cutoff:
            r["y"] = 1 - r["y"]
            r["x"] = [float("nan")] * 32
    after = f.fit_bundle(rows, cutoff)
    assert before["group_conditional_prior"] == after["group_conditional_prior"]
    for code in before["funds"]:
        left, right = before["funds"][code], after["funds"][code]
        assert left["fit_hash"] == right["fit_hash"]
        assert left["conditional_scores"] == right["conditional_scores"]
        np.testing.assert_array_equal(left["model"][-1].coef_, right["model"][-1].coef_)


def test_conditional_prior_does_not_double_count_product_shares():
    rows = sample_rows()
    expected = f.conditional_rates(rows)
    duplicated = rows + [dict(r) for r in rows if r["family"] == "a"]
    np.testing.assert_allclose(f.conditional_rates(duplicated), expected)
    bundle = f.fit_bundle(rows, "2023-07-01")
    fund = bundle["funds"]["a"]
    expected = (
        np.asarray(fund["conditional_ups"]) + 60 * np.asarray(bundle["group_conditional_prior"]["CN_EQUITY"])
    ) / (np.asarray(fund["conditional_counts"]) + 60)
    np.testing.assert_allclose(fund["conditional_scores"], expected)


def test_insufficient_fund_is_not_silently_excluded():
    with pytest.raises(ValueError, match="FUND_RESPONSE_TRAIN_INSUFFICIENT"):
        f.fit_bundle(sample_rows()[:100], "2023-07-01")


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
    b.save(o.root() / "live/2026-09-15/0700-response.json", observation)
    source = {
        "at": "2026-09-15T07:11:00+08:00",
        "u": "2026-09-15",
        "x": [0.1] * 32,
        "original_hash": b.digest(original),
        "observation_hash": b.digest(observation),
        "observation_file": "live/2026-09-15/0700-response.json",
        "answers": {"TREE32_252": {"prediction": 0}},
    }
    b.save(o.root() / "forward/2026-09-15/001000.json", source)
    b.save(
        o.root() / "receipts/2026-09-15/001000.json",
        {
            "readback_at": "2026-09-15T07:11:01+08:00",
            "status": "VERIFIED",
            "forecast_hash": b.digest(source),
        },
    )
    previous = {
        "at": "2026-09-15T07:12:00+08:00",
        "u": "2026-09-15",
        "parent_hash": b.digest(source),
        "original_hash": b.digest(original),
        "answers": {"SPX_SIGN": {"prediction": 0}},
    }
    b.save(s.root() / "forward/2026-09-15/001000.json", previous)
    b.save(
        s.root() / "receipts/2026-09-15/001000.json",
        {
            "readback_at": "2026-09-15T07:12:01+08:00",
            "status": "VERIFIED",
            "forecast_hash": b.digest(previous),
        },
    )
    b.save(f.root() / "result.json", {"winner": "FUND_LR2_BAL504"})
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 7, 15, tzinfo=b.ZONE))
    monkeypatch.setattr(o, "source", lambda: None)
    bundle = {
        "groups": {"CN_EQUITY": {"model": ConstantModel()}},
        "funds": {
            "001000": {
                "group": "CN_EQUITY",
                "model": ConstantModel(),
                "fit_dates": 504,
                "conditional_scores": [0.8] * 3,
            }
        },
    }
    monkeypatch.setattr(f, "models", lambda: ({"model_sha256": "abc"}, bundle))
    return tmp_path, original, previous


def test_future_answers_are_immutable_and_match_original_outcome(ready, monkeypatch):
    directory, original, previous = ready
    result = f.tick()
    assert result["verified_forecasts"] == 1 and result["pending"] == 1
    path = f.root() / "forward/2026-09-15/001000.json"
    before = path.read_bytes()
    assert b.read(path)["parent_hash"] == b.digest(previous)
    f.tick()
    assert path.read_bytes() == before
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {
            "y": 1,
            "actual_direction": "UP",
            "forecast_hash": b.digest(original),
        },
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    report = f.report()["matched_forward_metrics"]
    assert report["FUND_LR2_BAL504"]["accuracy"] == 1
    assert report["SPX_SIGN"]["accuracy"] == 0
    assert report["ORIGINAL7"]["count"] == report["GROUP_LR2_BAL504"]["count"]


def test_late_parent_receipt_is_rejected(ready):
    path = s.root() / "receipts/2026-09-15/001000.json"
    b.save(path, b.read(path) | {"readback_at": "2026-09-15T08:30:00+08:00"}, replace=True)
    with pytest.raises(ValueError, match="ROUND_04_PARENT_NOT_VERIFIED"):
        f.tick()


def test_own_save_crossing_deadline_is_not_counted(ready, monkeypatch):
    save = b.save

    def crossing(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == f.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", crossing)
    result = f.tick()
    assert result["verified_forecasts"] == 0 and result["invalid_or_late"] == 1
    assert result["missing_due_predictions"] == 1


def test_outside_window_no_source_or_model_read(ready, monkeypatch):
    monkeypatch.setattr(o, "source", lambda: pytest.fail("unexpected external read"))
    monkeypatch.setattr(f, "models", lambda: pytest.fail("unexpected inference"))
    for at in (datetime(2026, 9, 14, 21, tzinfo=b.ZONE), datetime(2026, 9, 17, 13, tzinfo=b.ZONE)):
        monkeypatch.setattr(b, "now", lambda at=at: at)
        assert f.tick()["verified_forecasts"] == 0


def test_model_digest_is_checked_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "ROOT", tmp_path)
    b.save(f.root() / "result.json", {"code": f.fingerprint(), "model_sha256": "wrong"})
    (f.root() / "models.joblib").write_bytes(b"invalid")
    monkeypatch.setattr(f.joblib, "load", lambda path: pytest.fail("untrusted artifact loaded"))
    with pytest.raises(ValueError, match="ROUND_05_MODEL_OR_CODE_CHANGED"):
        f.models()
