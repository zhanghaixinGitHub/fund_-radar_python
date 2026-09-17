"""行业相对收益的一日目标、原输入保留、成熟时点及未来证据绑定。"""

from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_sector as s

from test_direction_1d_sprint_hk import ConstantModel, markets
from test_direction_1d_sprint_hk import ready as hk_ready  # noqa: F401
from test_direction_1d_sprint_hk import rows as historical_rows
from test_direction_1d_sprint_sector_data import points
from test_direction_1d_sprint_sequence import ancestor_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401


def test_code_freeze_covers_real_sector_client_and_data_files(monkeypatch):
    monkeypatch.setattr(s.previous_round, "fingerprint", lambda: {"code": {}})
    code = s.fingerprint()["code"]
    assert len(code) == 8
    assert "app/integrations/tushare_sprint_sector.py" in code
    assert "app/services/direction_1d_sprint_sector_data.py" in code


def test_sector_appends_five_known_inputs_and_target_stays_one_day():
    x = [0.0] * 32
    sector_points = points()
    z = s.vector(x, "2026-09-14", "2026-09-15", markets(), sector_points)
    assert len(z) == 17 and z[:12] == s.hk.vector(x, "2026-09-14", "2026-09-15", markets())
    assert z[-5:] == s.sector_data.features(x, "2026-09-14", "2026-09-15", sector_points)
    with pytest.raises(ValueError, match="NOT_ADJACENT"):
        s.vector(x, "2026-09-14", "2026-09-16", markets(), sector_points)


def training_rows():
    return [r | {"z": r["z"] + [1.0, 0.1, -0.2, 0.3, -0.4]} for r in historical_rows()]


def test_future_labels_and_input_changes_do_not_change_fitting(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = training_rows()
    before = s.fit(rows, s.CANDIDATES[0], "2023-06-10")
    for row in rows:
        if row["mature"] >= "2023-06-10":
            row["z"], row["y"] = [float("nan")] * 17, 1 - row["y"]
    after = s.fit(rows, s.CANDIDATES[0], "2023-06-10")
    assert before["fit_hash"] == after["fit_hash"]
    assert before["mean"] == after["mean"] and before["scale"] == after["scale"]
    assert before["max_mature_date"] < "2023-06-10"
    np.testing.assert_allclose(
        before["model"].predict_proba([np.zeros(17)]), after["model"].predict_proba([np.zeros(17)]), atol=1e-12
    )


def test_natural_error_prior_is_preserved(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = training_rows()
    for i, row in enumerate(rows):
        direction = int(row["z"][0] >= 0)
        row["y"] = 1 - direction if i % 5 == 0 else direction
    assert s.fit(rows, s.CANDIDATES[0], "2023-07-01")["weighted_error_rate"] == pytest.approx(0.2)


def trained(score):
    return {"model": ConstantModel(score), "mean": [0.0] * 17, "scale": [1.0] * 17}


@pytest.mark.parametrize("move", [-0.01, 0, 0.01])
def test_correction_threshold_is_unchanged(move):
    z = [move] + [0.0] * 16
    baseline = int(move >= 0)
    assert s.answer(z, s.CANDIDATES[0], trained(0.55))["prediction"] == baseline
    assert s.answer(z, s.CANDIDATES[0], trained(0.551))["prediction"] == 1 - baseline


@pytest.fixture
def ready(hk_ready, monkeypatch):  # noqa: F811
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {s.CANDIDATES[0]: {"CN_EQUITY": trained(0.3)}}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    input_value = {
        "at": "2026-09-15T07:15:00+08:00",
        "base": "2026-09-14",
        "target": "2026-09-15",
        "rows": points(),
        "source": s.sector_data.SOURCE,
    }
    monkeypatch.setattr(s.sector_data, "capture", lambda at: input_value)
    monkeypatch.setattr(s.sector_data, "load", lambda target: input_value)
    return hk_ready


def test_old_target_is_not_backfilled(ready, monkeypatch):
    monkeypatch.setattr(s, "models", lambda: pytest.fail("old target model loading"))
    assert s.tick()["verified_forecasts"] == 0


def test_live_input_order_and_announcements_are_checked(ready):
    _, original, source = ready
    with pytest.raises(ValueError, match="SEQUENCE_LIVE_DATES_INVALID"):
        s.live_vector(source, original | {"inputs": original["inputs"][::-1]}, markets(), points())
    original["inputs"][0]["ann_date"] = "2026-09-16"
    with pytest.raises(ValueError, match="SEQUENCE_LIVE_DATES_INVALID"):
        s.live_vector(source, original, markets(), points())


def test_early_answers_are_immutable_and_outcomes_match(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    directory, original, _ = ready
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    before = path.read_bytes()
    s.tick()
    assert path.read_bytes() == before
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {"y": 1, "actual_direction": "UP", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    assert s.report()["matched_forward_metrics"][s.CANDIDATES[0]]["accuracy"] == 1


def test_late_readback_is_rejected(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    save = b.save

    def late(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == s.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", late)
    result = s.tick()
    assert result["verified_forecasts"] == 0 and result["invalid_or_late"] == 1


def test_changed_saved_sector_value_is_rejected(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    value["z"][-1] += 1
    b.save(path, value, replace=True)
    receipt = s.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt, b.read(receipt) | {"forecast_hash": b.digest(value)}, replace=True)
    with pytest.raises(ValueError, match="ROUND_37_VECTOR_CHANGED"):
        s.report()


def test_prior_failure_still_attempts_sector_branch(monkeypatch):
    from scripts import direction_1d_sprint_sector as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_sensitivity"
    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(entry.sector, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]
