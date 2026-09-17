"""有限历史技术指标的输入时点、常量边界、量纲与一日纠错隔离。"""

from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_technical as s

from test_direction_1d_sprint_hk import ConstantModel, markets
from test_direction_1d_sprint_hk import ready as hk_ready  # noqa: F401
from test_direction_1d_sprint_hk import rows as historical_rows
from test_direction_1d_sprint_sequence import ancestor_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401


def input_x(nav):
    x = b.vector(nav, True) + [0.0] * 14
    x[30], x[31] = 0.01, 1
    return x


def test_fully_flat_original_window_remains_ineligible():
    nav = np.ones(61)
    with pytest.raises(ValueError, match="FLAT_FEATURE_WINDOW"):
        s.technical_features([0.0] * 32, nav)


def test_flat_recent_tail_has_neutral_range_indicators():
    # 较早净值仍有变化，满足原样本资格；最近20点平盘只让区间指标取0。
    nav = np.concatenate([np.linspace(1, 2, 41), np.full(20, 2.0)])
    result = s.technical_features(input_x(nav), nav)
    assert np.isfinite(result).all()
    assert result[-2:] == pytest.approx([0.0, 0.0])
    assert result[0] > 0 and result[1] > 0


@pytest.mark.parametrize("direction", [-1, 1])
def test_monotone_nav_has_correct_oscillator_direction(direction):
    nav = 2 + direction * np.arange(61) / 100
    values = s.technical_features(input_x(nav), nav)
    assert values[:2] == pytest.approx([direction, direction])
    assert values[2] * direction > 0
    assert values[-1] == pytest.approx(direction)
    assert values[-2] * direction > 0
    assert max(abs(values[2]), abs(values[3])) <= 5 and abs(values[4]) <= 3


def test_nav_denomination_does_not_change_features():
    nav = np.exp(np.linspace(0, 0.4, 61) + 0.03 * np.sin(np.arange(61)))
    before = s.technical_features(input_x(nav), nav)
    scaled = nav * 10
    assert s.technical_features(input_x(scaled), scaled) == pytest.approx(before, abs=1e-10)


def test_ema_is_causal_and_extra_target_nav_is_rejected():
    nav = np.linspace(1, 2, 61)
    np.testing.assert_array_equal(s.ema(nav, 12)[:30], s.ema(nav[:30], 12))
    with pytest.raises(ValueError, match="SEQUENCE_INPUT_INVALID"):
        s.technical_features(input_x(nav), np.append(nav, 1000))
    x = input_x(nav)
    x[7] += 0.1
    with pytest.raises(ValueError, match="SEQUENCE_NAV_WINDOW_CHANGED"):
        s.technical_features(x, nav)


def test_existing_hk_inputs_are_preserved_and_horizon_is_one_day():
    nav = np.linspace(1, 2, 61)
    x = input_x(nav)
    z = s.vector(x, "2026-09-14", "2026-09-15", markets(), nav)
    assert len(z) == 18
    assert z[:12] == s.hk.vector(x, "2026-09-14", "2026-09-15", markets())
    with pytest.raises(ValueError, match="TARGET_NOT_ADJACENT"):
        s.vector(x, "2026-09-14", "2026-09-16", markets(), nav)


def training_rows():
    return [r | {"z": r["z"] + [0.1] * 6} for r in historical_rows()]


def test_future_labels_and_input_changes_do_not_change_fitting(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = training_rows()
    before = s.fit(rows, s.CANDIDATES[0], "2023-06-10")
    for row in rows:
        if row["mature"] >= "2023-06-10":
            row["z"], row["y"] = [float("nan")] * 18, 1 - row["y"]
    after = s.fit(rows, s.CANDIDATES[0], "2023-06-10")
    assert before["fit_hash"] == after["fit_hash"]
    assert before["mean"] == after["mean"] and before["scale"] == after["scale"]
    assert before["max_mature_date"] < "2023-06-10"
    np.testing.assert_allclose(
        before["model"].predict_proba([np.zeros(18)]), after["model"].predict_proba([np.zeros(18)]), atol=1e-12
    )


def test_natural_error_prior_is_preserved(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = training_rows()
    for i, row in enumerate(rows):
        direction = int(row["z"][0] >= 0)
        row["y"] = 1 - direction if i % 5 == 0 else direction
    assert s.fit(rows, s.CANDIDATES[0], "2023-07-01")["weighted_error_rate"] == pytest.approx(0.2)


def trained(score):
    return {"model": ConstantModel(score), "mean": [0.0] * 18, "scale": [1.0] * 18}


@pytest.mark.parametrize("move", [-0.01, 0, 0.01])
def test_correction_threshold_is_unchanged(move):
    z = [move] + [0.0] * 17
    baseline = int(move >= 0)
    assert s.answer(z, s.CANDIDATES[0], trained(0.55))["prediction"] == baseline
    assert s.answer(z, s.CANDIDATES[0], trained(0.551))["prediction"] == 1 - baseline


@pytest.fixture
def ready(hk_ready, monkeypatch):  # noqa: F811
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {s.CANDIDATES[0]: {"CN_EQUITY": trained(0.3)}}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    return hk_ready


def test_old_target_is_not_backfilled(ready, monkeypatch):
    monkeypatch.setattr(s, "models", lambda: pytest.fail("old target model loading"))
    assert s.tick()["verified_forecasts"] == 0


def test_live_input_order_and_announcements_are_checked(ready):
    _, original, source = ready
    with pytest.raises(ValueError, match="SEQUENCE_LIVE_DATES_INVALID"):
        s.live_vector(source, original | {"inputs": original["inputs"][::-1]}, markets())
    original["inputs"][0]["ann_date"] = "2026-09-16"
    with pytest.raises(ValueError, match="SEQUENCE_LIVE_DATES_INVALID"):
        s.live_vector(source, original, markets())


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


def test_changed_saved_technical_value_is_rejected(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    value["z"][-1] += 1
    b.save(path, value, replace=True)
    receipt = s.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt, b.read(receipt) | {"forecast_hash": b.digest(value)}, replace=True)
    with pytest.raises(ValueError, match="ROUND_19_VECTOR_CHANGED"):
        s.report()


def test_prior_failure_still_attempts_technical_branch(monkeypatch):
    from scripts import direction_1d_sprint_technical as entry

    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(entry.technical, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]
