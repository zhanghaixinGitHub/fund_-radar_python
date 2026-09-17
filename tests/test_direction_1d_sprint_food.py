"""单基金模型路由、成熟样本和严格纠错阈值，其他基金必须保持父模型行为。"""

from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_food as s

from test_direction_1d_sprint_food_data import points
from test_direction_1d_sprint_hk import ConstantModel, markets
from test_direction_1d_sprint_hk import ready as hk_ready  # noqa: F401
from test_direction_1d_sprint_hk import rows as historical_rows
from test_direction_1d_sprint_sequence import ancestor_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401


def trained(score):
    return {
        "control": {"model": ConstantModel(0.56), "mean": [0.0] * 12, "scale": [1.0] * 12},
        "food": {"model": ConstantModel(score), "mean": [0.0] * 16, "scale": [1.0] * 16},
    }


def test_nonfood_predictions_remain_exact_hk_even_if_food_model_changes():
    z = [-1.0] + [0.0] * 16
    for score in (0.0, 0.55, 1.0):
        t = trained(score)
        assert s.answer(z, s.CANDIDATES[0], t) == s.hk.answer(z[:12], s.CONTROL, t["control"])


@pytest.mark.parametrize("move", [-0.01, 0.0, 0.01])
def test_only_food_route_uses_strict_error_gate(move):
    z = [move] + [0.0] * 15 + [1.0]
    for score in (0.55, 0.5500001):
        answer = s.answer(z, s.CANDIDATES[0], trained(score))
        expected = int(move >= 0)
        assert answer["prediction"] == (1 - expected if score > 0.55 else expected)
        assert answer["kind"] == "UNCALIBRATED_BASELINE_ERROR_SCORE"
    with pytest.raises(ValueError, match="GROUP_ROUTE"):
        s.answer(z, s.CANDIDATES[0], trained(0.6) | {"food": None})


def test_invalid_route_does_not_silently_change_fund_identity():
    for tail in ([0.0, 0.0, 0.0, 0.0, 2.0], [1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, float("nan"), 0.0, 1.0]):
        with pytest.raises(ValueError):
            s.selected_features([0.0] * 12 + tail, s.CANDIDATES[0])


def test_live_and_historical_features_use_one_day_target():
    z = s.vector([0.0] * 32, "001632", "2026-09-14", "2026-09-15", markets(), points())
    assert len(z) == 17 and z[-1] == 1
    assert z[:12] == s.hk.vector([0.0] * 32, "2026-09-14", "2026-09-15", markets())
    with pytest.raises(ValueError, match="NOT_ADJACENT"):
        s.vector([0.0] * 32, "001632", "2026-09-14", "2026-09-16", markets(), points())


def test_future_and_other_fund_rows_cannot_change_food_fit(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = [r | {"code": "001632", "z": r["z"] + [0.0] * 4 + [1.0]} for r in historical_rows()]
    before = s.fit(rows, s.CANDIDATES[0], "2023-06-10")
    for row in rows:
        if row["mature"] >= "2023-06-10":
            row["z"], row["y"] = [float("nan")] * 17, 1 - row["y"]
    rows.append(rows[0] | {"code": "007045", "z": [float("nan")] * 17})
    after = s.fit(rows, s.CANDIDATES[0], "2023-06-10")
    assert before["fit_hash"] == after["fit_hash"] and after["max_mature_date"] < "2023-06-10"
    assert after["model"].n_features_in_ == 16
    np.testing.assert_allclose(
        before["model"].predict_proba([np.zeros(16)]), after["model"].predict_proba([np.zeros(16)])
    )


def test_zero_flag_really_excludes_benchmark_input():
    z = s.vector([0.0] * 32, "007045", "2026-09-14", "2026-09-15", markets(), {})
    assert z[-5:] == [0.0] * 5


def test_feature_ablation_keeps_food_route_and_drops_only_four_features():
    z = [float(i) for i in range(16)] + [1.0]
    assert s.selected_features(z, s.CANDIDATES[0]) == z[:16]
    assert s.selected_features(z, s.CANDIDATES[1]) == z[:12]


@pytest.fixture
def ready(hk_ready, monkeypatch):  # noqa: F811
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    model = trained(0.7)
    model["control"]["model"] = ConstantModel(0.4)
    bundle = {n: {"CN_EQUITY": model} for n in s.CANDIDATES}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    input_value = {
        "at": "2026-09-15T07:15:00+08:00",
        "base": "2026-09-14",
        "target": "2026-09-15",
        "rows": points(),
        "source": s.food_data.SOURCE,
    }
    monkeypatch.setattr(s.food_data, "capture", lambda at: input_value)
    monkeypatch.setattr(s.food_data, "load", lambda target: input_value)
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


def test_changed_saved_fxi_value_is_rejected(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    value["z"][-1] += 1
    b.save(path, value, replace=True)
    receipt = s.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt, b.read(receipt) | {"forecast_hash": b.digest(value)}, replace=True)
    with pytest.raises(ValueError, match="ROUND_46_VECTOR_CHANGED"):
        s.report()


def test_prior_failure_still_attempts_fxi_branch(monkeypatch):
    from scripts import direction_1d_sprint_food as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_cnya_neutral"
    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(entry.food, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]
