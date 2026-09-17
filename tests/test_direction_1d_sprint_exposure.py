"""滚动反应特征的时间隔离、收缩权重和新分支真实预测边界。"""

from datetime import datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_exposure as e

from test_direction_1d_sprint_dual_us import parent_ready  # noqa: F401
from test_direction_1d_sprint_dual_us import ready as previous_ready  # noqa: F401
from test_direction_1d_sprint_fund_response import sample_rows


def history_rows():
    return [r | {"return_target": r["x"][30] * (30 if r["code"] == "a" else -30)} for r in sample_rows()]


def test_current_and_future_labels_do_not_change_past_exposure_features():
    rows = history_rows()
    cutoff = "2023-06-10"
    before = e.ExposureHistory(rows).snapshot(cutoff)
    assert before["max_mature"] < cutoff
    for row in rows:
        if row["mature"] >= cutoff:
            row["return_target"] = float("nan")
            row["x"] = [float("nan")] * 32
            row["y"] = 1 - row["y"]
    after = e.ExposureHistory(rows).snapshot(cutoff)
    assert before == after
    assert before["funds"]["a"]["slope"] > 0 > before["funds"]["b"]["slope"]
    assert before["funds"]["a"]["dates"] == 126


def test_duplicate_share_does_not_change_group_prior_or_existing_fund_features():
    rows = history_rows()
    left = e.ExposureHistory(rows).snapshot("2023-07-01")
    extra = [r | {"code": "a_other_share"} for r in rows if r["code"] == "a"]
    right = e.ExposureHistory(rows + extra).snapshot("2023-07-01")
    for code in ("a", "b"):
        for key in ("slope", "mean_s", "mean_return", "up_rate"):
            assert left["funds"][code][key] == pytest.approx(right["funds"][code][key])


def test_cold_start_is_neutral_without_dropping_any_fund():
    snapshot = e.ExposureHistory(history_rows()).snapshot("2023-01-01")
    assert set(snapshot["funds"]) == {"a", "b"}
    assert snapshot["funds"]["a"]["slope"] == 0
    assert snapshot["funds"]["a"]["up_rate"] == 0.5


@pytest.mark.parametrize("name", e.LEARNED)
def test_unmatured_rows_never_change_fitted_head(name):
    rows = e.dataset(history_rows())
    first = e.fit(rows, name, "2023-06-10")
    for row in rows:
        if row["mature"] >= "2023-06-10":
            row["z"] = [float("nan")] * 6
            row["return_target"] = float("nan")
            row["y"] = 1 - row["y"]
    second = e.fit(rows, name, "2023-06-10")
    assert first["fit_hash"] == second["fit_hash"]
    np.testing.assert_array_equal(first["model"][-1].coef_, second["model"][-1].coef_)


class ConstantModel:
    def predict_proba(self, x):
        return np.asarray([[0.7, 0.3]] * len(x))

    def predict(self, x):
        return np.full(len(x), -0.3)


@pytest.fixture
def ready(previous_ready, monkeypatch):  # noqa: F811
    b.save(e.root() / "result.json", {"winner": e.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {n: {"CN_EQUITY": {"model": ConstantModel()}} for n in e.LEARNED}
    bundle["context"] = {
        "cutoff": "2026-09-14",
        "funds": {
            "001000": {
                "slope": 0.2,
                "mean_return": 0.01,
                "mean_s": 0,
                "up_rate": 0.5,
            }
        },
    }
    monkeypatch.setattr(e, "models", lambda: ({"model_sha256": "abc"}, bundle))
    return previous_ready


def test_old_target_is_not_backfilled(ready, monkeypatch):
    monkeypatch.setattr(e, "models", lambda: pytest.fail("old target cannot load new model"))
    assert e.tick()["verified_forecasts"] == 0


def test_saved_new_answers_are_immutable_and_post_cutoff_cannot_generate(ready, monkeypatch):
    monkeypatch.setattr(e, "FIRST_TARGET", "2026-09-15")
    assert e.tick()["verified_forecasts"] == 1
    path = e.root() / "forward/2026-09-15/001000.json"
    saved = path.read_bytes()
    assert len(b.read(path)["answers"]) == 4
    e.tick()
    assert path.read_bytes() == saved
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))
    monkeypatch.setattr(e, "models", lambda: pytest.fail("post cutoff"))
    assert e.tick()["verified_forecasts"] == 1


def test_future_context_blocks_prediction(ready, monkeypatch):
    monkeypatch.setattr(e, "FIRST_TARGET", "2026-09-15")
    e.models()[1]["context"]["cutoff"] = "2026-09-15"
    with pytest.raises(ValueError, match="EXPOSURE_FORWARD_CONTEXT_NOT_PRIOR"):
        e.tick()


def test_runner_keeps_new_branch_when_old_runner_fails(monkeypatch):
    from scripts import direction_1d_sprint_exposure as entry

    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(entry.exposure, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]
