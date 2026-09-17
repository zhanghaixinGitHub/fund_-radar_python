"""冲突路由只改变指定题目；成熟窗口、父答案和真实预测证据仍受约束。"""

from datetime import datetime
from itertools import product

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_etf_disagreement as s

from test_direction_1d_sprint_etf_joint import ready as joint_ready  # noqa: F401
from test_direction_1d_sprint_etf_joint import trained, vector
from test_direction_1d_sprint_hk import ConstantModel
from test_direction_1d_sprint_hk import ready as hk_ready  # noqa: F401
from test_direction_1d_sprint_hk import rows as historical_rows
from test_direction_1d_sprint_sequence import ancestor_ready  # noqa: F401
from test_direction_1d_sprint_sequence import ready as sequence_ready  # noqa: F401
from test_direction_1d_sprint_us_etf_direct import ready as direct_ready  # noqa: F401


@pytest.mark.parametrize("values", list(product((-1.0, 0.0, 1.0), repeat=3)))
def test_only_conflict_answers_use_learned_head(values):
    z = vector(values)
    for score in (0.499999, 0.5):
        model = trained()
        model["us_etf"]["model"] = ConstantModel(score)
        actual = s.answer(z, s.CANDIDATES[0], model)
        if (values[0] >= 0) != (values[1] >= 0):
            assert actual["prediction"] == int(score >= 0.5)
            assert actual["kind"] == "UNCALIBRATED_UP_SCORE"
        else:
            assert actual == s.previous_round.answer(z, s.previous_round.CANDIDATES[0], model)


def test_missing_source_keeps_parent_hk_even_with_conflicting_numbers():
    z = vector((1, -1, -1))
    z[17:20] = [0.0] * 3
    model = trained()
    assert s.answer(z, s.CANDIDATES[0], model) == s.previous_round.answer(z, s.previous_round.CANDIDATES[0], model)


def test_mature_conflict_training_excludes_agreement_labels(monkeypatch):
    monkeypatch.setattr(s, "active", lambda: None)
    rows = []
    for r in historical_rows():
        # 同一市场日期所有基金使用同一个信号状态，避免测试造出不真实的基金间市场分歧。
        i = datetime.fromisoformat(r["u"]).toordinal()
        z = r["z"] + [0.0] * 8
        z[0] = 1.0 if i % 2 else -1.0
        z[12] = -z[0] if i % 5 else z[0]
        z[17] = 0.3 if i % 3 else -0.3
        z[16] = z[19] = 1.0
        rows.append(r | {"z": z})
    before = s.fit(rows, s.CANDIDATES[0], "2023-06-10")
    for r in rows:
        if r["mature"] >= "2023-06-10":
            r["z"], r["y"] = [float("nan")] * 20, 1 - r["y"]
        elif not s.disagreement(r["z"]):
            r["y"] = 1 - r["y"]
    after = s.fit(rows, s.CANDIDATES[0], "2023-06-10")
    assert before["fit_hash"] == after["fit_hash"]
    assert before["fit_dates"] < before["parent_window_dates"]
    assert before["model"].fit_intercept is False and before["mean"] == [0.0] * 3
    np.testing.assert_allclose(before["model"].coef_, after["model"].coef_)


@pytest.fixture
def ready(joint_ready, monkeypatch):  # noqa: F811
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {s.CANDIDATES[0]: {"CN_EQUITY": trained()}}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    return joint_ready


def test_future_answers_and_original_outcomes_are_bound_and_immutable(ready, monkeypatch):
    directory, original, _ = ready
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    before = path.read_bytes()
    value = b.read(path)
    assert s.tick()["verified_forecasts"] == 1 and path.read_bytes() == before
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {"y": 1, "actual_direction": "UP", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    assert (
        s.report()["matched_forward_metrics"][s.CANDIDATES[0]]["accuracy"]
        == value["answers"][s.CANDIDATES[0]]["prediction"]
    )


def test_changed_prediction_cannot_be_hidden_by_new_receipt_hash(ready):
    s.tick()
    path = s.root() / "forward/2026-09-15/001000.json"
    value = b.read(path)
    value["answers"][s.CANDIDATES[0]]["prediction"] ^= 1
    b.save(path, value, replace=True)
    receipt = s.root() / "receipts/2026-09-15/001000.json"
    b.save(receipt, b.read(receipt) | {"forecast_hash": b.digest(value)}, replace=True)
    with pytest.raises(ValueError, match="SAVED_ANSWER_CHANGED"):
        s.report()


def test_late_readback_is_rejected(ready, monkeypatch):
    save = b.save

    def late(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == s.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", late)
    value = s.tick()
    assert value["verified_forecasts"] == 0 and value["invalid_or_late"] == 1


def test_entry_keeps_all_parent_rounds_and_attempts_new_model_after_failure(monkeypatch):
    from scripts import direction_1d_sprint_etf_disagreement as entry

    assert entry.existing.__name__ == "scripts.direction_1d_sprint_etf_joint"
    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(s, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *a, **kw: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]
