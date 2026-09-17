"""同伴必须在作答前可见；不能让下一日标签、重复份额或晚到输入改变过去特征。"""

from datetime import date, datetime

import numpy as np
import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_breadth as s
from app.services import direction_1d_sprint_overnight as o

from test_direction_1d_sprint_dual_us import parent_ready  # noqa: F401
from test_direction_1d_sprint_dual_us import ready as previous_ready  # noqa: F401
from test_direction_1d_sprint_fund_response import sample_rows


def history():
    days = list(map(str, b.input_days(date(2025, 1, 10))))
    funds = []
    for code, offset in (("a", 0.0001), ("b", -0.0001)):
        rows = [
            {"date": day, "ann_date": day, "nav": str(1 + offset * i + 0.00003 * (i % 5))} for i, day in enumerate(days)
        ]
        funds.append({"fund_code": code, "family": code, "group": "CN_EQUITY", "rows": rows})
    return {"funds": funds}


def test_peer_membership_does_not_require_next_day_nav_or_label():
    raw = history()
    before = s.PeerHistory(raw).snapshot("2025-01-10", "2025-01-13")
    raw["funds"][1]["rows"].append({"date": "2025-01-13", "ann_date": "2025-01-14", "nav": "999"})
    after = s.PeerHistory(raw).snapshot("2025-01-10", "2025-01-13")
    assert before == after
    assert set(before["peers"]) == {"a", "b"}


def test_unavailable_base_input_reduces_reported_coverage_without_losing_own_input():
    raw = history()
    raw["funds"][1]["rows"][-1]["ann_date"] = "2025-01-14"
    snapshot = s.PeerHistory(raw).snapshot("2025-01-10", "2025-01-13")
    assert set(snapshot["peers"]) == {"a"}
    assert snapshot["groups"]["CN_EQUITY"]["coverage"] == 0.5
    x = snapshot["peers"]["a"]["x"] + [0.0] * 14
    assert s.vector(x, "a", snapshot)[-1] == 0.5


def test_duplicate_share_cannot_change_peer_breadth_or_dispersion():
    raw = history()
    before = s.PeerHistory(raw).snapshot("2025-01-10", "2025-01-13")
    raw["funds"].append(raw["funds"][1] | {"fund_code": "b_share"})
    after = s.PeerHistory(raw).snapshot("2025-01-10", "2025-01-13")
    assert before["groups"] == after["groups"]


@pytest.mark.parametrize("name", s.CANDIDATES)
def test_future_labels_and_features_do_not_change_fitted_model(name):
    rows = [
        r | {"z": [r["x"][30], r["x"][31], 0.1, 0.2, 0.3, 0.4, 0.5, 1], "return_target": 1.0 if r["y"] else -1.0}
        for r in sample_rows()
    ]
    first = s.fit(rows, name, "2023-06-10")
    for row in rows:
        if row["mature"] >= "2023-06-10":
            row["z"] = [float("nan")] * 8
            row["return_target"] = float("nan")
            row["y"] = 1 - row["y"]
    second = s.fit(rows, name, "2023-06-10")
    assert first["fit_hash"] == second["fit_hash"]
    assert s.answer([0.01, 1, 0, 0, 0, 0, 0, 1], name, first) == s.answer([0.01, 1, 0, 0, 0, 0, 0, 1], name, second)


def test_live_peer_with_wrong_dates_or_late_save_is_rejected(monkeypatch):
    fund = history()["funds"][0]
    value = {
        "code": "a",
        "family": "a",
        "group": "CN_EQUITY",
        "base": "2025-01-10",
        "u": "2025-01-13",
        "at": "2025-01-13T07:00:00+08:00",
        "models_hash": "model",
        "inputs": fund["rows"],
    }
    monkeypatch.setattr(b, "now", lambda: datetime(2025, 1, 13, 7, 15, tzinfo=b.ZONE))
    assert len(s.validate_peer(value, fund, "2025-01-10", "2025-01-13", "model")["x"]) == 18
    value["at"] = "2025-01-13T08:30:00+08:00"
    with pytest.raises(ValueError, match="BREADTH_PEER_NOT_VALID"):
        s.validate_peer(value, fund, "2025-01-10", "2025-01-13", "model")
    value["at"] = "2025-01-13T07:00:00+08:00"
    value["inputs"] = value["inputs"][:-1]
    with pytest.raises(ValueError, match="BREADTH_PEER_INPUT_DATES_INVALID"):
        s.validate_peer(value, fund, "2025-01-10", "2025-01-13", "model")


class ConstantModel:
    def predict_proba(self, x):
        return np.asarray([[0.7, 0.3]] * len(x))

    def predict(self, x):
        return np.full(len(x), -0.3)


@pytest.fixture
def ready(previous_ready, monkeypatch):  # noqa: F811
    directory, original, _ = previous_ready
    b.save(s.root() / "result.json", {"winner": s.CANDIDATES[0], "model_sha256": "abc"})
    bundle = {n: {"CN_EQUITY": {"model": ConstantModel()}} for n in s.CANDIDATES}
    monkeypatch.setattr(s, "models", lambda: ({"model_sha256": "abc"}, bundle))
    source = b.read(o.root() / "forward/2026-09-15/001000.json")
    profile = {"fund_code": "001000", "family": original["family"], "group": "CN_EQUITY"}
    peers = {"001000": {"group": "CN_EQUITY", "family": original["family"], "x": source["x"][:18]}}
    context = s.summarize(peers, [profile]) | {
        "at": b.now().isoformat(),
        "base": "2026-09-14",
        "u": "2026-09-15",
        "original_references": {"001000": b.digest(original)},
    }
    monkeypatch.setattr(s, "peer_snapshot", lambda *_: context)
    return directory, original, context


def test_new_version_does_not_backfill_old_target(ready, monkeypatch):
    monkeypatch.setattr(s, "peer_snapshot", lambda *_: pytest.fail("old target should not create context"))
    assert s.tick()["verified_forecasts"] == 0


def test_live_context_and_predictions_are_immutable_and_outcomes_paired(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    directory, original, _ = ready
    assert s.tick()["verified_forecasts"] == 1
    path = s.root() / "forward/2026-09-15/001000.json"
    saved = path.read_bytes()
    s.tick()
    assert path.read_bytes() == saved
    b.save(
        directory / "outcomes/2026-09-15/001000.json",
        {"y": 0, "actual_direction": "DOWN", "forecast_hash": b.digest(original)},
    )
    monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 19, tzinfo=b.ZONE))
    assert s.report()["matched_forward_metrics"][s.CANDIDATES[0]]["accuracy"] == 1


def test_changed_peer_reference_is_rejected(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    assert s.tick()["verified_forecasts"] == 1
    path = next((s.root() / "live/2026-09-15").glob("peers-*.json"))
    context = b.read(path)
    context["original_references"]["001000"] = "changed"
    b.save(path, context, replace=True)
    with pytest.raises(ValueError, match="ROUND_11_PEER_INPUT_CHANGED"):
        s.report()


def test_late_saved_answer_is_invalid_and_closed_window_cannot_load_model(ready, monkeypatch):
    monkeypatch.setattr(s, "FIRST_TARGET", "2026-09-15")
    save = b.save

    def crossing(path, value, **kwargs):
        save(path, value, **kwargs)
        if path.parent.parent == s.root() / "forward":
            monkeypatch.setattr(b, "now", lambda: datetime(2026, 9, 15, 8, 30, tzinfo=b.ZONE))

    monkeypatch.setattr(b, "save", crossing)
    result = s.tick()
    assert result["verified_forecasts"] == 0 and result["invalid_or_late"] == 1
    monkeypatch.setattr(s, "models", lambda: pytest.fail("post deadline model loading"))
    assert s.tick()["verified_forecasts"] == 0


def test_prior_runner_failure_does_not_skip_breadth_branch(monkeypatch):
    from scripts import direction_1d_sprint_breadth as entry

    monkeypatch.setattr(entry.existing, "run", lambda: (_ for _ in ()).throw(ValueError("OLD_FAILED")))
    calls = []
    monkeypatch.setattr(entry.breadth, "tick", lambda: calls.append(1))
    monkeypatch.setattr(b, "save", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="INDEPENDENT_BRANCH_FAILED"):
        entry.run()
    assert calls == [1]
