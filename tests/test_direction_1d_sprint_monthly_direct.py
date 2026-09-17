"""验证更新频率实验的月初隔离、旧检查点复用和同模型绑定边界。"""

import pytest
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_monthly_direct as s

from test_direction_1d_sprint_hk import rows as historical_rows


def rows():
    return [r | {"z": r["z"] + [0.2, 0, 0, 0, 1, -0.3, 0, 1]} for r in historical_rows()]


def test_months_cover_next_calendar_month_without_changing_horizon():
    assert s.month_window(1) == ("2025-01-01", "2025-02-01")
    assert s.month_window(12) == ("2025-12-01", "2026-01-01")
    with pytest.raises(ValueError):
        s.month_window(13)


def test_future_and_immature_inputs_do_not_change_training_selection():
    data = rows()
    before = s.selected(data, "2023-06-10")
    for row in data:
        if row["mature"] >= "2023-06-10":
            row["z"], row["y"] = [float("nan")] * 20, 1 - row["y"]
    assert b.digest(before) == b.digest(s.selected(data, "2023-06-10"))
    assert all(r["u"] < "2023-06-10" and r["mature"] < "2023-06-10" for r in before)


def test_quarter_start_reuses_exact_prior_head_without_fitting(monkeypatch):
    data = rows()
    old = {"us_etf": {"fit_hash": b.digest(s.selected(data, "2025-01-01"))}}
    monkeypatch.setattr(s, "quarter_head", lambda *args: (old, {"sha256": "old"}))
    monkeypatch.setattr(s.original, "fit", lambda *args: pytest.fail("quarter already trained"))
    trained, receipt = s.checkpoint(data, 1, "CN_EQUITY")
    assert trained is old and receipt["reused"] is True
    old["us_etf"]["fit_hash"] = "changed"
    with pytest.raises(ValueError, match="REUSED_TRAINING_CHANGED"):
        s.checkpoint(data, 1, "CN_EQUITY")


def test_nonquarter_month_trains_once_and_preserves_quarter_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "root", lambda: tmp_path)
    data = rows()
    old = {"control": {"frozen_control": "quarter"}, "us_etf": {"old": True}, "group": "CN_EQUITY"}
    monkeypatch.setattr(s, "quarter_head", lambda *args: (old, {"sha256": "old"}))
    calls = []

    def fit(selected_rows, name, cutoff):
        calls.append((name, cutoff))
        chosen = s.selected(selected_rows, cutoff)
        return {
            "fit_hash": b.digest(chosen),
            "max_mature_date": max(r["mature"] for r in chosen),
            "fit_end": max(r["u"] for r in chosen),
            "cutoff": cutoff,
        }

    monkeypatch.setattr(s.original, "fit", fit)
    first, receipt = s.checkpoint(data, 2, "CN_EQUITY")
    again, second_receipt = s.checkpoint(data, 2, "CN_EQUITY")
    assert calls == [(s.original.LEARNED[0], "2025-02-01")]
    assert first == again and receipt == second_receipt and receipt["reused"] is False
    assert first["control"] == old["control"]


def test_current_alias_rejects_different_training_rows(monkeypatch):
    data = rows()
    group = data[0]["group"]
    monkeypatch.setattr(
        s.original,
        "models",
        lambda: ({"model_sha256": "model"}, {s.original.LEARNED[0]: {group: {"us_etf": {"fit_hash": "wrong"}}}}),
    )
    monkeypatch.setattr(s.original, "plan", lambda: {"current_fit_cutoff": "2025-01-01"})
    with pytest.raises(ValueError, match="ALIAS_ROWS_CHANGED"):
        s.current_alias(data)
