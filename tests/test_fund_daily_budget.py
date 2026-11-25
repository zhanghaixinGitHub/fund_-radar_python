from datetime import timedelta

import pytest
from app.services.fund_daily_budget import DailyBudgetReached, budget_status, complete_round
from app.services.fund_daily_commander import Commander
from app.services.fund_daily_stop_gate import armed_ticket
from app.services.fund_materials_store import versioned_save
from tests.test_fund_daily_commander import AT, audit, state


def prepare(tmp_path):
    s = state()
    ticket = armed_ticket(tmp_path / "daily-prediction-20261124", s, "original-commander", AT)
    for code in s["scope"]:
        versioned_save(tmp_path / "supplement-funds" / (code + ".json"), {"items": [{
            "fund_code": code, "field": "NAV", "required_date": "2026-11-23", "status": "REVIEW_REQUIRED",
            "attempted_at": AT.isoformat(), "summary": "实际核实但仍缺合格原件，已记录下一来源"}]})
    return s, ticket


def test_configured_runtime_boundary_and_five_round_boundary_are_independent(tmp_path):
    s, ticket = prepare(tmp_path)
    duration = timedelta(minutes=ticket["limits"]["max_runtime_minutes"])
    assert not budget_status(ticket, AT + duration - timedelta(seconds=1))["limit_reached"]
    assert budget_status(ticket, AT + duration)["pause_reason"] == "TIME_LIMIT"
    ticket["completed_rounds"] = 5
    assert budget_status(ticket, AT + timedelta(minutes=1))["pause_reason"] == "ROUND_LIMIT"


def test_five_real_full_scope_rounds_pause_and_duplicate_record_does_not_increment(tmp_path):
    s, ticket = prepare(tmp_path)
    for number in range(1, 6):
        checked = AT + timedelta(minutes=number)
        c = Commander(tmp_path, s, checked, audit=audit(s))
        for code in s["scope"]:
            c.change("accept", code=code)
        ticket, outcome = complete_round(ticket, c, checked)
        assert outcome == "RECORDED" and ticket["completed_rounds"] == number
        if number < 5:
            same, outcome = complete_round(ticket, c, checked)
            assert outcome == "ALREADY_RECORDED" and same["completed_rounds"] == number
    assert len(ticket["round_history"]) == 5
    assert all(row["scope"] == s["scope"] for row in ticket["round_history"])
    with pytest.raises(DailyBudgetReached, match="ROUND_LIMIT"):
        complete_round(ticket, c, checked)


def test_one_fund_and_unattempted_rows_cannot_be_claimed_as_full_round(tmp_path):
    s, ticket = prepare(tmp_path)
    checked = AT + timedelta(minutes=1)
    c = Commander(tmp_path, s, checked, audit=audit(s))
    c.change("accept", code=s["scope"][0])
    with pytest.raises(ValueError, match="NOT_REVIEWED"):
        complete_round(ticket, c, checked)
    for code in s["scope"]:
        versioned_save(tmp_path / "supplement-funds" / (code + ".json"), {"items": [{"status": "NOT_ATTEMPTED"}]})
        c.change("accept", code=code)
    with pytest.raises(ValueError, match="NOT_REVIEWED"):
        complete_round(ticket, c, checked)


def test_budget_reached_refuses_new_business_before_backend_is_called(tmp_path, monkeypatch):
    from app.commands import fund_daily_fund
    from app.services import fund_daily_budget

    s, ticket = prepare(tmp_path)
    ticket["run_dir"] = str(tmp_path)
    path = tmp_path / "budget-ticket.json"
    versioned_save(path, ticket)
    monkeypatch.setattr(fund_daily_budget, "TICKET_PATH", path)
    monkeypatch.setattr(fund_daily_fund, "now",
                        lambda: AT + timedelta(minutes=ticket["limits"]["max_runtime_minutes"]))

    class Backend:
        def context(self):
            pytest.fail("达到时间上限后不能开始新的业务或调用后端")

    with pytest.raises(DailyBudgetReached, match="TIME_LIMIT"):
        fund_daily_fund.advance_fund(tmp_path, s, s["scope"][0], Backend())


def test_same_day_rearming_preserves_start_and_rounds_after_restart(tmp_path, monkeypatch, capsys):
    from app.commands import fund_daily_gate
    from app.services import prediction_trial_v2
    from app.services.fund_exposure_common import read

    root = tmp_path / ".local-runs"
    root.mkdir()
    directory = root / "daily-prediction-20261124"
    s = state()
    versioned_save(directory / "run.json", s)
    ticket = armed_ticket(directory, s, "original-commander", AT)
    ticket.update(completed_rounds=3, round_started_at=(AT + timedelta(minutes=3)).isoformat(), enabled=False)
    gate_root = root / "daily-recovery"
    versioned_save(gate_root / "armed.json", ticket)
    monkeypatch.setattr(fund_daily_gate, "PROJECT", tmp_path)
    monkeypatch.setattr(fund_daily_gate, "GATE_ROOT", gate_root)
    monkeypatch.setattr(fund_daily_gate, "now", lambda: AT + timedelta(minutes=30))
    monkeypatch.setattr(prediction_trial_v2, "storage_check", lambda since: {})
    monkeypatch.setattr("sys.argv", ["gate", "--action", "arm", "--thread-id", "original-commander"])
    fund_daily_gate.main()
    saved = read(gate_root / "armed.json")
    assert saved["armed_at"] == ticket["armed_at"] and saved["completed_rounds"] == 3
    assert saved["round_started_at"] == ticket["round_started_at"] and not saved["enabled"]
