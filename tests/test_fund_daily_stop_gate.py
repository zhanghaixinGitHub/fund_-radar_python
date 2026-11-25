from copy import deepcopy
from datetime import datetime

import pytest
from app.services.fund_daily_commander import identity
from app.services.fund_daily_stop_gate import armed_ticket, evaluate_stop

AT = datetime.fromisoformat("2026-11-24T10:00:00+08:00")
THREAD = "daily-commander"


def inputs(tmp_path):
    state = {"owner_user_id": "owner", "scope": ["000001"], "supplement_run_id": "original-run",
             "checked_at": AT.isoformat(), "window": {"target_nav_date": "2026-11-24",
             "base_nav_date": "2026-11-23", "deadline_at": "2026-11-24T15:00:00+08:00"}}
    ticket = armed_ticket(tmp_path / "daily-prediction-20261124", state, THREAD, AT)
    # 这些用例固定验证历史四小时场景，不跟随用户后续修改的生产上限。
    ticket["limits"] = {"max_runtime_minutes": 240, "max_rounds": 5}
    event = {"hook_event_name": "Stop", "session_id": THREAD, "turn_id": "original-turn",
             "stop_hook_active": False, "last_assistant_message": "所有任务已经完成"}
    snapshot = {"status": "CONTINUE", "can_finish": False, "live_verified": False,
                "next_action": "CONTINUE", "counts": {"RETURN_TO_WORKER": 1},
                "funds": [{"fund_code": "000001", "action": "RETURN_TO_WORKER"}]}
    control = {"identity": identity(state), "status": "OPEN"}
    return state, ticket, event, snapshot, control


@pytest.mark.parametrize("already_continued", [False, True])
def test_unfinished_business_blocks_stop_repeatedly_despite_claimed_success(tmp_path, already_continued):
    state, ticket, event, snapshot, control = inputs(tmp_path)
    event["stop_hook_active"] = already_continued
    decision, record = evaluate_stop(event, ticket, AT, snapshot=snapshot, state=state, control=control)
    assert decision["decision"] == "block" and "000001" in decision["reason"]
    assert not record["accepted"]


def test_final_live_acceptance_and_formal_finish_are_both_required(tmp_path):
    state, ticket, event, snapshot, control = inputs(tmp_path)
    snapshot.update(status="SUCCEEDED", can_finish=True, live_verified=True)
    assert evaluate_stop(event, ticket, AT, snapshot=snapshot, state=state, control=control)[0]["decision"] == "block"
    control["status"] = "SUCCEEDED"
    decision, record = evaluate_stop(event, ticket, AT, snapshot=snapshot, state=state, control=control)
    assert decision == {} and record["status"] == "ACCEPTED"
    snapshot["live_verified"] = False
    assert evaluate_stop(event, ticket, AT, snapshot=snapshot, state=state, control=control)[0]["decision"] == "block"


@pytest.mark.parametrize("status", ["WAITING", "DONE_WITH_GAPS"])
def test_waiting_and_abstention_do_not_meet_full_acceptance_threshold(tmp_path, status):
    state, ticket, event, snapshot, control = inputs(tmp_path)
    snapshot.update(status=status, can_finish=True, live_verified=True)
    control["status"] = status
    assert evaluate_stop(event, ticket, AT, snapshot=snapshot, state=state, control=control)[0]["decision"] == "block"


def test_different_chat_unarmed_task_and_user_interrupt_are_unaffected(tmp_path):
    state, ticket, event, _, _ = inputs(tmp_path)
    for key, value in [("session_id", "another-chat"), ("hook_event_name", "SubagentStop"),
                       ("hook_event_name", "Interrupt")]:
        changed = {**event, key: value}
        assert evaluate_stop(changed, ticket, AT)[0] == {}
    assert evaluate_stop(event, {**ticket, "enabled": False}, AT)[0] == {}


def test_missing_initialization_and_changed_scope_cannot_pass(tmp_path):
    state, ticket, event, snapshot, control = inputs(tmp_path)
    assert evaluate_stop(event, ticket, AT, state={})[1]["status"] == "BLOCKED_NOT_STARTED"
    changed = deepcopy(state)
    changed["scope"] = []
    assert evaluate_stop(event, ticket, AT, state=changed)[1]["status"] == "BLOCKED_IDENTITY_CHANGED"


def test_expired_real_window_pauses_without_claiming_acceptance(tmp_path):
    state, ticket, event, _, _ = inputs(tmp_path)
    ticket["armed_at"] = AT.replace(hour=14).isoformat()
    decision, record = evaluate_stop(event, ticket, AT.replace(hour=16), state=state)
    assert decision["continue"] is False
    assert record == {"status": "PAUSED_WINDOW_EXPIRED", "accepted": False}
    _, record = evaluate_stop(event, ticket, AT.replace(day=25), state=state)
    assert record["status"] == "PAUSED_DATE_CHANGED" and not record["accepted"]


@pytest.mark.parametrize("elapsed,rounds,expected", [(240, 0, "PAUSED_TIME_LIMIT"), (20, 5, "PAUSED_ROUND_LIMIT")])
def test_limits_pause_without_claiming_success_even_if_assistant_says_done(tmp_path, elapsed, rounds, expected):
    from datetime import timedelta

    state, ticket, event, snapshot, control = inputs(tmp_path)
    ticket["completed_rounds"] = rounds
    decision, record = evaluate_stop(event, ticket, AT + timedelta(minutes=elapsed), state=state)
    assert decision["continue"] is False and record["status"] == expected
    assert not record["accepted"] and record["budget"]["limit_reached"]


def test_fifth_round_can_finish_if_formal_acceptance_has_passed(tmp_path):
    state, ticket, event, snapshot, control = inputs(tmp_path)
    ticket["completed_rounds"] = 5
    snapshot.update(status="SUCCEEDED", can_finish=True, live_verified=True)
    control["status"] = "SUCCEEDED"
    assert evaluate_stop(event, ticket, AT, snapshot=snapshot, state=state, control=control)[1]["accepted"]


def test_late_stop_uses_actual_finish_time_instead_of_claimed_success(tmp_path):
    from datetime import timedelta

    state, ticket, event, snapshot, control = inputs(tmp_path)
    snapshot.update(status="SUCCEEDED", can_finish=True, live_verified=True)
    control.update(status="SUCCEEDED", events=[{"action": "FINISH", "at": (AT + timedelta(minutes=239)).isoformat()}])
    later = AT + timedelta(minutes=240)
    assert evaluate_stop(event, ticket, later, snapshot=snapshot, state=state, control=control)[1]["accepted"]
    control["events"][0]["at"] = (later + timedelta(minutes=1)).isoformat()
    record = evaluate_stop(event, ticket, later, snapshot=snapshot, state=state, control=control)[1]
    assert not record["accepted"] and record["status"] == "PAUSED_TIME_LIMIT"


@pytest.mark.parametrize("armed_hour, expected", [(10, "PAUSED_TIME_LIMIT"), (14, "PAUSED_WINDOW_EXPIRED")])
def test_late_callback_preserves_the_earlier_time_boundary(tmp_path, armed_hour, expected):
    state, ticket, event, _, _ = inputs(tmp_path)
    ticket["armed_at"] = AT.replace(hour=armed_hour).isoformat()
    decision, record = evaluate_stop(event, ticket, AT.replace(hour=19), state=state)
    assert decision["continue"] is False and record["status"] == expected and not record["accepted"]


def hook_command(tmp_path, monkeypatch, *, session=THREAD):
    import io
    import json
    import sys

    from app.commands import fund_daily_gate as command
    from app.services.fund_materials_store import versioned_save

    _, ticket, event, _, _ = inputs(tmp_path)
    versioned_save(tmp_path / "armed.json", ticket)
    event.update(session_id=session, last_assistant_message="不应保存的对话正文")
    monkeypatch.setattr(command, "GATE_ROOT", tmp_path)
    monkeypatch.setattr(command, "now", lambda: AT.replace(hour=14))
    monkeypatch.setattr(sys, "argv", ["gate", "--action", "hook", "--thread-id", THREAD])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event)))
    monkeypatch.setattr("app.services.prediction_trial_v2.storage_check", lambda _: None)
    return command


def test_real_entry_records_safe_identity_and_disables_budget_paused_ticket(tmp_path, monkeypatch, capsys):
    import json

    from app.services.fund_exposure_common import read

    command = hook_command(tmp_path, monkeypatch)
    command.main()
    assert json.loads(capsys.readouterr().out)["continue"] is False
    entry = read(tmp_path / "last-hook-invocation.json")
    assert entry["applicable"] and entry["session_id"] == THREAD and entry["turn_id"] == "original-turn"
    assert "不应保存的对话正文" not in json.dumps(entry, ensure_ascii=False)
    ticket = read(tmp_path / "armed.json")
    assert not ticket["enabled"] and ticket["pause_status"] == "PAUSED_TIME_LIMIT"
    assert not read(tmp_path / "last-stop-check.json")["accepted"]


def test_unrelated_real_entry_is_visible_but_never_runs_business_acceptance(tmp_path, monkeypatch, capsys):
    import json

    from app.services.fund_exposure_common import read

    command = hook_command(tmp_path, monkeypatch, session="another-chat")
    monkeypatch.setattr("app.services.fund_daily_stop_gate.evaluate_stop",
                        lambda *a, **kw: pytest.fail("不能验收另一条聊天"))
    command.main()
    assert json.loads(capsys.readouterr().out) == {}
    assert not read(tmp_path / "last-hook-invocation.json")["applicable"]
    assert not (tmp_path / "last-stop-check.json").exists()
    assert read(tmp_path / "armed.json")["enabled"]


@pytest.mark.parametrize("session", [THREAD, "another-chat"])
def test_entry_storage_failure_never_writes_receipts_or_blocks_unrelated_chat(tmp_path, monkeypatch, capsys, session):
    import json

    command = hook_command(tmp_path, monkeypatch, session=session)

    def fail(_):
        raise OSError("storage unavailable")

    monkeypatch.setattr("app.services.prediction_trial_v2.storage_check", fail)
    command.main()
    response = json.loads(capsys.readouterr().out)
    assert response.get("continue") is False if session == THREAD else response == {}
    assert not (tmp_path / "last-hook-invocation.json").exists()
    assert not (tmp_path / "last-stop-check.json").exists()
