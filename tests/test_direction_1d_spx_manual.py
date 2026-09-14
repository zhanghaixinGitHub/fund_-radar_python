"""验证手动时间证据、防连点、真实进度、安全错误及鉴权，不把合成行情计为真实模型效果。"""

import json
from datetime import datetime, timedelta

import httpx
import pytest
from app.api import dependencies
from app.api.routes import spx_manual as api
from app.services import direction_1d_overnight_collect as original
from app.services import direction_1d_spx_manual as manual
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

SOURCE = {
    "source_id": "fixture",
    "source_code": "TUSHARE_PRO_FUND",
    "enabled": True,
    "authorization_verified_at": "2026-09-13T09:00:00+08:00",
    "retention_days": 365,
    "rate_limit_per_minute": 200,
}


@pytest.fixture
def env(tmp_path):
    proof, contract, root = tmp_path / "proof.json", tmp_path / "contract", tmp_path / "manual"
    original.write_new(
        proof,
        {
            "status": "PERMISSION_AVAILABLE_NOT_USAGE_OR_TIMING_PROOF",
            "http_status": 200,
            "provider_code": 0,
            "row_count": 1,
            "reservation": {"api_name": "index_global", "params": {"ts_code": "SPX"}},
        },
    )
    original.initialize(
        contract, proof, clock=lambda: datetime.fromisoformat("2026-09-13T09:00:00+08:00"), source_reader=lambda: SOURCE
    )
    stamp = [datetime.fromisoformat("2026-09-14T07:40:00+08:00")]
    calls = []

    def query(start, end):
        calls.append((start, end))
        dates = manual.current_plan(stamp[0])["required_us_dates"]
        rows = [["SPX", day.replace("-", ""), 100 + i, 99 + i, 100 / (99 + i)] for i, day in enumerate(dates)]
        return json.dumps({"code": 0, "data": {"fields": original.FIELDS, "items": rows}}).encode()

    kwargs = dict(root=root, contract_root=contract, clock=lambda: stamp[0], query=query, source_reader=lambda: SOURCE)
    return root, stamp, calls, kwargs


def view(kwargs):
    return manual.status(**{key: kwargs[key] for key in ("root", "contract_root", "clock")})


@pytest.mark.parametrize(
    ("stamp", "expected"),
    [
        ("2026-09-14T07:40:00+08:00", "ON_TIME"),
        ("2026-09-14T08:00:00+08:00", "LATE"),
        ("2026-09-14T15:00:00+08:00", "LATE"),
        ("2026-09-13T15:00:00+08:00", "REFERENCE_ONLY"),
    ],
)
def test_manual_sync_records_actual_timing_and_survives_status_reload(env, stamp, expected):
    root, clock, calls, kwargs = env
    clock[0] = datetime.fromisoformat(stamp)
    assert view(kwargs)["lastAttempt"] is None
    assert calls == []  # 页面只读状态不会调用外部接口。
    result = manual.synchronize(**kwargs)
    assert result["lastAttempt"]["state"] == expected
    assert result["lastAttempt"]["usableBeforeU08"] == (expected == "ON_TIME")
    assert result["lastAttempt"]["persistedAt"] == stamp
    assert len(calls) == 1
    saved = list(root.rglob("response.json"))[0].read_bytes()
    assert view(kwargs)["lastAttempt"] == result["lastAttempt"]
    assert manual.synchronize(**kwargs)["canSync"] is False
    assert len(calls) == 1
    assert list(root.rglob("response.json"))[0].read_bytes() == saved


def test_more_than_four_attempts_allowed_but_each_failure_keeps_cooldown(env):
    _, clock, calls, kwargs = env
    kwargs["source_reader"] = lambda: SOURCE | {"enabled": False}
    for number in range(6):
        result = manual.synchronize(**kwargs)
        assert result["lastAttempt"]["state"] == "FAILED"
        assert result["lastAttempt"]["errorCode"] == "OVERNIGHT_SOURCE_UNAVAILABLE"
        assert result["availability"] == "COOLDOWN"
        assert result["attemptsToday"] == number + 1
        assert manual.synchronize(**kwargs)["performedNow"] is False
        clock[0] += timedelta(minutes=2)
    assert view(kwargs)["attemptsToday"] == 6
    assert view(kwargs)["maxAttemptsPerDay"] is None
    assert view(kwargs)["canSync"] is True
    assert calls == []


def test_success_is_separate_from_cooldown_and_auto_read_becomes_ready(env):
    _, clock, calls, kwargs = env
    result = manual.synchronize(**kwargs)
    assert result["availability"] == "COOLDOWN"
    assert "正在执行" not in result["message"]
    assert "同步成功" in result["lastAttempt"]["message"]
    assert result["lastAttempt"]["completedSteps"] == 3
    clock[0] += timedelta(seconds=60)
    assert view(kwargs)["availability"] == "READY"
    assert view(kwargs)["nextAllowedAt"] is None
    assert len(calls) == 1


def test_poll_reads_real_stages_without_issuing_another_request(env, monkeypatch):
    _, _, calls, kwargs = env
    observed = []
    query = kwargs["query"]
    write = original.write_body

    def inspect(expected):
        current = view(kwargs)
        assert current["availability"] == "RUNNING"
        assert current["lastAttempt"]["stage"] == expected
        observed.append(current["lastAttempt"]["completedSteps"])
        assert manual.synchronize(**kwargs)["performedNow"] is False

    def source_reader():
        inspect("SOURCE")
        return SOURCE

    def fetch(start, end):
        inspect("FETCH")
        return query(start, end)

    def save(path, body):
        inspect("SAVE")
        write(path, body)

    kwargs.update(source_reader=source_reader, query=fetch)
    monkeypatch.setattr(original, "write_body", save)
    result = manual.synchronize(**kwargs)
    assert observed == [0, 1, 2]
    assert len(calls) == 1
    assert result["lastAttempt"]["stage"] == "DONE"


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (httpx.ReadTimeout("TEST_SECRET"), "SPX_TIMEOUT"),
        (httpx.ConnectError("TEST_SECRET"), "SPX_NETWORK_ERROR"),
        (ValueError("OVERNIGHT_PROVIDER_BUSINESS_FAILED"), "OVERNIGHT_PROVIDER_BUSINESS_FAILED"),
        (ValueError("OVERNIGHT_PRICE_INVALID"), "SPX_INVALID_RESPONSE"),
        (RuntimeError("TEST_SECRET"), "SPX_UNKNOWN_ERROR"),
    ],
)
def test_failure_has_safe_actionable_reason(env, failure, expected):
    root, _, _, kwargs = env

    def failed(*_):
        raise failure

    kwargs["query"] = failed
    value = manual.synchronize(**kwargs)
    assert value["lastAttempt"]["errorCode"] == expected
    assert value["lastAttempt"]["message"] == manual.ERROR_MESSAGES[expected]
    assert value["lastAttempt"]["stage"] == "FETCH"
    assert "TEST_SECRET" not in json.dumps(value)
    assert all("TEST_SECRET" not in path.read_text() for path in root.rglob("*.json"))


def test_parallel_click_does_not_issue_another_request(env):
    root, _, calls, kwargs = env
    with manual.operation_lock(root) as acquired:
        assert acquired
        result = manual.synchronize(**kwargs)
        assert result["canSync"] is False
    assert calls == []
    assert view(kwargs)["canSync"] is True


def test_missing_dates_and_changed_evidence_cannot_become_usable(env):
    root, _, _, kwargs = env
    real_query = kwargs["query"]

    def incomplete(start, end):
        body = json.loads(real_query(start, end))
        body["data"]["items"].pop()
        return json.dumps(body).encode()

    kwargs["query"] = incomplete
    result = manual.synchronize(**kwargs)
    assert result["lastAttempt"]["state"] == "INCOMPLETE"
    assert result["lastAttempt"]["usableBeforeU08"] is False
    list(root.rglob("response.json"))[0].write_bytes(b"{}")
    assert view(kwargs)["canSync"] is False


def test_request_received_before_eight_but_saved_after_eight_is_late(env, monkeypatch):
    _, clock, _, kwargs = env
    clock[0] = datetime.fromisoformat("2026-09-14T07:59:59+08:00")
    write = original.write_body

    def late_disk(path, body):
        write(path, body)
        clock[0] += timedelta(seconds=1)

    monkeypatch.setattr(original, "write_body", late_disk)
    result = manual.synchronize(**kwargs)
    assert result["lastAttempt"]["state"] == "LATE"
    assert result["lastAttempt"]["usableBeforeU08"] is False


def test_crash_before_result_can_be_retried_without_overwriting_reservation(env, monkeypatch):
    root, clock, calls, kwargs = env
    save = manual.save_new

    def interrupted(path, payload):
        if path.name == "result.json":
            raise OSError("simulated process interruption")
        return save(path, payload)

    monkeypatch.setattr(manual, "save_new", interrupted)
    with pytest.raises(OSError):
        manual.synchronize(**kwargs)
    clock[0] += timedelta(minutes=3)
    assert view(kwargs)["lastAttempt"]["state"] == "INTERRUPTED"
    monkeypatch.setattr(manual, "save_new", save)
    manual.synchronize(**kwargs)
    assert len(calls) == 2
    assert len(list(root.rglob("reserved.json"))) == 2


def test_unknown_calendar_makes_no_request(env):
    _, clock, calls, kwargs = env
    clock[0] = datetime.fromisoformat("2027-01-04T07:30:00+08:00")
    assert manual.synchronize(**kwargs)["canSync"] is False
    assert calls == []


def test_external_errors_do_not_leak_credentials(env):
    root, _, _, kwargs = env

    def failed(*_):
        raise RuntimeError("https://invalid/?token=TEST_SECRET")

    kwargs["query"] = failed
    assert manual.synchronize(**kwargs)["lastAttempt"]["state"] == "FAILED"
    assert all("TEST_SECRET" not in path.read_text() for path in root.rglob("*.json"))


def test_internal_routes_require_token_and_reject_browser_origin(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(dependencies, "get_settings", lambda: SimpleNamespace(ai_service_token=SecretStr("test-only")))
    calls = []
    monkeypatch.setattr(api, "status", lambda: {"mode": "MANUAL"})
    monkeypatch.setattr(api, "synchronize", lambda: calls.append(1) or {"mode": "MANUAL"})
    app = FastAPI()
    app.include_router(api.router, prefix="/spx")
    with TestClient(app) as client:
        assert client.post("/spx/sync").status_code == 403
        assert (
            client.post("/spx/sync", headers={"X-Service-Token": "test-only", "Origin": "http://localhost"}).status_code
            == 403
        )
        assert client.get("/spx/status", headers={"X-Service-Token": "test-only"}).status_code == 200
        assert calls == []
        assert client.post("/spx/sync", headers={"X-Service-Token": "test-only"}).status_code == 200
        assert len(calls) == 1
