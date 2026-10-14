"""付费传输契约、响应复用及失败不重发；只使用内存数据库替身和MockTransport。"""

import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from app.core.deepseek_policy import PrivateDeepSeekSettings, prediction_scope
from app.integrations import fund_information_analysis as provider
from app.services.deepseek_usage import UsageLedger


class CachedResponseStore:
    """替身只维护已保存响应，测试重点是HTTP次数和实际费用记录。"""

    def __init__(self):
        self.row = None

    @contextmanager
    def begin(self):
        yield self

    def execute(self, statement, parameters):
        sql = str(statement)
        if sql.startswith("SELECT *,lease_until"):
            return SimpleNamespace(mappings=lambda: SimpleNamespace(first=lambda: self.row))
        if "SET state='READY'" in sql:
            self.row = {
                "state": "READY",
                "payload": json.loads(parameters["value"]),
                "content_hash": parameters["hash"],
                "active": False,
            }
            return SimpleNamespace(first=lambda: ("saved",))
        if "SET state='FAILED'" in sql:
            self.row = {"state": "FAILED", "failure_code": parameters["code"], "active": False}
        return SimpleNamespace()


def setup_transport(monkeypatch, tmp_path, handler):
    database = CachedResponseStore()
    settings = PrivateDeepSeekSettings(_env_file=None, api_key="private-test-only")
    monkeypatch.setattr(provider, "get_engine", lambda: database)
    monkeypatch.setattr(provider, "provider_settings", lambda *_: settings)
    ledger = UsageLedger(tmp_path / "usage.sqlite3")
    monkeypatch.setattr(provider, "UsageLedger", lambda: ledger)
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    return database, ledger


def test_live_request_uses_flash_once_and_reuses_saved_response(monkeypatch, tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        body = json.loads(request.content)
        assert str(request.url) == "https://api.deepseek.com/chat/completions"
        assert body["model"] == "deepseek-flash"
        assert body["thinking"] == {"type": "disabled"}
        assert body["max_tokens"] == 800
        assert body["response_format"] == {"type": "json_object"}
        assert "tools" not in body
        return httpx.Response(
            200,
            json={
                "id": "test-response",
                "choices": [{"finish_reason": "stop", "message": {"content": '{"valid":true,"issues":[]}'}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 20, "prompt_cache_hit_tokens": 100},
            },
        )

    setup_transport(monkeypatch, tmp_path, handler)
    budget = provider.Budget(datetime.now(UTC) + timedelta(minutes=5))
    value = {"fund_code": "002112", "target_nav_date": "2026-10-13"}
    with prediction_scope("002112", 1, "2026-10-13"):
        first = provider.request("audit", value, "AUDIT", budget)
        second = provider.request("audit", value, "AUDIT", budget)
    assert first == second == {"valid": True, "issues": []}
    assert len(calls) == budget.calls == 1
    assert budget.usage[0]["prompt_tokens"] == 200
    assert budget.usage[0]["estimated_peak_cny"] == pytest.approx(0.000364)


def test_unknown_network_outcome_is_never_automatically_retried(monkeypatch, tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("unknown provider outcome", request=request)

    setup_transport(monkeypatch, tmp_path, handler)
    budget = provider.Budget(datetime.now(UTC) + timedelta(minutes=5))
    value = {"fund_code": "002112", "target_nav_date": "2026-10-13"}
    with prediction_scope("002112", 1, "2026-10-13"):
        with pytest.raises(ValueError, match="PROVIDER_FAILED"):
            provider.request("synthesis", value, "SYNTHESIS", budget)
        with pytest.raises(ValueError, match="STAGE_FAILED"):
            provider.request("synthesis", value, "SYNTHESIS", budget)
    assert calls == [1]
    assert budget.calls == 1
