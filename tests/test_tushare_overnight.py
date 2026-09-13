"""官方接口边界测试：限定SPX、禁重定向/重试、响应不允许泄漏凭据。"""

from datetime import date
from types import SimpleNamespace

import httpx
import pytest
from app.integrations import tushare_overnight as client
from pydantic import SecretStr


@pytest.mark.parametrize("case", ["success", "denied", "redirect", "token_echo", "oversized"])
def test_bounded_official_request(monkeypatch, case):
    settings = SimpleNamespace(
        tushare_token=SecretStr("synthetic-token-only"), tushare_api_url="https://api.tushare.pro"
    )
    monkeypatch.setattr(client, "get_settings", lambda: settings)
    calls = []

    def respond(request):
        calls.append(request)
        if case == "redirect":
            return httpx.Response(302, headers={"Location": "https://example.invalid"})
        if case == "denied":
            return httpx.Response(200, json={"code": 40203})
        if case == "token_echo":
            return httpx.Response(200, json={"code": 0, "msg": "synthetic-token-only"})
        if case == "oversized":
            return httpx.Response(200, content=b" " * 65537)
        return httpx.Response(200, json={"code": 0, "data": {"fields": client.FIELDS, "items": []}})

    original = httpx.Client
    monkeypatch.setattr(client.httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(respond), **kw))
    if case == "success":
        assert b'"code":0' in client.fetch_spx(date(2026, 9, 10), date(2026, 9, 11))
    else:
        with pytest.raises(ValueError, match="OVERNIGHT_"):
            client.fetch_spx(date(2026, 9, 10), date(2026, 9, 11))
    assert len(calls) == 1
    assert calls[0].url == "https://api.tushare.pro"


def test_unbounded_history_request_never_connects(monkeypatch):
    monkeypatch.setattr(client, "get_settings", lambda: pytest.fail("越界请求不能取凭据"))
    with pytest.raises(ValueError, match="QUERY_RANGE_INVALID"):
        client.fetch_spx(date(2021, 1, 1), date(2024, 12, 31))
