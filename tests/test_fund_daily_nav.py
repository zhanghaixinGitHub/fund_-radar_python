import json
from datetime import date
from types import SimpleNamespace

import httpx
import pytest
from app.services import fund_daily_nav as nav
from app.services.fund_exposure_common import read


def test_present_baseline_does_not_query_or_resync(monkeypatch, tmp_path):
    monkeypatch.setattr(nav, "current_nav_identity", lambda *args: "present")
    result = nav.refresh_missing_nav("008764", "2026-11-16", tmp_path,
        service_factory=lambda: pytest.fail("已有真实净值不重复抓取"))
    assert result["status"] == "BASE_NAV_PRESENT"


@pytest.mark.parametrize("arrived,api_code,expected", [
    (False, 0, "WAITING_NAV"), (True, 0, "AVAILABLE"), (False, 40203, "SOURCE_READ_FAILED")])
def test_missing_baseline_uses_exact_day_normal_ingestion_and_archives_actual_response(
    monkeypatch, tmp_path, arrived, api_code, expected
):
    empty = nav.nav_identity([], "2026-11-16")
    identities = iter([empty, "real-arrived" if arrived else empty])
    monkeypatch.setattr(nav, "current_nav_identity", lambda *args: next(identities))
    monkeypatch.setattr(nav, "get_settings", lambda: SimpleNamespace(
        tushare_token=SimpleNamespace(get_secret_value=lambda: "not-a-real-token")))
    class Service:
        def __init__(self):
            self._provider = SimpleNamespace(_client=SimpleNamespace(event_hooks={"response": []}))
        def sync_market_nav_history(self, codes, *, start_date, end_date):
            assert codes == ("008764.OF",) and start_date == end_date == date(2026, 11, 16)
            request = httpx.Request("POST", "https://source.example", json={
                "api_name": "fund_nav", "params": {
                    "ts_code": codes[0], "start_date": "20261116", "end_date": "20261116"}})
            response = httpx.Response(200, request=request, json={"code": api_code, "data": {"items": []}})
            for hook in self._provider._client.event_hooks["response"]:
                hook(response)
            return SimpleNamespace(to_payload=lambda: {"received": 1 if arrived else 0})
        def close(self):
            pass
    result = nav.refresh_missing_nav("008764", "2026-11-16", tmp_path, service_factory=Service)
    assert result["status"] == expected
    assert result["receipts"][0]["api"] == "fund_nav"
    assert read(tmp_path / "receipt.json")["sync"]["received"] == int(arrived)
    assert "not-a-real-token" not in json.dumps(result)
