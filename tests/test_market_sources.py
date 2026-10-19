"""市场采集的身份、缺失、历史及事项复核边界；全程使用本地替身。"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from app.services import fund_market_news as news
from app.services import market_company_materials as companies
from app.services.fund_exposure_common import read
from app.services.fund_market_review import pending_events, record_event_review
from pydantic import SecretStr

CODE, ARTICLE, SHA = "013330", "AN" + "1" * 18, "a" * 64


class NewsClient:
    fails = False

    def news_catalog(self, code, lower):
        assert code == CODE
        if self.fails:
            raise ValueError("upstream failed")
        return [{"id": ARTICLE, "title": "基金经理变更公告", "published": "2026-10-15"}]

    def news_body(self, code, entry, old):
        return {
            "source": {"sha256": SHA},
            "body": "公告原文测试" * 100,
            "received_at": "2026-10-16T09:00:00+08:00",
            "candidate_material_event": True,
        }


def test_pending_event_survives_repeated_checks_and_source_failure(tmp_path, monkeypatch):
    instant = datetime.fromisoformat("2026-10-16T10:00:00+08:00")
    monkeypatch.setattr(news, "now", lambda: instant)
    client = NewsClient()
    first = news.collect(CODE, client=client, root=tmp_path)
    instant += timedelta(hours=1)
    second = news.collect(CODE, client=client, root=tmp_path)
    assert first["candidate_events"] == second["candidate_events"]
    assert len(second["candidate_events"]) == 1
    client.fails = True
    instant += timedelta(hours=1)
    failed = news.collect(CODE, client=client, root=tmp_path)
    assert failed["status"] == "FAILED" and not failed["catalog_complete"]
    assert failed["items"] == first["items"] and failed["candidate_events"] == first["candidate_events"]
    assert read(tmp_path / "funds" / (CODE + ".json"))["errors"]


def test_only_review_of_same_source_version_can_clear_event(tmp_path, monkeypatch):
    monkeypatch.setattr(news, "now", lambda: datetime.fromisoformat("2026-10-16T10:00:00+08:00"))
    events = news.collect(CODE, client=NewsClient(), root=tmp_path)["candidate_events"]
    root = tmp_path / "reviews"
    with pytest.raises(ValueError, match="MISMATCH"):
        record_event_review(
            CODE, ARTICLE, "b" * 64, "NO_RATING_IMPACT", "测试复核依据。" * 20, root=root, news_root=tmp_path
        )
    record_event_review(
        CODE, ARTICLE, SHA, "REQUIRES_RATING_REVIEW", "测试复核依据。" * 20, root=root, news_root=tmp_path
    )
    assert pending_events(events, root=root) == events
    record_event_review(CODE, ARTICLE, SHA, "NO_RATING_IMPACT", "测试复核依据。" * 20, root=root, news_root=tmp_path)
    assert pending_events(events, root=root) == []
    assert pending_events([{**events[0], "source_hash": "c" * 64}], root=root)


def test_company_collection_deduplicates_quotes_without_expanding_financial_permissions(tmp_path, monkeypatch):
    instant = datetime.fromisoformat("2026-10-16T18:00:00+08:00")
    monkeypatch.setattr(companies, "now", lambda: instant)
    monkeypatch.setattr(companies, "calendar", lambda: ((instant.date(),), "verified-test-calendar"))
    monkeypatch.setattr(
        companies,
        "permission",
        lambda: {"authorized_api_names": ["daily"], "authorization_verified_at": instant, "retention_days": 365},
    )
    monkeypatch.setattr(
        companies,
        "get_settings",
        lambda: SimpleNamespace(tushare_token=SecretStr("test-only"), tushare_api_url="https://test.invalid"),
    )
    calls = []

    class Client:
        def post(self, url, *, json):
            calls.append(json["api_name"])
            assert json["api_name"] == "daily"
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "code": 0,
                    "data": {
                        "fields": ["ts_code", "trade_date", "close", "pct_chg", "amount"],
                        "items": [["600000.SH", "20261016", 10, 1, 20], ["000001.SZ", "20261016", 11, -1, 30]],
                    },
                },
            )

    collector = companies.MarketCompanyCollector(tmp_path)
    collector.client = Client()
    first = collector.company("600000.SH")
    assert first["quote"]["date"] == "2026-10-16" and len(first["gaps"]) == 3
    assert first["financials"] == {}  # 权限不足保留缺口，不以零值冒充公司数据。
    assert collector.company("000001.SZ")["quote"]["close"] == 11
    assert collector.company("600000.SH") == first and calls == ["daily"]
    assert collector.company("8035.JP")["quote"] is None and calls == ["daily"]
