"""市场采集的身份、缺失、历史及事项复核边界；全程使用本地替身。"""

import json
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


def test_recent_news_catalog_can_finish_before_large_history_ends(tmp_path, monkeypatch):
    """多年总公告超过两千条不妨碍核验本次30日窗口，原响应仍须归档。"""
    client = news.MarketNewsClient(tmp_path)
    rows = [
        {"ID": "AN" + str(n).zfill(18), "FUNDCODE": CODE, "TITLE": "真实目录", "PUBLISHDATEDesc": day}
        for n, day in enumerate(["2026-10-15", "2026-09-01"], 1)
    ]
    monkeypatch.setattr(
        client,
        "_get",
        lambda u, p: json.dumps({"ErrCode": 0, "PageIndex": 1, "TotalCount": 5000, "Data": rows}).encode(),
    )
    try:
        result = client.news_catalog(CODE, "2026-09-28")
        assert len(result) == 1 and result[0]["published"] == "2026-10-15"
        assert (tmp_path / result[0]["catalog_source"]["file"]).exists()
        rows.reverse()
        with pytest.raises(ValueError, match="CATALOG_ORDER"):
            client.news_catalog(CODE, "2026-09-28")
    finally:
        client.close()


@pytest.mark.parametrize("failure", ["changed_total", "newer_second_page", "empty_second_page"])
def test_recent_news_catalog_rejects_incomplete_or_inconsistent_paging(tmp_path, monkeypatch, failure):
    client = news.MarketNewsClient(tmp_path)

    def response(url, params):
        page = params["pageIndex"]
        day = "2026-10-15" if page == 1 else "2026-10-16" if failure == "newer_second_page" else "2026-09-01"
        rows = [{"ID": "AN" + str(page).zfill(18), "FUNDCODE": CODE, "TITLE": "目录", "PUBLISHDATEDesc": day}]
        if page == 2 and failure == "empty_second_page":
            rows = []
        return json.dumps(
            {
                "ErrCode": 0,
                "PageIndex": page,
                "TotalCount": 3001 if page == 2 and failure == "changed_total" else 3000,
                "Data": rows,
            }
        ).encode()

    monkeypatch.setattr(client, "_get", response)
    try:
        with pytest.raises(ValueError, match="CATALOG_(?:CHANGED|ORDER|INCOMPLETE)"):
            client.news_catalog(CODE, "2026-09-28")
    finally:
        client.close()


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


def test_one_body_network_failure_does_not_abort_other_announcements(tmp_path):
    """目录成功后单篇网络失败应保留公告号，并继续取得其余原文。"""
    second = "AN" + "2" * 18

    class Client(NewsClient):
        def news_catalog(self, code, lower):
            first = super().news_catalog(code, lower)[0]
            return [first, {**first, "id": second}]

        def news_body(self, code, entry, old):
            if entry["id"] == ARTICLE:
                raise httpx.ReadError("真实网络失败替身")
            return super().news_body(code, entry, old)

    result = news.collect(CODE, client=Client(), root=tmp_path)
    assert result["status"] == "PARTIAL_SUCCESS" and result["catalog_complete"]
    assert result["errors"] == [{"article_id": ARTICLE, "reason": "ReadError"}]
    assert second in result["items"] and ARTICLE not in result["items"]


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
                        "fields": ["ts_code", "trade_date", "close", "pre_close", "pct_chg", "amount"],
                        "items": [
                            ["600000.SH", "20261016", 10, 9.9, 1, 20],
                            ["000001.SZ", "20261016", 11, 11.11, -1, 30],
                        ],
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


def test_empty_quote_response_is_retained_but_does_not_block_later_collection(tmp_path, monkeypatch):
    """先空后有数据时继续取得真实数据，不能把第一次空响应缓存一整天。"""
    instant = datetime.fromisoformat("2026-10-16T18:00:00+08:00")
    monkeypatch.setattr(companies, "now", lambda: instant)
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
            items = [] if len(calls) == 1 else [["600000.SH", "20261016", 10, 9.9, 1, 20]]
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "code": 0,
                    "data": {
                        "fields": ["ts_code", "trade_date", "close", "pre_close", "pct_chg", "amount"],
                        "items": items,
                    },
                },
            )

    collector = companies.MarketCompanyCollector(tmp_path)
    collector.client = Client()
    first, first_receipt = collector.query("daily", {"trade_date": "20261016"})
    second, _ = collector.query("daily", {"trade_date": "20261016"})
    assert first == [] and second[0]["ts_code"] == "600000.SH" and len(calls) == 2
    assert (tmp_path / first_receipt["file"]).exists()
    assert len(list((tmp_path / "query-attempts").glob("*.json"))) == 2
