"""参照基金补充的身份、缓存、来源版本、缺口和任务终态检查。"""

import hashlib
import json
from datetime import datetime
from types import SimpleNamespace

import pytest
from app.integrations import public_fund_reports as reports
from app.integrations.dbfund_reports import parse_text
from app.services import fund_peer_materials as peers
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import read, save
from tests.test_fund_exposure import report_text


@pytest.mark.parametrize("code", [None, "", " ", "002112", "000001", "005187.OF"])
def test_peer_catalog_rejects_unbounded_or_unregistered_scope(code):
    with pytest.raises(ValueError, match="PEER_SCOPE"):
        reports.catalog(SimpleNamespace(get=lambda *_a, **_k: pytest.fail("network")), code)


def test_shared_parser_keeps_fund_identity_and_chinese_publication():
    body = report_text()[0].replace("德邦鑫星价值 002112", "华夏磐泰 160323")
    body = body.replace("2021年8月31日", "二〇二一年八月三十一日")
    result = parse_text(
        [body], "华夏磐泰2021年中期报告", fund_code="160323", fund_name="华夏磐泰", master_code="160323"
    )
    assert result["fund_code"] == "160323"
    assert result["published_date"] == "2021-08-31"
    with pytest.raises(ValueError, match="IDENTITY"):
        parse_text([body.replace("二〇二一年八月三十一日", "2021年8月31日")], "德邦鑫星价值2021年中期报告")


def test_report_period_excludes_summaries_and_distinguishes_half_and_quarter():
    assert reports.report_period("基金2023年第2季度报告") == ("2023-06-30", "QUARTER")
    assert reports.report_period("基金2023年中期报告") == ("2023-06-30", "HALF")
    assert reports.report_period("富国新活力二0二三年年度报告") == ("2023-12-31", "ANNUAL")
    assert reports.report_period("富国新活力二〇二四年第3季度报告") == ("2024-09-30", "QUARTER")
    assert reports.report_period("基金2023年年度报告摘要") is None
    assert reports.report_period("关于基金2023年年度报告的提示性公告") is None
    assert len(peers.expected_periods("005187")) == 25
    assert len(peers.expected_periods("017493")) == 12


def test_catalog_empty_and_duplicate_pages_are_not_success():
    client = SimpleNamespace(get=lambda *_a, **_k: ({"TotalCount": 0, "PageIndex": 1, "Data": []}, {}))
    with pytest.raises(ValueError, match="EMPTY"):
        reports.catalog(client, "005187")
    entry = {
        "ID": "AN202403301629177879",
        "FUNDCODE": "005187",
        "TITLE": "长安鑫兴2023年年度报告",
        "PUBLISHDATEDesc": "2024-03-30",
    }
    client.get = lambda _url, params, **_k: ({"TotalCount": 2, "PageIndex": params["pageIndex"], "Data": [entry]}, {})
    with pytest.raises(ValueError, match="DUPLICATE"):
        reports.catalog(client, "005187")


def test_unchanged_body_is_reused_and_corruption_is_not_redownloaded_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "ROOT", tmp_path)
    monkeypatch.setattr(reports, "STORE", tmp_path / "peers")
    monkeypatch.setattr(reports, "now", lambda: datetime.fromisoformat("2026-09-25T18:00:00+08:00"))
    params = {"client_source": "web_fund", "show_all": 1, "art_code": "AN202403301629177879"}
    raw = b'{"success":1,"data":{"notice_content":"saved"}}'
    path = tmp_path / "raw.json"
    path.write_bytes(raw)
    receipt = {"sha256": hashlib.sha256(raw).hexdigest(), "file": "raw.json", "checked_at": "2026-09-25T17:00:00+08:00"}
    output = tmp_path / "peers/receipts" / (digest({"url": reports.BODY_URL, "params": params}) + ".json")
    save(output, receipt)
    monkeypatch.setattr(reports, "bounded_get", lambda *_a, **_k: pytest.fail("unchanged body downloaded"))
    client = reports.ReportClient("new-check")
    try:
        assert client.get(reports.BODY_URL, params)[0]["success"] == 1
        assert client.count == 0
        path.write_bytes(raw + b" ")
        with pytest.raises(ValueError, match="CACHED_BODY_CHANGED"):
            client.get(reports.BODY_URL, params)
    finally:
        client.close()


def test_body_metadata_cannot_substitute_another_fund(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "STORE", tmp_path)
    entry = {"FUNDCODE": "005187", "ID": "AN202403301629177879", "report_end": "2023-12-31", "report_type": "ANNUAL"}
    client = SimpleNamespace(
        get=lambda *_a: ({"data": {"art_code": entry["ID"], "security": [{"stock": "002112"}]}}, {})
    )
    with pytest.raises(ValueError, match="IDENTITY"):
        reports.acquire_one(client, "005187", entry)


def test_quote_reuse_validates_hash_and_excludes_unverified_market(tmp_path, monkeypatch):
    monkeypatch.setattr(peers, "ROOT", tmp_path)
    monkeypatch.setattr(peers, "permission", lambda: {"authorized_api_names": ["daily", "index_daily"]})
    monkeypatch.setattr(peers, "Provider", lambda: pytest.fail("must reuse raw"))
    monkeypatch.setattr(peers, "parse", lambda raw, day: None)
    raw = json.dumps(
        {"data": {"items": [["600000.SH", "20230103", 1, 1, 0, 10, 10], ["830001.BJ", "20230103", 1, 1, 0, 10, 10]]}}
    ).encode()
    p = tmp_path / "raw.json"
    p.write_bytes(raw)
    receipt = {"file": "raw.json", "sha256": hashlib.sha256(raw).hexdigest(), "expires_at": "2099-01-01T00:00:00+08:00"}
    save(tmp_path / "stock-days/2023-01-03.json", {"receipt": receipt})
    days = peers.QuoteDays()
    assert set(days.get("2023-01-03")["rows"]) == {"600000.SH"}
    p.write_bytes(b"broken")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        peers.QuoteDays().get("2023-01-03")


def test_empty_body_can_be_retried_without_redownloading_successful_body(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "ROOT", tmp_path)
    monkeypatch.setattr(reports, "STORE", tmp_path / "peers")
    params = {"client_source": "web_fund", "show_all": 1, "art_code": "AN202403301629177879"}
    raw = b'{"success":1,"data":{"notice_content":""}}'
    (tmp_path / "body.json").write_bytes(raw)
    path = reports.STORE / "receipts" / (digest({"url": reports.BODY_URL, "params": params}) + ".json")
    save(
        path,
        {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "file": "body.json",
            "received_at": reports.now().isoformat(),
            "checked_at": reports.now().isoformat(),
        },
    )
    client = reports.ReportClient("retry")
    calls = []
    fresh = b'{"success":1,"data":{"notice_content":"available"}}'
    monkeypatch.setattr(reports, "bounded_get", lambda *a, **kw: calls.append(kw) or fresh)
    monkeypatch.setattr(reports, "blob", lambda raw, kind: (hashlib.sha256(raw).hexdigest(), "fresh.json"))
    try:
        client.invalidate_body(params["art_code"])
        value, receipt = client.get(reports.BODY_URL, params)
        assert len(calls) == 1 and value["data"]["notice_content"] == "available"
        assert receipt["previous_sha256"] and not receipt["revised_after_receipt"]
    finally:
        client.close()


def test_revised_report_cannot_be_backdated_to_historical_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "STORE", tmp_path)
    code, article = "005187", "AN202403301629177879"
    body = report_text()[0].replace("德邦鑫星价值 002112", "长安鑫兴 005187") + "\n" + "说明" * 800
    entry = {
        "ID": article,
        "FUNDCODE": code,
        "report_end": "2021-06-30",
        "report_type": "HALF",
        "published_date": "2021-08-31",
    }
    receipt = {"sha256": "a" * 64, "received_at": "2026-09-25T12:00:00+08:00", "revised_after_receipt": True}
    client = SimpleNamespace(
        get=lambda *a: (
            {
                "data": {
                    "art_code": article,
                    "security": [{"stock": code}],
                    "notice_title": "长安鑫兴2021年中期报告",
                    "notice_content": body,
                }
            },
            receipt,
        )
    )
    parsed, _, _ = reports.acquire_one(client, code, entry)
    assert parsed["available_at"] == receipt["received_at"]
    assert parsed["training_eligible"] is False


def test_quote_gap_keeps_unknown_separate_from_proven_nontrading(tmp_path, monkeypatch):
    monkeypatch.setattr(peers, "STORE", tmp_path)

    class Provider:
        count = 0
        scope = {}

        def query(self, api, params, fields=""):
            assert params["ts_code"] == "688001.SH"
            if api == "stock_basic":
                return {"data": {"fields": ["ts_code", "list_date"], "items": [["688001.SH", "20230104"]]}}, {}
            return {
                "data": {
                    "fields": ["ts_code", "trade_date", "suspend_type", "suspend_timing"],
                    "items": [["688001.SH", "20230105", "S", None]],
                }
            }, {}

    monkeypatch.setattr(peers, "SupplementProvider", Provider)
    result = peers.explain_quote_gaps(
        {"688001.SH": {"2023-01-03", "2023-01-05", "2023-01-06"}, "833819.BJ": {"2023-01-06"}}, lambda *a: None
    )
    assert result["stocks"]["688001.SH"]["explained"] == {
        "2023-01-03": "BEFORE_LISTING",
        "2023-01-05": "FULL_DAY_SUSPENSION",
    }
    assert result["stocks"]["688001.SH"]["unresolved_dates"] == ["2023-01-06"]
    assert len(result["unresolved_stocks"]) == 2


def test_recheck_failure_preserves_previously_verified_report(tmp_path, monkeypatch):
    monkeypatch.setattr(peers, "STORE", tmp_path)
    monkeypatch.setattr(peers, "PEERS", {"005187": "长安鑫兴"})
    document = {"file": "old.json", "sha256": "old", "report_end": "2023-12-31", "report_type": "ANNUAL"}
    save(tmp_path / "005187/manifest.json", {"documents": {"a": document}})
    monkeypatch.setattr(
        peers, "ReportClient", lambda *a: SimpleNamespace(count=0, close=lambda: None, invalidate_body=lambda a: None)
    )
    monkeypatch.setattr(peers, "catalog", lambda *a: ([{"ID": "a"}], []))
    monkeypatch.setattr(
        peers, "acquire_one", lambda *a: (_ for _ in ()).throw(ValueError("REPORT_BODY_EMPTY_OR_TRUNCATED"))
    )
    result = peers.sync_reports("retry", lambda *a: None)
    saved = read(tmp_path / "005187/manifest.json")
    assert saved["documents"]["a"]["file"] == "old.json"
    assert result["errors"]
