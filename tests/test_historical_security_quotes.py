"""历史证券身份与报告修复边界；只用内存原文和临时目录，不联网。"""

import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from app.integrations.fund_report_layout_v3 import normalize_layout
from app.integrations.tushare_sprint_stock_breadth_v2 import FIELDS
from app.services import historical_security_quotes as quotes
from app.services.fund_exposure_quotes import safe_error


def response(code="920819.BJ", day="20230106"):
    return {"code": 0, "data": {"fields": FIELDS, "items": [[code, day, 10, 10, 0, 1, 1]]}}


def test_provider_alias_keeps_report_identity_and_only_valid_dates():
    result = quotes.parse_history(response(), "833819.BJ", ["2023-01-06"])
    assert result["2023-01-06"]["close"] == 10
    for code, day in [("833819.BJ", "2025-05-06"), ("833819.BJ", "2021-11-14"), ("999999.BJ", "2023-01-06")]:
        with pytest.raises(ValueError, match="IDENTITY_DATE_UNVERIFIED"):
            quotes.parse_history(response(), code, [day])
    with pytest.raises(ValueError, match="IDENTITY_OR_DATE"):
        quotes.parse_history(response("920445.BJ"), "833819.BJ", ["2023-01-06"])


def test_missing_or_invalid_quotes_are_not_zero_or_success():
    empty = {"code": 0, "data": {"fields": FIELDS, "items": []}}
    assert quotes.parse_history(empty, "833819.BJ", ["2023-01-06"]) == {}
    bad = response()
    bad["data"]["items"][0][2] = None
    with pytest.raises(ValueError, match="NUMERIC"):
        quotes.parse_history(bad, "833819.BJ", ["2023-01-06"])


def test_versioned_source_roundtrip_and_tamper_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(quotes, "ROOT", tmp_path)
    monkeypatch.setattr(quotes, "STORE", tmp_path / "new-history")
    raw = json.dumps(response()).encode()
    (tmp_path / "raw.json").write_bytes(raw)
    receipt = {"file": "raw.json", "sha256": hashlib.sha256(raw).hexdigest(), "expires_at": "2099-01-01T00:00:00+08:00"}
    provider = SimpleNamespace(query=lambda *args: (response(), receipt))
    result = quotes.acquire_history("833819.BJ", ["2023-01-06", "2023-01-09"], provider)
    assert result["status"] == "UNRESOLVED" and result["unresolved_dates"] == ["2023-01-09"]
    loaded = quotes.load_history()["2023-01-06"]["833819.BJ"]
    assert loaded["identity"]["provider_code"] == "920819.BJ"
    assert loaded["identity"]["historical_code"] == "833819.BJ"
    (tmp_path / "raw.json").write_text("{}")
    with pytest.raises(ValueError, match="RAW_CHANGED"):
        quotes.load_history()


def test_delisted_window_is_proven_without_replacing_security():
    calls = []

    def query(api, params, fields):
        calls.append(params)
        return {
            "data": {
                "fields": ["ts_code", "list_date", "delist_date"],
                "items": [["600005.SH", "19990803", "20170214"]],
            }
        }, {"sha256": "proof"}

    result = quotes.explain_empty_history("600005.SH", ["2017-02-27", "2017-03-30"], SimpleNamespace(query=query))
    assert result["reason"] == "REPORT_SECURITY_OUTSIDE_LISTING" and result["retryable"] is False
    assert set(result["explained_dates"].values()) == {"AFTER_DELISTING"}
    assert calls == [{"ts_code": "600005.SH", "list_status": "D"}]


def test_http_status_is_preserved_without_url_or_credentials():
    request = httpx.Request("GET", "https://example.test/report?token=private")
    error = httpx.HTTPStatusError("private body", request=request, response=httpx.Response(404, request=request))
    assert safe_error(error) == "HTTP_STATUS_404"


def test_report_page_number_removed_only_after_exact_report_header():
    text = (
        "5.3报告期末按公允价值占基金资产净值比例大小排序的股票投资明细\n"
        "1 600000 浦发银行 10 100.00 50.00\n华夏磐泰混合型证券投资基金（LOF）2021年第4季度报告\n"
        "9\n2 600001 测试股票 10 100.00 50.00\n5.4报告期末按债券品种分类的债券投资组合\n"
    )
    _, evidence = normalize_layout([text], fund_name="华夏磐泰")
    assert any(e["rule"] == "REPORT_HEADER_PAGE_NUMBER" for e in evidence["blocks"])
    with pytest.raises(ValueError):
        normalize_layout(
            [text.replace("华夏磐泰混合型证券投资基金（LOF）2021年第4季度报告", "其他数值")], fund_name="华夏磐泰"
        )
