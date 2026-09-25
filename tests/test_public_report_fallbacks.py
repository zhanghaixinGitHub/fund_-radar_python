"""公开来源补缺的白名单、时间和断路器边界，测试不访问外部网络。"""

import json
from types import SimpleNamespace

import pytest
from app.integrations import public_fund_reports as primary
from app.integrations import sohu_fund_reports as sohu
from app.services import fund_training_readiness as readiness


@pytest.mark.parametrize(
    "code,url",
    [
        ("", "https://q.fund.sohu.com/q/notice.php?code="),
        ("007509", "https://q.fund.sohu.com/q/notice.php?code=002112"),
        ("007509", "https://evil.test/q/notice.php?code=007509"),
        ("007509", "https://q.fund.sohu.com/q/read.php?code=007509&id=1"),
    ],
)
def test_reprint_cannot_expand_scope_or_send_to_another_host(code, url):
    with pytest.raises(ValueError, match="SCOPE"):
        sohu.page(SimpleNamespace(), code, url)


def test_body_outage_does_not_prevent_separate_catalog_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(primary, "STORE", tmp_path)
    monkeypatch.setattr(primary, "blob", lambda raw, ext: ("sha", "raw.json"))
    monkeypatch.setattr(primary, "bounded_get", lambda *_a, **_k: json.dumps({"ErrCode": 0, "Data": [1]}).encode())
    client = primary.ReportClient("check")
    client.failed_requests = 4
    try:
        value, _ = client.get(primary.CATALOG_URL, {"fundcode": "006038"}, catalog=True)
        assert value["ErrCode"] == 0 and client.failed_requests == 4
        with pytest.raises(ValueError, match="TEMPORARILY_UNAVAILABLE"):
            client.get(primary.BODY_URL, {"art_code": "AN202403291629028802"})
    finally:
        client.close()


def test_reprint_uses_page_publication_and_checks_period(monkeypatch, tmp_path):
    monkeypatch.setattr(primary, "STORE", tmp_path)
    entry = {"ID": "AN202403291629028802", "report_end": "2023-03-31", "report_type": "QUARTER"}
    monkeypatch.setattr(
        sohu,
        "catalog",
        lambda *_: {
            ("2023-03-31", "QUARTER"): {
                "url": "https://q.fund.sohu.com/q/read.php?code=007509&id=1",
                "published_date": "2023-04-23",
                "catalog_receipt": {},
            }
        },
    )
    raw = (
        '<h1>华商润丰混合2023年第1季度报告</h1><div class="article_info"><div class="txt">'
        '<span class="c">2023-04-23</span></div><div id="sohu_content">基金007509<br>'
        '报告送出日期2023年4月21日</div></div>'
    ).encode()
    monkeypatch.setattr(sohu, "page", lambda *_a, **_k: (raw, {"sha256": "abc"}))
    value, receipt = sohu.report(None, "007509", entry)
    assert value["data"]["notice_date"] == "2023-04-23"
    assert not receipt["revised_after_receipt"]
    monkeypatch.setattr(sohu, "page", lambda *_a, **_k: (raw.replace(b"2023-04-23", b"2023-04-24"), {"sha256": "def"}))
    with pytest.raises(ValueError, match="DATE_MISMATCH"):
        sohu.report(None, "007509", entry)


def test_aggregated_prices_must_reproduce_source_values(monkeypatch):
    from app.integrations.tushare_sprint_stock_breadth_v2 import FIELDS

    raw = json.dumps({"data": {"fields": FIELDS, "items": [["600000.SH", "20160104", 10, 10, 0, 1, 1]]}}).encode()
    monkeypatch.setattr(readiness, "checked_raw", lambda _: raw)
    altered = {
        "receipts": {"600000.SH": {"sha256": "s"}},
        "days": {"2016-01-04": {"rows": {"600000.SH": {"close": 11}}}},
    }
    with pytest.raises(ValueError, match="QUOTE_AGGREGATE_CHANGED"):
        readiness.verify_derived_market(altered, {})


def test_legacy_quote_rounding_is_explicit_without_relaxing_default():
    from app.integrations.tushare_sprint_stock_breadth_v2 import validate_quote_values

    row = {"close": 10.01, "pre_close": 10.03, "pct_chg": -0.20, "vol": 1, "amount": 1}
    with pytest.raises(ValueError):
        validate_quote_values(row)
    assert validate_quote_values(row, rounding_tolerance=0.0051) < 0.0051
    with pytest.raises(ValueError):
        validate_quote_values({**row, "pct_chg": float("nan")}, rounding_tolerance=0.0051)
