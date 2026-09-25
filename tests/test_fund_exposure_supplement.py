"""新增来源必须守住日期、来源、分页与真实采集时间；不让补齐动作改写历史答案。"""

import hashlib
import json
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest
from app.integrations import cninfo_exposure as cn
from app.integrations import dbfund_supplement as db
from app.integrations.dbfund_news_mirrors import mirror_text
from app.services import fund_exposure_supplement as supplement
from app.services.direction_1d_protocol import ZONE
from app.services.fund_exposure_common import read, save


def financial(items):
    return {"data": {"fields": ["ts_code", "ann_date", "f_ann_date", "end_date", "revenue"], "items": items}}


def test_mirror_requires_matching_title_and_excludes_live_sidebar():
    body = "德邦基金历史访谈内容。" * 25
    html = (
        f'<html><title>陆阳访谈</title><div id="article">{body}<script>广告</script></div>'
        '<aside>今日市场消息</aside></html>'
    )
    assert mirror_text(html.encode("utf-8"), "#article", "陆阳访谈") == body
    with pytest.raises(ValueError, match="TITLE_MISMATCH"):
        mirror_text(html.encode("utf-8"), "#article", "其他基金访谈")


def test_financial_actual_publication_after_cutoff_is_quarantined():
    value = financial(
        [["000001.SZ", "20260820", "20260925", "20260630", 10], ["000001.SZ", "20260820", None, "20260630", None]]
    )
    rows, future = supplement.financial_rows(value, "000001.SZ", "20260924")
    assert len(rows) == len(future) == 1
    assert rows[0]["revenue"] is None  # 来源空值不能补成 0。


@pytest.mark.parametrize("code,day", [("OTHER", "20260820"), ("000001.SZ", "20260231")])
def test_financial_scope_and_invalid_calendar_date_rejected(code, day):
    with pytest.raises(ValueError, match="EXPOSURE_FINANCIAL_"):
        supplement.financial_rows(financial([[code, day, None, "20260630", 10]]), "000001.SZ", "20260924")


def test_source_bytes_must_match_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(supplement, "ROOT", tmp_path)
    (tmp_path / "raw.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        supplement.verified_bytes({"file": "raw.json", "sha256": hashlib.sha256(b"original").hexdigest()})


def test_expired_evidence_is_rejected_even_if_raw_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(supplement, "ROOT", tmp_path)
    monkeypatch.setattr(supplement, "now", lambda: datetime(2026, 9, 25, tzinfo=ZONE))
    (tmp_path / "raw.json").write_bytes(b"ok")
    with pytest.raises(ValueError, match="EVIDENCE_EXPIRED"):
        supplement.verified_bytes(
            {"file": "raw.json", "sha256": hashlib.sha256(b"ok").hexdigest(), "expires_at": "2026-09-24T23:00:00+08:00"}
        )


def nav_page(**changes):
    row = {"date": "2026-09-24", "fundcode": "002112", "netvalue": "5.1437", "totalnetvalue": "5.2917"}
    row.update(changes)
    return {"currentPage": 1, "dataList": [row]}


def test_nav_unit_and_accumulated_are_separate():
    row = db.parse_nav_page(nav_page(), 1, date(2026, 9, 24))[0]
    assert row["unit_nav"] == "5.1437" and row["accum_nav"] == "5.2917"


@pytest.mark.parametrize("change", [{"fundcode": "008888"}, {"date": "2026-09-25"}, {"netvalue": "NaN"}])
def test_official_nav_rejects_wrong_fund_future_date_and_nonfinite(change):
    with pytest.raises(ValueError, match="EXPOSURE_OFFICIAL_NAV_"):
        db.parse_nav_page(nav_page(**change), 1, date(2026, 9, 24))


def live_nav_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "SUPPLEMENT", tmp_path)
    monkeypatch.setattr(db, "verified_bytes", lambda _: b"verified-by-separate-test")
    days = [date(2026, 9, 21) + timedelta(days=i) for i in range(4)]
    nav = [{"nav_date": d, "unit_nav": Decimal("5"), "content_hash": "a"} for d in days[:-1]]
    rows = [
        {
            "date": str(d),
            "unit_nav": "5",
            "first_seen_at": "2026-09-24T22:00:00+08:00",
            "receipt": {"source": "official"},
        }
        for d in days
    ]
    save(tmp_path / "official-nav.json", {"rows": rows})
    return days, nav, rows


def test_official_fallback_never_backdates_and_marks_source(tmp_path, monkeypatch):
    days, nav, _ = live_nav_fixture(tmp_path, monkeypatch)
    before = datetime(2026, 9, 24, 21, tzinfo=ZONE)
    assert len(db.fill_live_nav_gap(nav, days, before)) == 3
    filled = db.fill_live_nav_gap(nav, days, before + timedelta(hours=2))
    assert len(filled) == 4 and len(nav) == 3
    assert filled[-1]["source_code"] == "DBFUND_OFFICIAL_PUBLIC"
    assert filled[-1]["ann_date"] is None


def test_official_conflict_stops_fallback(tmp_path, monkeypatch):
    days, nav, rows = live_nav_fixture(tmp_path, monkeypatch)
    rows[0]["unit_nav"] = "6"
    save(tmp_path / "official-nav.json", {"rows": rows}, replace=True)
    with pytest.raises(ValueError, match="NAV_CONFLICT"):
        db.fill_live_nav_gap(nav, days, datetime(2026, 9, 24, 23, tzinfo=ZONE))


@pytest.mark.parametrize("url", ["https://example.com/report.pdf", "http://static.cninfo.com.cn/a.pdf"])
def test_external_attachment_domain_rejected_before_network(url):
    with pytest.raises(ValueError, match="HOST_NOT_ALLOWED"):
        cn.PublicClient().fetch(url)


def test_cninfo_uses_record_count_when_totalpages_is_wrong(tmp_path, monkeypatch):
    monkeypatch.setattr(cn, "SUPPLEMENT", tmp_path)
    monkeypatch.setattr(
        cn,
        "holding_intervals",
        lambda: {
            "windows": {"000001.SZ": [["2026-09-01", "2026-09-24"]]},
            "maximum_pages_per_window": 150,
        },
    )
    calls = []

    def fetch(_, url, params=None):
        if params is None:
            return json.dumps({"stockList": [{"code": "000001", "orgId": "org"}]}).encode(), {}
        page = params["pageNum"]
        calls.append(page)
        items = [
            {
                "secCode": "000001",
                "announcementId": str(i),
                "announcementTitle": "财报",
                "announcementTime": datetime(2026, 9, 20, tzinfo=ZONE).timestamp() * 1000,
            }
            for i in (range(30) if page == 1 else range(30, 35))
        ]
        return json.dumps({"totalAnnouncement": 35, "totalpages": 1, "announcements": items}).encode(), {}

    monkeypatch.setattr(cn.PublicClient, "fetch", fetch)
    result = cn.acquire_announcements()
    assert calls == [1, 2] and result["rows"] == 35 and result["errors"] == []
    assert len(read(tmp_path / "company-announcements/000001.SZ.json")["rows"]) == 35


def test_main_business_splits_full_response_instead_of_accepting_first_hundred():
    calls = []

    class Provider:
        def query(self, api, params):
            calls.append(params)
            count = 100 if params["start_date"] != params["end_date"] else 60
            rows = [["000001.SZ", params["start_date"], str(i)] for i in range(count)]
            return {"data": {"fields": ["ts_code", "end_date", "bz_item"], "items": rows}}, {"page": len(calls)}

    value, receipts = supplement.main_business(Provider(), "000001.SZ", "20200101", "20200102")
    assert len(calls) == len(receipts) == 3
    assert len(value["data"]["items"]) == 120


def test_verified_main_business_merge_can_exceed_single_response_guard():
    """20 天各 60 条，分页后 1,200 条都应保留；单次原始响应仍必须防止截断。"""
    class Provider:
        def query(self, api, params, fields):
            assert fields == "ts_code,end_date,bz_item"
            count = 100 if params["start_date"] != params["end_date"] else 60
            rows = [["000001.SZ", params["start_date"], str(i)] for i in range(count)]
            return {"data": {"fields": fields.split(","), "items": rows}}, {"params": params}

    value, _ = supplement.main_business(
        Provider(), "000001.SZ", "20200101", "20200120", "ts_code,end_date,bz_item"
    )
    rows, future = supplement.financial_rows(value, "000001.SZ", "20260924", row_limit=None)
    assert len(rows) == 1200 and future == []
    with pytest.raises(ValueError, match="PAGINATION_REQUIRED"):
        supplement.financial_rows(value, "000001.SZ", "20260924")


def test_pdf_text_extractor_marks_blank_page_for_review():
    """没有文字的页面不能被标成正文提取完成，即使 PDF 原文件完整。"""
    import io

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    pages, status = db.article_text(buffer.getvalue(), True)
    assert pages == [""] and status == "PARTIAL_TEXT_CHECK_IMAGES"


@pytest.mark.parametrize("changed", [False, True])
def test_report_same_url_refresh_preserves_old_source_version(tmp_path, monkeypatch, changed):
    from types import SimpleNamespace

    from app.integrations import dbfund_reports as reports
    from app.services.direction_1d_protocol import digest

    monkeypatch.setattr(reports, "ROOT", tmp_path)
    at = datetime(2026, 9, 24, 23, tzinfo=ZONE)
    monkeypatch.setattr(reports, "now", lambda: at)
    monkeypatch.setattr(reports, "initialize", lambda: {})
    entry = {"url": "https://www.dbfund.com.cn/upload/pdf/report.pdf", "title": "示例"}
    monkeypatch.setattr(reports, "catalog", lambda _: [entry])
    old = b"%PDF-old"
    fresh = b"%PDF-new" if changed else old
    (tmp_path / "old.pdf").write_bytes(old)
    old_sha = hashlib.sha256(old).hexdigest()
    old_received = (at - timedelta(days=8)).isoformat()
    receipt_path = tmp_path / "report-receipts" / (digest(entry["url"]) + ".json")
    save(receipt_path, {"url": entry["url"], "file": "old.pdf", "sha256": old_sha, "received_at": old_received})
    old_parsed = {
        "report_end": "2026-06-30",
        "report_type": "HALF",
        "holding_count": 0,
        "raw": {"sha256": old_sha, "url": entry["url"]},
        "parsed_at": old_received,
    }
    save(tmp_path / "reports" / (old_sha + "-" + reports.PARSER_VERSION + ".json"), old_parsed)
    monkeypatch.setattr(reports, "bounded_get", lambda *_: fresh)
    monkeypatch.setattr(reports, "blob", lambda raw, _: (hashlib.sha256(raw).hexdigest(), "new.pdf"))
    monkeypatch.setattr(reports, "PdfReader", lambda _: SimpleNamespace(pages=[]))
    monkeypatch.setattr(
        reports, "parse_text", lambda *_: {"report_end": "2026-06-30", "report_type": "HALF", "holding_count": 0}
    )
    result = reports.acquire()
    assert not result["errors"] and result["reports"] == 1
    assert read(receipt_path)["received_at"] == (at.isoformat() if changed else old_received)
    archived = list((tmp_path / "report-receipt-versions").glob("*.json"))
    assert len(archived) == 1 and read(archived[0])["sha256"] == old_sha
