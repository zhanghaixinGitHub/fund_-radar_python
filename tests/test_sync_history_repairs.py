"""本轮同步修复的证据边界：退市不是零行情，转载不是官网原件。"""

import json
from copy import deepcopy

import pytest
from app.integrations import dbfund_report_recovery as recovery
from app.integrations import dbfund_reports
from app.integrations.tushare_sprint_stock_breadth_v2 import FIELDS
from app.services import fund_training_readiness as readiness


def absence(monkeypatch):
    bodies = {
        "quotes": {"code": 0, "data": {"fields": FIELDS, "items": []}},
        "security": {
            "code": 0,
            "data": {
                "fields": ["ts_code", "list_date", "delist_date"],
                "items": [["600005.SH", "19990803", "20170214"]],
            },
        },
    }
    monkeypatch.setattr(readiness, "checked_raw", lambda r: json.dumps(bodies[r["file"]]).encode())
    issue = {
        "stock_code": "600005.SH",
        "requested_dates": ["2017-02-27", "2017-03-30"],
        "reason": "REPORT_SECURITY_OUTSIDE_LISTING",
        "retryable": False,
        "unresolved_dates": [],
        "explained_dates": {d: "AFTER_DELISTING" for d in ("2017-02-27", "2017-03-30")},
        "quote_receipt": {"file": "quotes"},
        "security_receipts": [{"file": "security"}],
    }
    return issue, bodies


def test_proven_outside_listing_is_separate_and_does_not_create_quotes(monkeypatch):
    issue, _ = absence(monkeypatch)
    errors, excluded = readiness.split_quote_issues([issue, {"reason": "ReadTimeout"}])
    assert errors == [{"reason": "ReadTimeout"}] and excluded == [issue]
    assert "rows" not in excluded[0]


@pytest.mark.parametrize("fault", ["code", "date", "missing_day", "unresolved", "nonempty", "expired"])
def test_absence_flags_cannot_bypass_original_evidence(monkeypatch, fault):
    issue, bodies = absence(monkeypatch)
    if fault == "code":
        bodies["security"]["data"]["items"][0][0] = "600006.SH"
    elif fault == "date":
        bodies["security"]["data"]["items"][0][2] = "20170401"
    elif fault == "missing_day":
        issue["explained_dates"].pop("2017-03-30")
    elif fault == "unresolved":
        issue["unresolved_dates"] = ["2017-02-27"]
    elif fault == "nonempty":
        bodies["quotes"]["data"]["items"] = [["600005.SH"]]
    else:
        monkeypatch.setattr(
            readiness, "checked_raw", lambda _: (_ for _ in ()).throw(ValueError("READINESS_SOURCE_EXPIRED"))
        )
    errors, excluded = readiness.split_quote_issues([issue])
    assert errors and errors[0]["proof_error"] and excluded == []


def html_report():
    return (
        "<div class='content bordernone'><h2>鑫星价值：2020年第1季度报告</h2>"
        "<p class='oif'>时间：2020-04-21 源自：基金公司网站</p>"
        "<div class='text-content'>德邦鑫星价值002112 报告送出日期：2020年4月21日<br>正文</div></div>"
    ).encode()


def test_recovery_checks_scope_dates_raw_hash_and_reuses_body(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    raw = html_report()
    (tmp_path / "raw.html").write_bytes(raw)
    import hashlib

    monkeypatch.setattr(recovery, "blob", lambda *_: (hashlib.sha256(raw).hexdigest(), "raw.html"))
    calls = []
    monkeypatch.setattr(dbfund_reports, "bounded_get", lambda *_: calls.append(1) or raw)
    monkeypatch.setattr(
        dbfund_reports,
        "parse_text",
        lambda *_a, **_k: {
            "fund_code": "002112",
            "report_end": "2020-03-31",
            "report_type": "QUARTER",
            "published_date": "2020-04-21",
        },
    )
    entry = {"url": recovery.PRIMARY, "title": recovery.TITLE}
    failure = {"url": recovery.PRIMARY, "http_status": 404}
    for _ in range(2):
        parsed = recovery.recover_report(None, entry, failure)
        assert parsed["raw"]["primary_failure"] == failure
        assert parsed["raw"]["format"] == "HTML" and parsed["raw"]["url"] == recovery.REPRINT
    assert len(calls) == 1
    (tmp_path / "raw.html").write_bytes(b"changed")
    with pytest.raises(ValueError, match="RAW_CHANGED"):
        recovery.recover_report(None, entry, failure)
    with pytest.raises(ValueError, match="SCOPE"):
        recovery.recover_report(None, {**entry, "url": "https://evil.test/a"}, failure)


@pytest.mark.parametrize("fault", ["wrapper_date", "title", "period", "code"])
def test_recovery_rejects_other_report_or_changed_publication(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    raw = html_report()
    if fault == "wrapper_date":
        raw = raw.replace(b"2020-04-21", b"2020-04-22")
    if fault == "title":
        raw = raw.replace("2020年第1".encode(), "2021年第1".encode())
    monkeypatch.setattr(dbfund_reports, "bounded_get", lambda *_: raw)
    monkeypatch.setattr(recovery, "blob", lambda *_: ("sha", "raw.html"))
    parsed = {
        "fund_code": "002112",
        "report_end": "2020-03-31",
        "report_type": "QUARTER",
        "published_date": "2020-04-21",
    }
    if fault == "period":
        parsed["report_end"] = "2021-03-31"
    if fault == "code":
        parsed["fund_code"] = "001412"
    monkeypatch.setattr(dbfund_reports, "parse_text", lambda *_a, **_k: deepcopy(parsed))
    with pytest.raises(ValueError, match="MISMATCH"):
        recovery.recover_report(None, {"url": recovery.PRIMARY, "title": recovery.TITLE}, {"http_status": 404})


def test_changed_reprint_cannot_be_backdated_to_original_publication(tmp_path, monkeypatch):
    """原文重取发生变化时，历史可得时间至少为实际取得新版的时间。"""
    import hashlib

    from app.services.fund_exposure_common import read
    from app.services.fund_materials_store import versioned_save

    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    payload = [html_report()]

    def blob(raw, _extension):
        key = hashlib.sha256(raw).hexdigest()
        (tmp_path / (key + ".html")).write_bytes(raw)
        return key, key + ".html"

    monkeypatch.setattr(recovery, "blob", blob)
    monkeypatch.setattr(dbfund_reports, "bounded_get", lambda *_: payload[0])
    monkeypatch.setattr(
        dbfund_reports,
        "parse_text",
        lambda *_a, **_k: {
            "fund_code": "002112",
            "report_end": "2020-03-31",
            "report_type": "QUARTER",
            "published_date": "2020-04-21",
            "available_at": "2020-04-22T08:00:00+08:00",
        },
    )
    entry, failure = {"url": recovery.PRIMARY, "title": recovery.TITLE}, {"url": recovery.PRIMARY, "http_status": 404}
    recovery.recover_report(None, entry, failure)
    cache = next((tmp_path / "readiness-extended/report-recovery").glob("*.json"))
    receipt = read(cache)
    receipt["checked_at"] = "2020-04-21T08:00:00+08:00"
    versioned_save(cache, receipt)
    payload[0] += b"\nchanged"
    result = recovery.recover_report(None, entry, failure)
    assert result["raw"]["revised_after_receipt"] is True
    assert result["available_at"] == result["raw"]["received_at"]
