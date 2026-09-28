"""资料补查的网络回执、范围和来源完整性边界；使用替身，不实际外呼。"""

import hashlib

import httpx
import pytest
from app.services.fund_002112_zero_fit_review import read_json, save_once
from scripts import fund_002112_peer_gap_materials as task

URL = "https://static.cninfo.com.cn/finalpage/2021-04-21/1209736995.PDF"


@pytest.fixture
def output(tmp_path, monkeypatch):
    monkeypatch.setattr(task, "OUTPUT", tmp_path)
    save_once(tmp_path / "protocol.json", {"maximum_download_requests": 40, "report_targets": []})
    return tmp_path


def prohibit_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Unexpected HTTP call")

    monkeypatch.setattr(task.httpx, "Client", fail)


def test_received_cache_verified_without_network(output, monkeypatch):
    raw = output / "source.pdf"
    raw.write_bytes(b"%PDF-1.7 local fixture")
    save_once(
        output / "cached-receipt.json",
        {"url": URL, "status": "RECEIVED", "path": str(raw), "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()},
    )
    (output / "downloads").mkdir()
    save_once(output / "downloads/cached.json", {"url": URL, "status": "ATTEMPT_RESERVED"})
    prohibit_network(monkeypatch)
    assert task.fetch("cached", URL)["status"] == "RECEIVED"
    raw.write_bytes(b"changed")
    with pytest.raises(ValueError, match="RECEIPT_CONTENT_CHANGED"):
        task.fetch("cached", URL)


def test_interrupted_attempt_is_not_retried(output, monkeypatch):
    (output / "downloads").mkdir()
    save_once(output / "downloads/interrupted.json", {"url": URL, "status": "ATTEMPT_RESERVED"})
    prohibit_network(monkeypatch)
    assert task.fetch("interrupted", URL)["status"] == "ATTEMPT_RESERVED"


def test_budget_counts_failed_attempts(output, monkeypatch):
    (output / "downloads").mkdir()
    for i in range(40):
        save_once(output / f"downloads/failed-{i}.json", {"status": "ATTEMPT_RESERVED"})
    prohibit_network(monkeypatch)
    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        task.fetch("new", URL)


def test_redirect_is_preserved_and_not_followed(output, monkeypatch):
    real_client = httpx.Client
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://example.invalid/login"})

    monkeypatch.setattr(task.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    result = task.fetch("redirect", URL)
    assert result["status"] == "UNRESOLVED"
    assert result["http_status"] == 302
    assert task.fetch("redirect", URL) == result
    assert calls == [URL]
    assert read_json(output / "downloads/redirect.json")["status"] == "ATTEMPT_RESERVED"


@pytest.mark.parametrize(
    "url",
    ["https://unknown.test/a.pdf", "http://static.cninfo.com.cn/a.pdf", "https://secret@static.cninfo.com.cn/a.pdf"],
)
def test_out_of_scope_host_rejected(output, monkeypatch, url):
    prohibit_network(monkeypatch)
    with pytest.raises(ValueError, match="URL_NOT_ALLOWED"):
        task.fetch("out", url)


def test_search_scope_and_receipt_identity(output, monkeypatch):
    prohibit_network(monkeypatch)
    with pytest.raises(ValueError, match="SEARCH_SCOPE_INVALID"):
        task.fetch("out", URL, form={"searchkey": "new company"})
    (output / "downloads").mkdir()
    save_once(output / "downloads/cached.json", {"url": URL, "status": "ATTEMPT_RESERVED"})
    with pytest.raises(ValueError, match="URL_CHANGED"):
        task.fetch("cached", URL + "?changed=1")


def test_parse_rejects_unlisted_period_before_reading_file(output):
    with pytest.raises(ValueError, match="REPORT_OUTSIDE_FIXED_SCOPE"):
        task.parse_report({"fund_code": "160323", "report_end": "2025-12-31", "report_type": "ANNUAL"}, {})
