"""补证边界测试：替身 HTTP 和 PDF，不外呼、不拟合、不访问封存标签。"""

from types import SimpleNamespace

import httpx
import pytest
from app.services.fund_002112_zero_fit_review import file_hash, read_json, save_once
from scripts import fund_002112_gap_evidence as task

URL = "https://www.dfham.com/upload/pdf/example.pdf"
ENTRY = {
    "fund_code": "017493",
    "report_end": "2023-06-30",
    "report_type": "HALF",
    "published_date": "2023-08-31",
    "title": "东方红新动力2023年中期报告",
}


@pytest.fixture
def output(tmp_path, monkeypatch):
    monkeypatch.setattr(task, "OUTPUT", tmp_path)
    save_once(
        tmp_path / "protocol.json",
        {
            "current_fit_budget": 0,
            "maximum_download_requests": 2,
            "report_targets": [],
            "missing_report_targets": [ENTRY],
        },
    )
    return tmp_path


def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("不允许测试外呼")

    monkeypatch.setattr(task.httpx, "Client", fail)


@pytest.mark.parametrize("url", ["https://unknown.test/a", "http://www.dfham.com/a", "https://secret@www.dfham.com/a"])
def test_unknown_or_credential_url_rejected(output, monkeypatch, url):
    no_network(monkeypatch)
    with pytest.raises(ValueError, match="PUBLIC_URL_NOT_ALLOWED"):
        task.fetch("outside", url)


def test_interruption_and_failure_consume_budget_without_retry(output, monkeypatch):
    no_network(monkeypatch)
    (output / "downloads").mkdir()
    for name in ("one", "two"):
        save_once(output / f"downloads/{name}.json", {"url": URL, "status": "ATTEMPT_RESERVED"})
    assert task.fetch("one", URL)["status"] == "ATTEMPT_RESERVED"
    with pytest.raises(ValueError, match="BUDGET_EXHAUSTED"):
        task.fetch("three", URL)


def test_redirect_is_not_followed(output, monkeypatch):
    client = httpx.Client
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://example.invalid/login"})

    monkeypatch.setattr(task.httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(handler), **kw))
    result = task.fetch("redirect", URL)
    assert result["status"] == "UNRESOLVED" and result["http_status"] == 302
    assert task.fetch("redirect", URL) == result
    assert calls == [URL]


def test_successful_http_html_is_not_a_report(output, monkeypatch):
    client = httpx.Client
    monkeypatch.setattr(
        task.httpx,
        "Client",
        lambda **kw: client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"<script>challenge</script>")), **kw
        ),
    )
    result = task.fetch("html", URL)
    assert result["status"] == "RECEIVED"
    with pytest.raises(ValueError, match="SOURCE_IS_NOT_PDF"):
        task.parse_received(ENTRY, result)
    assert read_json(output / "downloads/html.json")["status"] == "ATTEMPT_RESERVED"


def test_cached_raw_tampering_detected(output, monkeypatch):
    raw = output / "raw.pdf"
    raw.write_bytes(b"%PDF old")
    (output / "downloads").mkdir()
    receipt = {"url": URL, "status": "RECEIVED", "path": str(raw), "sha256": file_hash(raw)}
    save_once(output / "downloads/cached.json", {"url": URL, "status": "ATTEMPT_RESERVED"})
    save_once(output / "cached-receipt.json", receipt)
    no_network(monkeypatch)
    assert task.fetch("cached", URL) == receipt
    raw.write_bytes(b"%PDF changed")
    with pytest.raises(ValueError, match="RECEIPT_CONTENT_CHANGED"):
        task.fetch("cached", URL)
    with pytest.raises(ValueError, match="RECEIPT_CONTENT_CHANGED"):
        task.parse_received(ENTRY, receipt)


@pytest.mark.parametrize(
    "change,error",
    [
        ({"report_end": "2025-12-31"}, "OUTSIDE_FIXED_SCOPE"),
        ({"fund_code": "002112"}, "OUTSIDE_FIXED_SCOPE"),
        ({"published_date": "2023-08-30"}, "PUBLICATION_CHANGED"),
    ],
)
def test_scope_and_publication_cannot_be_relaxed(output, change, error):
    with pytest.raises(ValueError, match=error):
        task.parse_received({**ENTRY, **change}, {})


def test_new_budget_cannot_be_inferred_from_pending_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(task, "OUTPUT", tmp_path)
    save_once(tmp_path / "protocol.json", {"current_fit_budget": 12})
    with pytest.raises(ValueError, match="ZERO_FIT_BUDGET_CHANGED"):
        task.freeze()


def test_source_cannot_self_assert_eligibility(output, monkeypatch):
    raw = output / "raw.pdf"
    raw.write_bytes(b"%PDF fixture")
    receipt = {
        "url": URL,
        "status": "RECEIVED",
        "path": str(raw),
        "sha256": file_hash(raw),
        "historical_version_verified": True,
        "training_eligible": True,
    }
    monkeypatch.setattr(
        task,
        "PdfReader",
        lambda p: SimpleNamespace(
            is_encrypted=False, pages=[SimpleNamespace(extract_text=lambda: "基金原文替身")], metadata={}
        ),
    )
    monkeypatch.setattr(task, "validate_pdf_identity", lambda pages, code: None)
    monkeypatch.setattr(task, "parse_text", lambda *a, **kw: dict(ENTRY))
    result = task.parse_received({**ENTRY, "training_eligible": True}, receipt)
    assert result["available_at"] == "2023-09-01T08:00:00+08:00"
    assert result["training_eligible"] is False and result["historical_version_verified"] is False


def test_pdf_period_mismatch_rejected(output, monkeypatch):
    raw = output / "raw.pdf"
    raw.write_bytes(b"%PDF fixture")
    receipt = {"status": "RECEIVED", "path": str(raw), "sha256": file_hash(raw)}
    monkeypatch.setattr(
        task,
        "PdfReader",
        lambda p: SimpleNamespace(
            is_encrypted=False, pages=[SimpleNamespace(extract_text=lambda: "替身")], metadata={}
        ),
    )
    monkeypatch.setattr(task, "validate_pdf_identity", lambda pages, code: None)
    monkeypatch.setattr(task, "parse_text", lambda *a, **kw: dict(ENTRY, report_type="QUARTER"))
    with pytest.raises(ValueError, match="REPORT_PERIOD_OR_IDENTITY_MISMATCH"):
        task.parse_received(ENTRY, receipt)


def test_search_cannot_request_sealed_period(output, monkeypatch):
    no_network(monkeypatch)
    with pytest.raises(ValueError, match="PUBLIC_SEARCH_SCOPE_INVALID"):
        task.fetch(
            "sealed",
            "https://www.cninfo.com.cn/new/hisAnnouncement/query",
            form={"searchkey": "华夏磐泰", "seDate": "2025-01-01~2025-12-31", "pageNum": 1, "pageSize": 30},
        )
