"""日本新财报必须有同证券原文及真实时期，聚合摘要和其他证券不能放行。"""
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest
from app.services import fund_prediction_jp_financials as jp
from app.services.direction_1d_protocol import digest
from app.services.fund_materials_store import versioned_save

PDF_URL = "https://" + jp.PDF_HOST + "/disclosure/20261030/20260924539503.pdf"
BODY = ("2027年3月期 第2四半期決算短信\n2026年10月30日\nコード番号 8035 URL\n"
        "連結業績（2026年4月1日～2026年9月30日）\n金額の単位：百万円")
HTML = ('<title>東京エレクトロン【8035】決算情報</title><a href="' + PDF_URL
        + '">2027年3月期第2四半期決算短信（PDF）</a>')


@pytest.mark.parametrize("problem", ["wrong_code", "wrong_publication", "invalid_market"])
def test_reporting_identity_rejects_other_security_or_date(problem):
    args = [BODY, "8035.JP", "2026-10-30"]
    if problem == "wrong_code":
        args[0] = BODY.replace("番号 8035", "番号 8036")
    if problem == "wrong_publication":
        args[2] = "2026-10-29"
    if problem == "invalid_market":
        args[1] = "8035.HK"
    with pytest.raises(ValueError):
        jp.reporting_identity(*args)


def test_next_fiscal_year_is_not_a_future_realized_period():
    assert jp.reporting_identity(BODY, "8035.JP", "2026-10-30") == "2026-09-30"
    assert jp.reporting_identity(BODY.replace("2026年9月30日", "2027年3月31日"), "8035.JP", "2026-10-30") is None


def test_explicit_from_until_financial_column_preserves_real_period():
    assert jp.reporting_identity(BODY.replace("～", " 至 "), "8035.JP", "2026-10-30") == "2026-09-30"


@pytest.mark.parametrize("problem", ["other_ticker", "summary_link", "future_link", "foreign_host"])
def test_source_link_requires_exact_ticker_original_and_real_date(problem):
    html = HTML
    if problem == "other_ticker":
        html = html.replace("【8035】", "【8036】")
    if problem == "summary_link":
        html = html.replace("決算短信", "AI要約")
    if problem == "future_link":
        html = html.replace("20261030/", "20261106/")
    if problem == "foreign_host":
        html = html.replace(jp.PDF_HOST, "example.com")
    with pytest.raises(ValueError):
        jp.financial_link(html, "8035.JP", "2026-11-05")


def test_collection_creates_source_directory_and_reads_existing_native_pages(tmp_path, monkeypatch):
    """真实新目录写入和AES兼容页读取都走已有资料库接口，无加密依赖安装。"""
    monkeypatch.setattr(jp, "now", lambda: datetime.fromisoformat("2026-11-05T09:00:00+08:00"))
    monkeypatch.setattr(jp, "write_source", lambda path, raw: path.write_bytes(raw))
    monkeypatch.setattr(jp, "versioned_save", lambda path, value: None)

    class Reader:
        def __enter__(self):
            return iter([SimpleNamespace(close=lambda: None, get_textpage=lambda: SimpleNamespace(
                close=lambda: None, get_text_range=lambda: BODY))])

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(jp, "document_pdfium", lambda path: Reader())
    calls = []

    def request(request):
        calls.append(str(request.url))
        content = HTML.encode() if request.url.path.endswith("financials") else b"%PDF-AES-test"
        return httpx.Response(200, content=content)

    with httpx.Client(transport=httpx.MockTransport(request)) as client:
        result = jp.collect("8035.JP", tmp_path / "new" / "company", client=client)
    assert result["status"] == "ORIGINAL_PARSED_READING_REQUIRED"
    assert result["report_end"] == "2026-09-30" and result["body"] == BODY
    assert len(calls) == 2 and len(result["source_attempts"]) == 2
    assert result["body_scope"] == "FULL_TEXT" and digest(result["body"])


def test_disclosure_fallback_selects_latest_real_original_without_forecast(tmp_path):
    older = HTML.replace("20261030/", "20260730/")
    current = HTML.replace("20260924539503.pdf", "20261030600000.pdf")
    actual = jp.financial_link(older + current, "8035.JP", "2026-11-05", disclosure=True)
    assert actual["published_date"] == "2026-10-30"


@pytest.mark.parametrize("wrong_security", [False, True])
def test_public_issuer_supplement_resumes_same_security_and_keeps_first_failure(tmp_path, monkeypatch, wrong_security):
    """官网补件可被同一通用任务消费；不请求旧失败目录，不批准资料或生成预测。"""
    monkeypatch.setattr(jp, "now", lambda: datetime.fromisoformat("2026-11-05T10:00:00+08:00"))
    output = tmp_path / "8035.JP" / "jp-financials"
    versioned_save(output / "latest.json", {"status": "ACTION_REQUIRED", "failure": "HTTP500", "source_attempts": []})
    raw_path = tmp_path / "source.pdf"
    raw_path.write_bytes(b"%PDF-native-original")
    selected = {"security_code": "8035.JP", "body": BODY, "published_date": "2026-10-30",
                "report_end": "2026-09-30", "raw_path": str(raw_path),
                "raw_sha256": jp.content_digest(raw_path.read_bytes()), "received_at": "2026-11-05T09:00:00+08:00"}
    versioned_save(output.parent / "primary-ir/financials.json", {
        "security_code": "8036.JP" if wrong_security else "8035.JP", "originals": [selected], "attempts": [],
    })
    if wrong_security:
        with pytest.raises(ValueError, match="JP_PRIMARY_FALLBACK_SECURITY_CONFLICT"):
            jp.collect("8035.JP", output)
    else:
        result = jp.collect("8035.JP", output)
        assert result["first_failure"] == "HTTP500" and result["status"] == "ORIGINAL_PARSED_READING_REQUIRED"
        assert result["received_at"] == selected["received_at"] and result["body"] == BODY
