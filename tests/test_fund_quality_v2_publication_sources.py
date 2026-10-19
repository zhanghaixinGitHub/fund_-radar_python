"""发布日期证明的反例；全部临时原文只测边界，不计入真实校准或来源核验。"""

import copy
import hashlib
import json
from datetime import date, datetime

import pytest
from app.services.fund_quality_v2_publication_sources import publication_proof_refs, verify_prospectus_publication

TITLE = "测试海外股票型证券投资基金（QDII）招募说明书（更新）"
AS_OF = date(2026, 9, 30)


@pytest.fixture
def original(tmp_path, monkeypatch):
    """目录与索引分别保留字节，解析替身只用于构造临时PDF版式反例。"""
    from app.services import fund_rating_public_market as public

    (tmp_path / "sources").mkdir()

    def save(name, raw, url):
        path = tmp_path / "sources" / name
        path.write_bytes(raw)
        return {"file": "sources/" + name, "sha256": hashlib.sha256(raw).hexdigest(), "url": url,
                "acquired_at": "2026-10-17T08:00:00+08:00"}

    row = {"id": 123, "title": TITLE, "publishDate": "20251209", "articleFile": "/notices/book.pdf"}
    catalog = {"code": "0000", "data": [row]}
    ref = {**save("book.pdf", b"%PDF-TEST_ONLY_BOOK", "https://www.fsfund.com/notices/book.pdf"),
           "label": "historical-prospectus", "title": TITLE, "published_on": "2025-12-09"}
    proof = {
        "article_id": 123,
        "catalog": save("catalog.json", json.dumps(catalog).encode(), "local-archive:issuer-catalog"),
        "notice_index": save("annual.pdf", b"%PDF-TEST_ONLY_INDEX", "https://www.fsfund.com/notices/annual.pdf"),
    }
    ref["publication_evidence"] = proof
    notice = {"text": "测试海外股票型证券投资基金（QDII）2025年年度报告 123456 "
              + TITLE + " 基金管理人网站 2025-12-09 其他公告基金管理人网站2026-01-01"}
    monkeypatch.setattr(public, "public_document", lambda *a: ([notice["text"]], None))

    def rewrite_catalog():
        raw = json.dumps(catalog).encode()
        (tmp_path / proof["catalog"]["file"]).write_bytes(raw)
        proof["catalog"]["sha256"] = hashlib.sha256(raw).hexdigest()

    return tmp_path, ref, catalog, notice, rewrite_catalog


def test_two_originals_bind_exact_title_url_and_first_row_date(original):
    root, ref, _, _, _ = original
    proof = verify_prospectus_publication(ref, root, "123456", AS_OF)
    assert proof["published_on"] == "2025-12-09" and proof["catalog_article_id"] == 123
    assert proof["source_hashes"] == [ref["publication_evidence"][k]["sha256"] for k in ("catalog", "notice_index")]
    assert "RETROSPECTIVE_PUBLICATION_METADATA_ONLY" in proof["scope"]
    assert "其他公告" not in proof["notice_excerpt"]


@pytest.mark.parametrize("mutation", ["row_url", "row_date", "title", "article_id", "duplicate_id", "business_error"])
def test_catalog_identity_conflicts_are_rejected(original, mutation):
    root, ref, catalog, _, rewrite = original
    if mutation == "row_url":
        catalog["data"][0]["articleFile"] = "/notices/other.pdf"
    elif mutation == "row_date":
        catalog["data"][0]["publishDate"] = "20251210"
    elif mutation == "title":
        catalog["data"][0]["title"] = TITLE.replace("海外", "境内")
    elif mutation == "article_id":
        ref["publication_evidence"]["article_id"] = 456
    elif mutation == "duplicate_id":
        catalog["data"].append(copy.deepcopy(catalog["data"][0]))
    else:
        catalog["code"] = "9999"
    rewrite()
    with pytest.raises(ValueError, match="PUBLICATION_"):
        verify_prospectus_publication(ref, root, "123456", AS_OF)


@pytest.mark.parametrize("new_date,allowed", [("20261015", True), ("20260901", False)])
def test_newer_observed_version_must_respect_historical_cutoff(original, new_date, allowed):
    root, ref, catalog, _, rewrite = original
    catalog["data"].append({**catalog["data"][0], "id": 456, "publishDate": new_date})
    rewrite()
    if allowed:
        assert verify_prospectus_publication(ref, root, "123456", AS_OF)["published_on"] == "2025-12-09"
    else:
        with pytest.raises(ValueError, match="NEWER_LISTED_VERSION"):
            verify_prospectus_publication(ref, root, "123456", AS_OF)


@pytest.mark.parametrize("bad", [True, False, {"verified": True}])
def test_boolean_or_success_marker_never_proves_publication(original, bad):
    _, ref, _, _, _ = original
    ref["publication_evidence"] = bad
    with pytest.raises(ValueError, match="PUBLICATION_EVIDENCE_SCHEMA"):
        publication_proof_refs([ref])


@pytest.mark.parametrize("change,reason", [
    ("next_notice_date", "NOTICE_ROW_MISSING"), ("other_product", "NOTICE_PRODUCT"),
    ("other_code", "NOTICE_PRODUCT"), ("not_annual", "NOTICE_PRODUCT"),
    ("tamper", "EVIDENCE_HASH"), ("escape", "EVIDENCE_PATH_SIZE"), ("other_host", "ISSUER_UNSUPPORTED"),
])
def test_index_original_identity_date_and_bytes_cannot_be_substituted(original, change, reason):
    root, ref, _, notice, _ = original
    evidence = ref["publication_evidence"]
    if change == "next_notice_date":
        notice["text"] = notice["text"].replace("2025-12-09 其他公告基金管理人网站2026-01-01",
                                                   "2025-12-08 其他公告基金管理人网站2025-12-09")
    elif change == "other_product":
        notice["text"] = notice["text"].replace("测试海外", "测试其他")
    elif change == "other_code":
        notice["text"] = notice["text"].replace("123456", "654321")
    elif change == "not_annual":
        notice["text"] = notice["text"].replace("年度报告", "中期报告")
    elif change == "tamper":
        (root / evidence["notice_index"]["file"]).write_bytes(b"%PDF-CHANGED")
    elif change == "escape":
        evidence["notice_index"]["file"] = "../outside.pdf"
    else:
        evidence["notice_index"]["url"] = "https://example.org/annual.pdf"
    with pytest.raises(ValueError, match=reason):
        verify_prospectus_publication(ref, root, "123456", AS_OF)


def test_proof_sources_are_frozen_and_same_pdf_is_not_double_counted(original):
    from app.services.fund_quality_v2_materials import inspect_sources

    root, ref, _, _, _ = original
    refs = [ref, {**ref["publication_evidence"]["notice_index"], "label": "historical-annual"}]
    inspected = inspect_sources(refs, root, datetime.fromisoformat("2026-10-17T09:00:00+08:00"))
    assert len(inspected) == 3 and all(r["state"] == "ORIGINAL_AVAILABLE_PARSE_REQUIRED" for r in inspected)
    assert {r["label"] for r in inspected} == {"historical-prospectus", "historical-annual", "publisher-catalog"}
    ref["publication_evidence"]["catalog"]["acquired_at"] = "2026-10-18T09:00:00+08:00"
    assert next(r for r in inspect_sources([ref], root, datetime.fromisoformat("2026-10-17T09:00:00+08:00"))
                if r["label"] == "publisher-catalog")["reason"] == "SOURCE_TIME"


def test_fallback_uses_latest_only_and_preserves_primary(original, monkeypatch):
    from app.services import fund_quality_v2_disclosures as disclosures
    from app.services import fund_quality_v2_publication_sources as sources

    root, latest, _, _, _ = original
    monkeypatch.setattr(disclosures, "public_document", lambda *a: (["测试123456招募说明书"], None))
    calls = []

    def verify(ref, *args):
        calls.append(ref["sha256"])
        raise ValueError("TEST_LATEST_INVALID_NOT_OLDER_FALLBACK")

    monkeypatch.setattr(sources, "verify_prospectus_publication", verify)
    older = {**latest, "published_on": "2025-02-13", "sha256": "OLD_TEST_ONLY"}
    result = disclosures.source_documents("123456", [older, latest], root, AS_OF)
    assert result["prospectus"]["error"] == "TEST_LATEST_INVALID_NOT_OLDER_FALLBACK"
    assert calls == [latest["sha256"]]
    primary = {k: v for k, v in older.items() if k != "publication_evidence"}
    primary["label"] = "prospectus"
    assert "error" not in disclosures.source_documents("123456", [primary, latest], root, AS_OF)["prospectus"]
    assert calls == [latest["sha256"]]
    conflicting = {**latest, "sha256": "SAME_DAY_DIFFERENT_BYTES"}
    assert disclosures.source_documents("123456", [latest, conflicting], root, AS_OF)["prospectus"]["error"] \
        == "PROSPECTUS_LATEST_PUBLICATION_CONFLICT"
