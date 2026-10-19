"""定期原件准入及优先级的合成反例；不作真实来源证明或校准样本。"""

import hashlib
from datetime import date

import pytest
from app.services.fund_quality_v2_periodic_sources import select_periodic_document, verify_cninfo_periodic


def document(sha, authority, *, period="2025-12-31", published="2026-03-31"):
    return {"economic_period": ["2025-01-01", period], "source_authority": authority,
            "ref": {"sha256": sha, "published_on": published, "url": "https://example.test/"+sha}}


def test_issuer_original_wins_even_when_reprint_hash_sorts_later():
    issuer, reprint = document("000", "ISSUER_ORIGINAL"), document("fff", "FULL_PUBLIC_REPRINT")
    for inputs in [[issuer, reprint], [reprint, issuer]]:
        selected, evidence = select_periodic_document(inputs)
        assert selected is issuer and evidence["selected_sha256"] == "000"
        assert not evidence["whole_document_equivalence_proven"]
        assert len(evidence["counterparts"]) == 2


def test_same_priority_different_bytes_rejects_instead_of_selecting_hash_or_old_period():
    older = document("old", "ISSUER_ORIGINAL", period="2025-06-30", published="2025-08-31")
    selected, evidence = select_periodic_document([older, document("000", "ISSUER_ORIGINAL"),
                                                   document("fff", "ISSUER_ORIGINAL")])
    assert selected is None and evidence["error"].endswith("CONFLICT")
    assert evidence["conflicting_sha256"] == ["000", "fff"]


def test_same_original_duplicates_do_not_create_source_conflict():
    original = document("same", "ISSUER_ORIGINAL")
    selected, evidence = select_periodic_document([original, dict(original)])
    assert selected is original and evidence["selected_sha256"] == "same"


def test_latest_economic_period_and_publication_precede_source_priority():
    old = document("old", "ISSUER_ORIGINAL", period="2025-06-30", published="2025-08-31")
    newer = document("new", "FULL_PUBLIC_REPRINT")
    assert select_periodic_document([old, newer])[0] is newer
    corrected = document("correction", "REGULATORY_DISCLOSURE_PDF", published="2026-04-01")
    assert select_periodic_document([newer, corrected])[0] is corrected


def test_regulatory_source_is_preferred_over_provider_pdf_and_public_reprint():
    regulatory = document("a", "REGULATORY_DISCLOSURE_PDF")
    selected, _ = select_periodic_document([document("z", "BOUND_PROVIDER_PDF"), regulatory,
                                           document("zz", "FULL_PUBLIC_REPRINT")])
    assert selected is regulatory


def synthetic_cninfo(tmp_path):
    """只构造字段与原句的单元测试输入，PDF解码在真实原件回放中另核。"""
    folder = tmp_path / "sources"
    folder.mkdir()
    raw = b"%PDF-UNIT-TEST-NOT-A-REAL-REPORT"
    sha = hashlib.sha256(raw).hexdigest()
    url = "https://static.cninfo.com.cn/finalpage/2026-03-31/1234567890.PDF"
    received = "2026-10-17T14:00:00+08:00"
    receipt = {"url": url, "status": 200, "bytes": len(raw), "sha256": sha, "received_at": received}
    proof = folder / "receipt.json"
    from app.services.fund_materials_store import versioned_save

    versioned_save(proof, receipt)
    proof_raw = proof.read_bytes()
    proof_ref = {"label": "cninfo-download-receipt", "sha256": hashlib.sha256(proof_raw).hexdigest(),
                 "file": "sources/receipt.json", "url": url, "acquired_at": received,
                 "published_on": "2026-03-31"}
    ref = {"label": "historical-annual", "url": url, "sha256": sha, "published_on": "2026-03-31",
           "acquired_at": received, "cninfo_receipt_sha256": proof_ref["sha256"]}
    pages = ["测试股票型证券投资基金2025年年度报告\n送出日期：2026年03月31日\n"
             "基金名称 测试股票型证券投资基金 基金简称 测试基金 基金主代码 123456\n"
             "下属分级基金的交易代码 123456 123457 报告期末下属份额数量 100.00"]
    return raw, pages, ref, proof_ref


def test_cninfo_rebuilds_actual_receipt_and_explicit_master_share_table(tmp_path):
    raw, pages, ref, proof = synthetic_cninfo(tmp_path)
    result = verify_cninfo_periodic(raw, pages, ref, refs=[ref, proof], root=tmp_path,
                                    code="123457", as_of=date(2026, 5, 1))
    assert result["master_code"] == "123456" and result["share_codes"] == ["123456", "123457"]
    assert result["source_hashes"] == [ref["sha256"], proof["sha256"]]


@pytest.mark.parametrize("mutation", ["http", "fake_host", "wrong_role", "query", "path_date", "not_pdf",
    "raw_hash", "missing_receipt", "receipt_changed", "receipt_binding", "cover_date", "wrong_cover_name",
    "code_only_in_holdings", "fake_verified_flag", "future_publication", "receipt_ref_url", "receipt_ref_time"])
def test_cninfo_flags_route_or_incidental_code_never_replace_actual_source_binding(tmp_path, mutation):
    raw, pages, ref, proof = synthetic_cninfo(tmp_path)
    refs, code, as_of = [ref, proof], "123457", date(2026, 5, 1)
    if mutation == "http":
        ref["url"] = ref["url"].replace("https:", "http:")
    elif mutation == "fake_host":
        ref["url"] = ref["url"].replace(".com.cn", ".com.cn.attacker.test")
    elif mutation == "wrong_role":
        ref["label"] = "summary"
    elif mutation == "query":
        ref["url"] += "?other=1"
    elif mutation == "path_date":
        ref["url"] = ref["url"].replace("2026-03-31", "2026-03-30")
    elif mutation == "not_pdf":
        raw = b"<html>access challenge</html>"
    elif mutation == "raw_hash":
        ref["sha256"] = "0"*64
    elif mutation in {"missing_receipt", "fake_verified_flag"}:
        refs = [ref]
        ref["verified"] = True
    elif mutation == "receipt_changed":
        (tmp_path/proof["file"]).write_text("{}", encoding="utf-8")
    elif mutation == "receipt_binding":
        ref["acquired_at"] = "2026-10-17T13:59:59+08:00"
    elif mutation == "cover_date":
        pages[0] = pages[0].replace("送出日期：2026年03月31日", "送出日期：2026年03月30日")
    elif mutation == "wrong_cover_name":
        pages[0] = pages[0].replace("测试股票型证券投资基金2025", "其他股票型证券投资基金2025")
    elif mutation == "code_only_in_holdings":
        code = "654321"
        pages[0] += "\n持仓证券代码654321"
    elif mutation == "future_publication":
        as_of = date(2026, 3, 30)
    elif mutation == "receipt_ref_url":
        proof["url"] = "https://unrelated.test/receipt"
    elif mutation == "receipt_ref_time":
        proof["acquired_at"] = "2026-10-17T13:00:00+08:00"
    with pytest.raises(ValueError):
        verify_cninfo_periodic(raw, pages, ref, refs=refs, root=tmp_path, code=code, as_of=as_of)
