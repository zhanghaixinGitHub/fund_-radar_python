"""研究目录复用只接受真实字节与收据；以下临时算例绝不能用于真实校准。"""

import hashlib
import json
from datetime import date, datetime

import pytest
from app.services.fund_quality_v2_source_reuse import reuse_originals


@pytest.mark.parametrize("mutation", [None, "missing", "changed", "outside_root"])
def test_cninfo_reuse_keeps_actual_receipt_bytes_and_rejects_only_verified_flag(tmp_path, mutation):
    """这里只核档案复制；合成PDF不能成为真实披露准入或基金校准依据。"""
    from app.services.fund_materials_store import versioned_save

    source, target = tmp_path/"archive", tmp_path/"rating"
    (source/"raw").mkdir(parents=True)
    (source/"reports/123456").mkdir(parents=True)
    raw = b"%PDF-TEST-ONLY"
    pdf_hash = hashlib.sha256(raw).hexdigest()
    (source/"raw/report.pdf").write_bytes(raw)
    url = "https://static.cninfo.com.cn/finalpage/2026-03-31/1234567890.PDF"
    receipt = {"file": "raw/report.pdf", "url": url, "sha256": pdf_hash,
               "received_at": "2026-10-17T12:00:00+08:00", "status": 200, "bytes": len(raw)}
    proof_path = source/"raw/download.json"
    versioned_save(proof_path, receipt)
    proof_raw = proof_path.read_bytes()
    proof_hash = hashlib.sha256(proof_raw).hexdigest()
    item = {"fund_code": "123456", "title": "测试基金2025年年度报告", "endDate": "2025-12-31",
            "published_date": "2026-03-31", "receipt": receipt,
            "cninfo_download_receipt": {"file": "raw/download.json", "sha256": proof_hash}}
    if mutation == "missing":
        del item["cninfo_download_receipt"]
        item["verified"] = True
    elif mutation == "changed":
        proof_path.write_bytes(proof_raw+b" ")
    elif mutation == "outside_root":
        item["cninfo_download_receipt"]["file"] = "../outside.json"
    (source/"reports/123456/report.json").write_text(json.dumps(item), encoding="utf-8")
    args = ("123456", "测试-A", source, target, date(2026, 5, 1),
            datetime.fromisoformat("2026-10-17T13:00:00+08:00"))
    if mutation is not None:
        with pytest.raises(ValueError, match="CNINFO_DOWNLOAD_RECEIPT"):
            reuse_originals(*args)
    else:
        refs = reuse_originals(*args)
        original = next(r for r in refs if r["label"] == "historical-annual")
        proof = next(r for r in refs if r["label"] == "cninfo-download-receipt")
        assert original["cninfo_receipt_sha256"] == proof_hash
        assert (target/proof["file"]).read_bytes() == proof_raw


def test_annual_keeps_history_attachment_role_while_primary_selection_is_downstream(tmp_path):
    """同目录只有年报时也不能改变附件角色，避免旧完整包筛选遗失该原件。"""
    source, target = tmp_path / "archive", tmp_path / "ratings"
    (source / "raw").mkdir(parents=True)
    (source / "reports/123456").mkdir(parents=True)
    raw = b"%PDF-test-only-not-a-calibration-document"
    (source / "raw/report.pdf").write_bytes(raw)
    item = {
        "fund_code": "123456", "title": "测试基金2025年年度报告", "endDate": "2025-12-31",
        "published_date": "2026-03-31",
        "receipt": {
            "file": "raw/report.pdf", "url": "https://www.cmfchina.com/report.pdf",
            "received_at": "2026-10-17T00:00:00+08:00", "sha256": hashlib.sha256(raw).hexdigest(),
        },
    }
    (source / "reports/123456/report.json").write_text(json.dumps(item), encoding="utf-8")
    refs = reuse_originals(
        "123456", "测试-A", source, target, date(2026, 4, 30),
        datetime.fromisoformat("2026-10-17T01:00:00+08:00"),
    )
    assert len(refs) == 1 and refs[0]["label"] == "historical-annual"
    assert refs[0]["report_end"] == "2025-12-31"


def test_cash_snapshot_keeps_normalized_provenance_and_future_notice_is_not_admitted(tmp_path):
    from app.services.fund_quality_v2_source_reuse import reuse_cash_originals

    source, target = tmp_path / "cash", tmp_path / "rating"
    (source / "raw").mkdir(parents=True)
    raw = json.dumps({"fund_code": "006730", "dividend_records": []}).encode()
    (source / "raw/events.json").write_bytes(raw)
    manifest = {"schema": "QUALITY_CASH_ORIGINALS_V1", "items": [{
        "fund_code": "006730", "label": "cash-event-snapshot", "title": "测试规范记录非校准",
        "receipt": {"file": "raw/events.json", "url": "https://tushare.pro/document/2?doc_id=119",
                    "received_at": "2026-10-17T00:00:00+08:00", "sha256": hashlib.sha256(raw).hexdigest()},
    }, {"fund_code": "006730", "label": "cash-dividend-notice", "published_on": "2026-01-08"}]}
    (source / "quality-cash-originals.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = reuse_cash_originals("006730", source, target, datetime.fromisoformat("2026-10-17T01:00:00+08:00"),
                                  date(2025, 4, 30))
    assert len(result) == 1 and result[0]["source_kind"] == "NORMALIZED_PROVIDER_EVENT_SNAPSHOT_NOT_RAW_HTTP_RESPONSE"
    (source / "raw/events.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="BYTES_CHANGED"):
        reuse_cash_originals("006730", source, target, datetime.fromisoformat("2026-10-17T01:00:00+08:00"),
                             date(2025, 4, 30))


def test_reprint_rebuilt_from_raw_response_not_metadata_body_and_hash_tamper_rejected(tmp_path):
    source, target = tmp_path / "research", tmp_path / "ratings"
    (source / "raw").mkdir(parents=True)
    (source / "fund-documents" / "123456").mkdir(parents=True)
    original = {
        "success": 1,
        "data": {
            "art_code": "TEST_ONLY",
            "security": [{"stock": "123456"}],
            "notice_title": "测试基金C类份额产品资料概要",
            "notice_content": "测试原文，不是基金材料。" * 200,
        },
    }
    raw = json.dumps(original, ensure_ascii=False).encode("utf-8")
    path = source / "raw" / "original.json"
    path.write_bytes(raw)
    metadata = {
        "fund_code": "123456",
        "title": original["data"]["notice_title"],
        "published_date": "2026-08-31",
        "body": "不可作为来源证明",
        "receipt": {
            "file": "raw/original.json",
            "url": "https://np-cnotice-fund.eastmoney.com/api/content/ann",
            "received_at": "2026-10-01T00:00:00+08:00",
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    }
    (source / "fund-documents" / "123456" / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    frozen = datetime.fromisoformat("2026-10-16T00:00:00+08:00")
    refs = reuse_originals("123456", "测试-C", source, target, date(2026, 9, 30), frozen)
    assert len(refs) == 1 and refs[0]["article_id"] == "TEST_ONLY"
    assert (target / refs[0]["file"]).read_bytes() == raw
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="ADDITIONAL_SOURCE_HASH"):
        reuse_originals("123456", "测试-C", source, target, date(2026, 9, 30), frozen)


def test_explicit_manager_receipt_rechecks_bytes_and_product_not_success_boolean(tmp_path):
    source, target = tmp_path / "research", tmp_path / "ratings"
    (source / "raw").mkdir(parents=True)
    data = {
        "code": 0,
        "data": {
            "fields": ["ts_code", "ann_date", "name", "begin_date", "end_date"],
            "items": [["123456.OF", "20230101", "测试甲", "20230101", None]],
        },
    }
    raw = json.dumps(data).encode()
    path = source / "raw" / "appointment.json"
    path.write_bytes(raw)
    receipt = {
        "file": "raw/appointment.json",
        "url": "https://api.tushare.pro",
        "received_at": "2026-10-16T00:00:00+08:00",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    manifest = {
        "schema": "QUALITY_SOURCE_RECEIPTS_V1",
        "items": [
            {
                "fund_code": "123456",
                "api_name": "fund_manager",
                "ts_code": "123456.OF",
                "label": "manager-api",
                "receipt": receipt,
                "verified": True,
            }
        ],
    }
    manifest_path = source / "quality-originals.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    frozen = datetime.fromisoformat("2026-10-16T01:00:00+08:00")
    refs = reuse_originals("123456", "测试-C", source, target, date(2026, 9, 30), frozen)
    assert len(refs) == 1 and refs[0]["label"] == "manager-api"
    assert (target / refs[0]["file"]).read_bytes() == raw
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="SUPPLEMENTAL_ORIGINAL_HASH"):
        reuse_originals("123456", "测试-C", source, target, date(2026, 9, 30), frozen)


def test_c_share_alias_reads_own_summary_directory_and_does_not_take_a_share_fees(tmp_path):
    source, target = tmp_path / "research", tmp_path / "ratings"
    (source / "raw").mkdir(parents=True)
    (source / "fund-documents/012836").mkdir(parents=True)
    (source / "fund-documents/012835").mkdir(parents=True)
    original = {
        "success": 1,
        "data": {
            "art_code": "TEST_ONLY",
            "security": [{"stock": "012836"}],
            "notice_title": "测试基金C类份额产品资料概要",
            "notice_content": "测试原文，不能作为校准来源。" * 200,
        },
    }
    raw = json.dumps(original, ensure_ascii=False).encode("utf-8")
    (source / "raw/c.json").write_bytes(raw)
    receipt = {
        "file": "raw/c.json",
        "url": "https://np-cnotice-fund.eastmoney.com/api/content/ann",
        "received_at": "2026-10-16T00:00:00+08:00",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    item = {
        "fund_code": "012836",
        "title": original["data"]["notice_title"],
        "published_date": "2026-07-17",
        "receipt": receipt,
    }
    (source / "fund-documents/012836/c.json").write_text(json.dumps(item), encoding="utf-8")
    (source / "fund-documents/012835/a.json").write_text(
        json.dumps({**item, "fund_code": "012835", "title": "测试基金A类份额产品资料概要"}), encoding="utf-8"
    )
    refs = reuse_originals(
        "012836", "测试-C", source, target, date(2026, 9, 30), datetime.fromisoformat("2026-10-17T00:00:00+08:00")
    )
    assert len(refs) == 1 and refs[0]["label"] == "summary" and "C类份额" in refs[0]["title"]
