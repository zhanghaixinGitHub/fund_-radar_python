"""验证这次真实故障与时间边界：必须打开正文文件，不能借用未来持仓或评估期词频。"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from scipy import sparse

from scripts import fund_002112_recent_bodies_train_v1 as train
from scripts import fund_002112_recent_bodies_v1 as source


def test_body_unicode_paragraph_mark_does_not_split_json_record(tmp_path):
    path = tmp_path / "documents.jsonl"
    expected = [{"body": "正文第一段\u2028正文第二段\u0085原文分隔"}, {"body": "下一篇"}]
    source.io.save_lines(path, expected)
    assert source.read_lines(path) == expected


@pytest.fixture
def material(tmp_path, monkeypatch):
    entry = {"id": "company-12345", "kind": "company", "title": "公司业绩预告",
             "stockCode": "002463.SZ", "publishedDate": "2025-01-01", "textComplete": False,
             "sourceUrl": "https://example.invalid/12345.PDF"}
    raw = tmp_path / "raw/test.pdf"
    raw.parent.mkdir()
    raw.write_bytes(b"fixture-pdf")
    value = {"announcement_id": "12345", "stock_code": "002463", "title": entry["title"],
             "announced_at_source": "2025-01-01T18:00:00+08:00", "text_status": "TEXT_EXTRACTED",
             "pages": ["营业收入增长，公司完成新产品研发。" * 20], "training_eligible": False,
             "receipt": {"file": "raw/test.pdf", "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                         "url": entry["sourceUrl"], "received_at": "2026-09-30T10:00:00+08:00"}}
    path = tmp_path / source.body_relative(entry)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    class FakePDF:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_metadata_dict(self):
            return {"CreationDate": "D:20250101120000", "ModDate": "D:20250101120000"}

    monkeypatch.setattr(source.pdfium, "PdfDocument", FakePDF)
    return tmp_path, entry, path, value, FakePDF


def test_index_expands_actual_body_despite_old_training_flag(material):
    root, entry, _, value, _ = material
    doc = source.resolve_document(entry, source.Snapshot(root / "snapshot"), root)
    assert doc["body"] == value["pages"][0]
    assert doc["body_available_at"] == "2025-01-02T00:00:00+08:00"
    assert doc["body_exclusions"] == []
    assert doc["old_training_eligible"] is False
    assert doc["raw_hash_verified"] is True


@pytest.mark.parametrize("field,value,reason", [
    ("announcement_id", "wrong", "ANNOUNCEMENT_ID_MISMATCH"),
    ("stock_code", "wrong", "STOCK_ID_MISMATCH"),
    ("text_status", "PARTIAL_TEXT_CHECK_IMAGES", "INCOMPLETE_TEXT_EXTRACTION"),
])
def test_identity_and_partial_body_excluded(material, field, value, reason):
    root, entry, path, saved, _ = material
    saved[field] = value
    path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    doc = source.resolve_document(entry, source.Snapshot(root / "snapshot"), root)
    assert doc["body"] == "" and reason in doc["body_exclusions"]
    assert doc["title"] == entry["title"]


def test_raw_hash_mismatch_is_not_silent_title_upgrade(material):
    root, entry, _, _, _ = material
    (root / "raw/test.pdf").write_bytes(b"changed-pdf")
    doc = source.resolve_document(entry, source.Snapshot(root / "snapshot"), root)
    assert not doc["body"]
    assert "RAW_FILE_MISSING_OR_HASH_MISMATCH" in doc["body_exclusions"]


def test_known_later_pdf_delays_only_body(material, monkeypatch):
    root, entry, _, _, fake = material
    monkeypatch.setattr(fake, "get_metadata_dict", lambda self: {"ModDate": "D:20250201100000"})
    doc = source.resolve_document(entry, source.Snapshot(root / "snapshot"), root)
    assert doc["title_available_at"] == "2025-01-02T00:00:00+08:00"
    assert doc["body_available_at"] == "2025-02-02T00:00:00+08:00"


def report(at, weight):
    return {"fund_code": "002112", "fund_master_code": "001412", "report_end": "2024-12-31",
            "available_at": at, "raw": {"sha256": at},
            "holdings": [{"stock_code": "002463.SZ", "nav_weight_pct": weight}]}


def test_later_holding_report_cannot_backfill_past_company_link():
    past = report("2025-01-01T00:00:00+08:00", 2)
    future = report("2025-02-01T00:00:00+08:00", 9)
    event = {"kind": "company", "stock_code": "002463.SZ", "title_available_at": "2025-01-02T00:00:00+08:00"}
    cutoff = "2025-01-03T08:00:00+08:00"
    assert source.choose_report([future], cutoff) is None
    assert source.event_weight(event, cutoff, past, [past, future]) == 0.02
    assert source.event_weight(event, "2025-01-01T08:00:00+08:00", past, [past]) == 0
    assert source.event_weight(event, cutoff, future, [future]) == 0


def test_text_idf_and_median_never_fit_evaluation_rows():
    rows = [{"numeric": [1, None], "counts": [0]}, {"numeric": [3, 2], "counts": [1]},
            {"numeric": [99999, 99999], "counts": [999]}]
    matrices = {"title": sparse.csr_matrix([[1, 0, 0], [1, 1, 0], [0, 0, 99999]])}
    _, fitted = train.transform(rows, matrices, [0, 1], "LR_H_TITLE")
    np.testing.assert_array_equal(fitted["imputer"].statistics_, [2, 2, 0.5])
    before = fitted["idf"]["title"].idf_.copy()
    train.transform(rows, matrices, [2], "LR_H_TITLE", fitted)
    np.testing.assert_array_equal(fitted["idf"]["title"].idf_, before)
    assert before[2] > before[0]


def test_full_body_changes_inputs_when_title_and_count_unchanged():
    rows = [{"numeric": [1], "counts": [1]}, {"numeric": [1], "counts": [1]}]
    matrices = {"title": sparse.csr_matrix([[1, 0], [1, 0]]),
                "body": sparse.csr_matrix([[1, 0], [0, 1]])}
    titled, _ = train.transform(rows, matrices, [0, 1], "LR_H_TITLE")
    body, _ = train.transform(rows, matrices, [0, 1], "LR_H_BODY")
    np.testing.assert_array_equal(titled[0].toarray(), titled[1].toarray())
    assert (body[0] != body[1]).nnz > 0
