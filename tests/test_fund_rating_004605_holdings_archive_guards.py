"""持仓组件的精确大原件例外不能扩大为通用文件或未封存输入的旁路。"""

import hashlib
import json

import pytest
from app.services.fund_rating_004605_asset_quality import LARGE_VERIFIED_SOURCE, verify_registered_original
from app.services.fund_rating_004605_holdings import FIELD, validate_input_archive
from app.services.fund_rating_reviewed_snapshot import _read


def test_large_pdf_exception_requires_the_exact_real_size(tmp_path):
    """即使复用例外摘要标识，截短文件也必须在读出评分之前拒绝。"""
    source = tmp_path / "sources" / "short.pdf"
    source.parent.mkdir()
    source.write_bytes(b"%PDF-1.4\n")
    reference = {"file": "sources/short.pdf", "sha256": LARGE_VERIFIED_SOURCE, "kind": "pdf"}
    inputs = {"sources": {LARGE_VERIFIED_SOURCE: {**reference, "archive_kind": "ORIGINAL"}}}
    with pytest.raises(ValueError, match="EXACT_LARGE_ORIGINAL"):
        verify_registered_original(inputs, reference, root=tmp_path)


def test_registered_json_cannot_enter_the_original_pdf_exception(tmp_path):
    """派生JSON即使已登记，也不能借PDF专用钩子扩大读取范围。"""
    reference = {"file": "sources/review.json", "sha256": "a" * 64, "kind": "json"}
    inputs = {"sources": {reference["sha256"]: {**reference, "archive_kind": "ORIGINAL"}}}
    with pytest.raises(ValueError, match="REGISTERED_PDF_ONLY"):
        verify_registered_original(inputs, reference, root=tmp_path)


def test_unregistered_reference_cannot_use_the_component_exception(tmp_path):
    reference = {"file": "sources/unregistered.pdf", "sha256": "b" * 64, "kind": "pdf"}
    with pytest.raises(ValueError, match="REGISTERED_ORIGINAL_REF"):
        verify_registered_original({"sources": {}}, reference, root=tmp_path)


def test_embedded_inputs_must_equal_the_actual_archived_json(tmp_path):
    """实际读取并验摘要后的封存JSON与内嵌输入不一致，不能继续按候选分验收。"""
    content = json.dumps({"fund_code": "004605"}).encode()
    source = tmp_path / "sources" / "inputs.json"
    source.parent.mkdir()
    source.write_bytes(content)
    reference = {"file": "sources/inputs.json", "sha256": hashlib.sha256(content).hexdigest(), "kind": "json"}
    bundle = {FIELD: {"inputs": {"fund_code": "000009"}, "input_archive": reference}}
    with pytest.raises(ValueError, match="INPUT_ARCHIVE"):
        validate_input_archive(bundle, root=tmp_path, archive_reader=_read)
