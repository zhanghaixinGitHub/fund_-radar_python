"""同月费用续办仍须先规范导入原件，不放宽严格评分来源边界。"""

from copy import deepcopy

import pytest
from app.services import fund_rating_fee_inputs as fees
from app.services import fund_rating_pipeline_prepare as prepare
from app.services import fund_rating_reviewed_snapshot as snapshot


@pytest.fixture
def resumed(monkeypatch):
    raw = dict(fund_code="017437", as_of_date="2026-10-30", share_class="C",
               legal_name="自身QDII基金", primary_ref="original",
               sources={"original": {"sha256": "original", "file": "news/original.json"}})
    bundle = dict(basis=prepare.BASIS, fund_code="017437", share_class="C",
                  identity={"legal_name": "自身QDII基金"},
                  original_report_archive={"sha256": "original"},
                  reviewed_document={"dimensions": {"fee": None, "return": None}})
    monkeypatch.setattr(prepare, "current_inputs", lambda *a: {
        "017437": {"as_of_date": "2026-10-30", "input_snapshot": {
            "members": [{"fund_code": "017437", "bundle": bundle}]}}})
    monkeypatch.setattr(prepare, "archive_json", lambda *a: {"sha256": "compiled"})
    monkeypatch.setattr(snapshot, "validate_sources", lambda *a, **k: None)
    monkeypatch.setattr(snapshot, "calculate", lambda *a, **k: {"status": "PARTIAL", "score": None})
    request = dict(fund_code="017437", as_of_date="2026-10-30", inputs=raw,
                   explanation="自身C份额本期原件独立费用重算；其他七维仍未完成，保持空值且不按覆盖维度折算完整总分。")
    return request, bundle


def test_existing_fee_review_is_canonicalized_before_strict_calculation(monkeypatch, tmp_path, resumed):
    request, original = resumed
    imported, calculated = [], []

    def canonical(root, ref):
        imported.append(deepcopy(ref))
        return {**ref, "file": "sources/pipeline/original.json"}

    def calculate(raw, **kwargs):
        assert imported and raw["sources"]["original"]["file"] == "sources/pipeline/original.json"
        calculated.append(raw)
        return {"score": "54.5", "published_on": "2025-12-09"}

    monkeypatch.setattr(prepare, "canonical_source", canonical)
    monkeypatch.setattr(fees, "calculate", calculate)
    result = prepare.prepare_fees(None, request, root=tmp_path)
    assert len(imported) == len(calculated) == 1
    assert result["entries"][0]["fee_calculation"]["inputs"]["sources"]["original"]["file"].startswith("sources/")
    assert original["reviewed_document"]["dimensions"]["return"] is None
    assert request["inputs"]["sources"]["original"]["file"] == "news/original.json"


@pytest.mark.parametrize("failure", ["RATING_R5_IMPORT_SOURCE_HASH", "RATING_R5_IMPORT_SOURCE_SCOPE"])
def test_invalid_original_still_stops_fee_resume_before_calculation(monkeypatch, tmp_path, resumed, failure):
    request, _ = resumed
    calls = []

    def reject(*args):
        raise ValueError(failure)

    monkeypatch.setattr(prepare, "canonical_source", reject)
    monkeypatch.setattr(fees, "calculate", lambda *a, **k: calls.append(1))
    with pytest.raises(ValueError, match=failure):
        prepare.prepare_fees(None, request, root=tmp_path)
    assert not calls
