"""当前公告专项证据必须绑定同一原件和目录，不能宽松匹配后误报成功。"""

import hashlib
import json
from copy import deepcopy

import pytest
from app.services import company_publication_review as module
from app.services.company_news_facts import inspect_company
from tests.test_company_news_facts import AT, inputs


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    item, doc, holding, report = deepcopy(inputs())
    pages = ["华工科技产业股份有限公司限制性股票激励计划草案尚待审议。" * 10]
    source = {"id": "123", "title": item["title_plain"], "stock_code": "000988",
              "published_at": item["announced_at_source"], "receipt": doc["receipt"], "normalized_pages": pages}
    source_raw = json.dumps(source, ensure_ascii=False).encode()
    audit = {
        "id": "123", "at": AT.isoformat(), "title": source["title"], "stock_code": "000988",
        "source_sha256": hashlib.sha256(source_raw).hexdigest(), "raw_sha256": doc["receipt"]["sha256"],
        "catalog_rows": [deepcopy(item)], "source_announced_at": source["published_at"],
        "source_publication_verified": True, "current_document_nature_verified": True,
        "current_publication_gaps": [], "version_issues": [], "historical_training_eligible": False,
        "numeric_table_semantics_verified": False, "document_kind": "CONDITIONAL_GOVERNANCE_DRAFT",
        "business_boundary": "保留草案条件，尚待审议。",
        "identity_anchors": [{"page": 1, "offset": 0, "literal": "华工科技产业股份有限公司"}],
        "kind_anchors": [{"page": 1, "offset": 13, "literal": "限制性股票激励计划草案"}],
    }
    audit["kind_anchors"][0]["offset"] = pages[0].index("限制性")
    for name in ("current", "current-audit-v2"):
        (tmp_path / name).mkdir()
    (tmp_path / "current/123.json").write_bytes(source_raw)
    audit_path = tmp_path / "current-audit-v2/123.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(module, "REVIEW_ROOT", tmp_path)
    monkeypatch.setattr(module, "verified_bytes", lambda receipt: b"verified original")
    return item, doc, holding, report, audit, audit_path


def test_scanned_document_can_reuse_exact_review_without_claiming_financial_or_training_facts(evidence):
    item, doc, holding, report, *_ = evidence
    doc.update(pages=[""], text_status="SCANNED")
    fact = inspect_company(item, doc, holding, report, AT)
    assert fact["business_eligible"] and fact["summary"] == "保留草案条件，尚待审议。"
    assert not fact["prediction_eligible"] and not fact["training_eligible"] and fact["effective_at"] is None
    assert fact["evidence"]["verified_publication_review"]["numeric_table_semantics_verified"] is False


@pytest.mark.parametrize("fault", ["source", "raw", "catalog", "identity", "anchor", "future", "semantics"])
def test_review_rejects_changed_identity_version_anchors_or_unproved_semantics(evidence, fault):
    item, doc, _holding, _report, audit, audit_path = evidence
    if fault == "source":
        audit["source_sha256"] = "0" * 64
    elif fault == "raw":
        doc["receipt"]["sha256"] = "0" * 64
    elif fault == "catalog":
        item["announcementTime"] += 1
    elif fault == "identity":
        audit["stock_code"] = "999999"
    elif fault == "anchor":
        audit["identity_anchors"][0]["offset"] = 1
    elif fault == "future":
        audit["at"] = "2026-10-01T00:00:00+08:00"
    else:
        audit["numeric_table_semantics_verified"] = True
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError):
        module.reviewed_publication(item, doc, AT)


def test_review_never_bypasses_catalog_version_check(evidence, monkeypatch):
    item, doc, holding, report, *_ = evidence
    doc.update(pages=[""], text_status="SCANNED", catalog_hash="changed")
    monkeypatch.setattr(module, "reviewed_publication", lambda *_: pytest.fail("catalog failure must not reuse review"))
    with pytest.raises(ValueError, match="CATALOG_VERSION_MISMATCH"):
        inspect_company(item, doc, holding, report, AT)
