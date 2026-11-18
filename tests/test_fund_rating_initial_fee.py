import json
from types import SimpleNamespace

import pytest
from app.services import fund_rating_pipeline_prepare as prepare
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS


@pytest.fixture
def initial(monkeypatch, tmp_path):
    legal = "华宝纳斯达克精选股票型发起式证券投资基金(QDII)"
    content = {
        "success": 1,
        "data": {
            "notice_content": legal + "2026年9月30日 华宝基金管理有限公司 017436 A 017437 C",
            "notice_title": legal + "2026年第3季度报告",
            "notice_date": "2026-10-27 00:00:00",
            "security": [{"stock": "017437"}],
        },
    }
    folder = tmp_path / "sources"
    folder.mkdir()
    (folder / "q3.json").write_text(json.dumps(content), encoding="utf-8")
    monkeypatch.setattr(prepare, "canonical_source", lambda root, ref: ref)
    master = {
        "public_name": legal + "C",
        "share_class": "C",
        "fund_master_id": 1,
        "catalog_legal_name": legal,
        "manager_name": "华宝基金管理有限公司",
    }
    session = SimpleNamespace(execute=lambda *a: SimpleNamespace(mappings=lambda: SimpleNamespace(one=lambda: master)))
    raw = {
        "fund_code": "017437",
        "basis": ACTIVE_BASIS,
        "as_of_date": "2026-10-30",
        "share_class": "C",
        "legal_name": legal,
        "primary_ref": "h",
        "sources": {"h": {"sha256": "h", "file": "sources/q3.json", "published_on": "2026-10-27", "url": "own"}},
    }
    return (
        session,
        {"fund_code": "017437", "as_of_date": "2026-10-30", "initial_partial_fee": True, "inputs": raw},
        tmp_path,
        content,
    )


def test_initial_fee_never_copies_other_seven_scores(initial):
    session, request, root, _ = initial
    bundle = prepare._initial_fee_bundle(session, request, root)
    assert bundle["fund_code"] == "017437" and bundle["family"] == "QDII_EQUITY"
    assert bundle["reviewed_document"]["full_total"] is None
    assert all(
        v["candidate"] is None and v["lower"] == "0" and v["upper"] == "100"
        for v in bundle["reviewed_document"]["dimensions"].values()
    )


@pytest.mark.parametrize(
    "case", ["different_fund", "no_explicit_initial", "different_share", "future_report", "foreign_original"]
)
def test_initial_fee_rejects_unproved_identity_and_future(initial, case):
    session, request, root, content = initial
    if case == "different_fund":
        request["fund_code"] = "019449"
    if case == "no_explicit_initial":
        request.pop("initial_partial_fee")
    if case == "different_share":
        request["inputs"]["share_class"] = "A"
    if case == "future_report":
        request["inputs"]["sources"]["h"]["published_on"] = "2026-11-01"
    if case == "foreign_original":
        content["data"]["security"] = [{"stock": "019449"}]
        (root / "sources/q3.json").write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError):
        prepare._initial_fee_bundle(session, request, root)
