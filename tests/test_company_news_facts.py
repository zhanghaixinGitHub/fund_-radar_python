"""公司公告仅作披露事实，不能跨公司、跨原件版本或把草案当实施。"""

from copy import deepcopy
from datetime import datetime

import pytest
from app.services import company_news_facts as service

AT = datetime.fromisoformat("2026-09-29T18:00:00+08:00")


def inputs():
    title = "关于限制性股票激励计划（草案）的公告"
    item = {
        "announcementId": "123",
        "adjunctUrl": "finalpage/2026-09-29/123.pdf",
        "adjunctSize": 12,
        "announcementTitle": title,
        "announcementTime": 1790611200000,
        "announced_at_source": "2026-09-29T00:00:00+08:00",
        "secCode": "000988",
        "secName": "华工科技",
        "title_plain": title,
        "listing_status": "LISTED",
    }
    document = {
        "announcement_id": "123",
        "stock_code": "000988",
        "text_status": "TEXT_EXTRACTED",
        "catalog_hash": service.digest(
            {k: item.get(k) for k in ("adjunctUrl", "adjunctSize", "announcementTitle", "announcementTime")}
        ),
        "pages": ["证券代码：000988 华工科技 " + title + "尚需股东会审议通过后实施。" * 12],
        "receipt": {
            "url": "https://static.cninfo.com.cn/finalpage/2026-09-29/123.pdf",
            "sha256": "a" * 64,
            "received_at": "2026-09-29T17:00:00+08:00",
        },
    }
    holding = {"stockCode": "000988.SZ", "stockName": "华工科技", "weightPct": "5.0"}
    report = {"endDate": "2026-06-30", "publishedDate": "2026-08-31"}
    return item, document, holding, report


def test_draft_is_disclosure_only_with_historical_holding_relation():
    fact = service.inspect_company(*inputs(), AT)
    assert fact["stage"] == "已披露文件" and fact["effective_at"] is None
    assert "2026-06-30" in fact["relation"] and "不代表当前实际仓位" in fact["relation"]
    assert not fact["prediction_eligible"] and not fact["training_eligible"]


def test_exact_legal_name_can_bind_a_document_without_stock_code_header():
    item, doc, holding, report = inputs()
    doc["pages"] = ["华工科技产业股份有限公司" + item["title_plain"] + "尚待审议。" * 30]
    identity = {"row": {"ts_code": "000988.SZ", "symbol": "000988", "fullname": "华工科技产业股份有限公司"}}
    assert service.inspect_company(item, doc, holding, report, AT, identity_evidence=identity)["business_eligible"]
    identity["row"]["ts_code"] = "999999.SZ"
    with pytest.raises(ValueError, match="IDENTITY_ANCHOR_MISSING"):
        service.inspect_company(item, doc, holding, report, AT, identity_evidence=identity)


@pytest.mark.parametrize("fault", ["company", "title", "hash", "receipt", "date", "withdrawn"])
def test_mismatched_or_unavailable_evidence_is_not_a_verified_fact(fault):
    item, doc, holding, report = deepcopy(inputs())
    if fault == "company":
        doc["stock_code"] = "999999"
    elif fault == "title":
        doc["pages"] = ["证券代码：000988另一份文书" * 30]
    elif fault == "hash":
        doc["catalog_hash"] = "b" * 64
    elif fault == "receipt":
        doc["receipt"]["received_at"] = "2026-09-30T00:00:00+08:00"
    elif fault == "date":
        item["announced_at_source"] = "2026-09-28T00:00:00+08:00"
    else:
        item["listing_status"] = "NOT_LISTED_ON_RECHECK"
    with pytest.raises(ValueError, match="NOT_VERIFIED"):
        service.inspect_company(item, doc, holding, report, AT)


def test_missing_holdings_does_not_expand_collection(monkeypatch):
    monkeypatch.setattr(service, "snapshot", lambda code: None)
    monkeypatch.setattr(service, "acquire_announcements", lambda **kwargs: pytest.fail("must not collect"))
    with pytest.raises(ValueError, match="HOLDINGS_MISSING"):
        service.collect_company_news(AT)


def test_catalog_company_prefix_and_title_quotes_do_not_discard_exact_words():
    item, doc, holding, report = inputs()
    item["title_plain"] = "华工科技：" + item["title_plain"]
    doc["pages"][0] = doc["pages"][0].replace("限制性股票激励计划", "《限制性股票激励计划》")
    fact = service.inspect_company(item, doc, holding, report, AT)
    assert fact["evidence"]["publication_title_anchor"]["rule"] == "CATALOG_SECURITY_NAME_PREFIX"


@pytest.mark.parametrize("changed", ["关于限制性股票激励计划的公告", "关于限制性股票激励计划（修订）的公告"])
def test_title_layout_normalization_keeps_draft_and_revision_conditions(changed):
    item, doc, holding, report = inputs()
    doc["pages"] = ["证券代码：000988华工科技" + changed + "相关说明。" * 30]
    with pytest.raises(ValueError, match="TITLE_ANCHOR_MISSING"):
        service.inspect_company(item, doc, holding, report, AT)


def test_legacy_catalog_requires_original_response_and_exact_metadata(tmp_path, monkeypatch):
    """缺新增字段不等于旧证据失效；但没有原响应或当前版本变化也不能凭空补字段。"""
    import json

    from app.services.fund_exposure_common import save

    item, doc, holding, report = inputs()
    del doc["catalog_hash"]
    doc.update(title=item["title_plain"], announced_at_source=item["announced_at_source"])
    monkeypatch.setattr(service, "ROOT", tmp_path)
    original = {**item, "receipt": {"url": "https://www.cninfo.com.cn/new/hisAnnouncement/query"}}
    save(tmp_path / "supplement/company-announcements/000988.SZ.json", {"rows": [original]})
    monkeypatch.setattr(service, "verified_bytes", lambda receipt: json.dumps({"announcements": [item]}).encode())
    evidence = service.legacy_catalog_evidence(item, doc, holding["stockCode"])
    assert evidence and service.inspect_company(item, doc, holding, report, AT, legacy_catalog=evidence)
    # 当前版本元数据改变，原文件即使存在也不匹配；原字节校验失败同样拒绝。
    changed = {**item, "adjunctSize": 999}
    assert service.legacy_catalog_evidence(changed, doc, holding["stockCode"]) is None
    monkeypatch.setattr(service, "verified_bytes", lambda receipt: (_ for _ in ()).throw(ValueError("HASH_MISMATCH")))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        service.legacy_catalog_evidence(item, doc, holding["stockCode"])
