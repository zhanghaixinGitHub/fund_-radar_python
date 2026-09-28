"""公告研究契约的合成反例；不使用真实预测标签。"""

import copy

import pytest
from app.schemas.news_research import NewsResearchBundle, content_hash
from pydantic import ValidationError


def payload():
    anchors = [{"page": 1, "text": "100"}]
    card = {
        "id": "N01",
        "announcement_id": "synthetic-1",
        "target": "2024-03-04",
        "published_date": "2024-03-01",
        "company": {"stock_code": "000001.SZ"},
        "holding": {"public_date": "2024-01-01"},
        "unknowns": ["市场预期未知"],
        "uncovered": "其他公司未知",
        "analysis": {"horizon": "未知"},
        "next_day_direction": None,
        "impact_magnitude": None,
        "prediction_eligible": False,
        "evidence": {"anchors": anchors, "pdf_sha256": "1" * 64},
        "events": {"ids": ["ONE", "TWO"], "stage": "计划与进展", "aggregation": "EVIDENCE_ONLY_NO_ADDITION"},
        "facts": {
            "amounts": [
                {"meaning": "limit", "value": "100", "unit": "CNY", "status": "PLAN"},
                {"meaning": "paid", "value": None, "unit": None, "status": "UNKNOWN", "unknown_reason": "未披露"},
            ]
        },
    }
    analysis = {
        "id": "N01",
        "target": card["target"],
        "published_date": card["published_date"],
        "stock_code": "000001.SZ",
        "prediction_eligible": False,
        "training_eligible": False,
        "event_topics": ["ONE", "TWO"],
        "event_stage": "计划与进展",
        "anchors": anchors,
        "facts": "计划限额，实际发生额未知",
        "priced_in_status": "UNKNOWN",
        "unknowns": ["市场预期未知"],
    }
    return {
        "dataset_key": "synthetic-news",
        "revision": "v1",
        "analysis_version": "test-v1",
        "fund_code": "002112",
        "source_manifest": {"synthetic.json": "a" * 64},
        "context": {"synthetic": True},
        "records": [{"card": card, "analysis": analysis}],
    }


def test_unknown_amount_unit_and_multiple_events_roundtrip():
    p = payload()
    b = NewsResearchBundle.model_validate(p)
    out = b.model_dump(mode="json")
    assert out["records"] == p["records"]
    assert out["records"][0]["card"]["facts"]["amounts"][1]["value"] is None
    assert content_hash(out) == content_hash(
        NewsResearchBundle.model_validate_json(b.model_dump_json()).model_dump(mode="json")
    )


@pytest.mark.parametrize(
    "kind", ["direction", "amount", "unknown_reason", "eligible", "unit", "late", "anchor", "duplicate"]
)
def test_contract_rejects_invented_or_inconsistent_values(kind):
    p = copy.deepcopy(payload())
    c = p["records"][0]["card"]
    if kind == "direction":
        c["next_day_direction"] = "FLAT"
    if kind == "amount":
        c["facts"]["amounts"][0]["value"] = "NaN"
    if kind == "unknown_reason":
        del c["facts"]["amounts"][1]["unknown_reason"]
    if kind == "eligible":
        p["records"][0]["analysis"]["training_eligible"] = True
    if kind == "unit":
        c["facts"]["amounts"][0]["unit"] = ""
    if kind == "late":
        c["holding"]["public_date"] = c["target"]
    if kind == "anchor":
        c["evidence"]["anchors"] = []
    if kind == "duplicate":
        p["records"].append(copy.deepcopy(p["records"][0]))
    with pytest.raises((ValidationError, ValueError)):
        NewsResearchBundle.model_validate(p)
