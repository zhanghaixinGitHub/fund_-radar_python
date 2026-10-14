"""完整链路的无损汇总、非公告基底隔离及固定预算边界。"""

import json

from scripts.local_fund_fact_chain import build_book
from scripts.local_fund_validation_chain import compressed_evidence, summary_request


def book():
    return build_book(
        {
            "id": "test",
            "body": "未来三年累计现金分红基数为该三年实现的年均可分配利润。尚未购买公司股份。",
            "body_scope": "VERIFIED_EXCERPT",
            "body_truncated": False,
        }
    )


def test_all_fact_fields_quotes_and_partial_scope_roundtrip():
    source = book()
    encoded, facts = compressed_evidence({"doc-01": source})
    decoded = json.loads(encoded)
    assert len(decoded["rows"]) == len(source["facts"]) == len(facts)
    assert all(row[-2:] == ["VERIFIED_EXCERPT", False] for row in decoded["rows"])
    assert any("年均" in row[3] for row in decoded["rows"])
    assert all(f["direction_eligible"] is False for f in facts.values())


def test_budget_failure_keeps_all_registry_facts_without_cropping():
    class Huge:
        def count(self, text):
            return 65536

    data = {
        "facts": {
            "nav": {"category": "净值", "text": "冻结净值"},
            "old_event": {"category": "公告", "text": "不能复用的旧公告解释"},
        },
        "fund_code": "002112",
        "window": {},
        "as_of": "2026-10-09",
        "latest_nav_date": "2026-10-09",
        "inventory": {},
        "overflow": [],
    }
    request, facts, receipt = summary_request(data, {"doc-01": book()}, {}, Huge())
    assert request == {} and receipt["status"] == "BUDGET_BLOCKED_NO_SOURCE_REMOVED"
    assert len(facts) == len(book()["facts"]) + 1
    assert "old_event" not in facts and "nav" in facts
