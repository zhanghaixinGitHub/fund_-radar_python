"""具体业务说明的事实、时间、链接及解释边界；不连接数据库、不训练。"""

import copy

import pytest
from app.services import direction_1d_business_explanation as explain


def original():
    full = {
        "numeric": [-0.16, -0.08, -0.18, 0.03, -0.36, 0.347, 2.0] + [0.0] * 80,
        "market_date": "2026-10-08",
        "holding_report_date": "2026-06-30",
        "counts": {"ANNOUNCEMENT": 1, "POLICY": 0, "NEWS": 0},
        "text": (
            "授予限制性股票公告 证券代码：300394 证券简称：天孚通信 公告编号：2026-054 "
            "授予数量：227.69万股 3、授予价格：119.5元/股 授予日：2026年10月8日 2、其他内容。"
        ),
        "sources": [
            {
                "id": "grant",
                "kind": "ANNOUNCEMENT",
                "title": "授予限制性股票公告",
                "available_at": "2026-10-09T08:00:00+08:00",
                "published_date": "2026-10-08",
                "source_url": "https://static.cninfo.com.cn/example.pdf",
            }
        ],
    }
    full["numeric"][7] = -0.06
    body = {
        "fund_code": "002112",
        "base_nav_date": "2026-10-08",
        "target_nav_date": "2026-10-09",
        "input": {
            "features": full["numeric"][:7],
            "feature_as_of": "2026-10-09T10:00:00+08:00",
            "values": [{"unit_nav": str(3.6 + i / 100)} for i in range(61)],
            "information": full,
        },
    }
    return body


def test_contradictory_facts_are_not_rewritten_into_bullish_causes(monkeypatch):
    monkeypatch.setattr(explain, "original_holdings", lambda _: [])
    body = original()
    unchanged = copy.deepcopy(body)
    result = explain.narrative(body, {"branches": [{"direction": "UP"}]})
    assert "尚不足以确认反弹" in result["summary"]
    assert "34.7%" in result["drivers"][0]["observation"]
    assert "并不等于估值便宜" in result["drivers"][0]["implication"]
    assert "政策、新闻" in result["context"]
    assert body == unchanged


def test_specific_grant_numbers_and_source_preserve_original_stage():
    result = explain.event_drivers(original(), [{"code": "300394.SZ", "name": "天孚通信", "weight": 0.0793}])
    assert len(result) == 1
    assert "227.69万股" in result[0]["observation"] and "119.5元/股" in result[0]["observation"]
    assert "2026年10月8日2" not in result[0]["observation"]
    assert "不是股价的合理估值" in result[0]["implication"]
    assert result[0]["source"]["url"].startswith("https://static.cninfo.com.cn/")


@pytest.mark.parametrize("change", ["future", "new_body", "unsafe_url", "routine"])
def test_no_future_unmatched_or_routine_material_becomes_a_price_reason(change):
    body = original()
    event = body["input"]["information"]["sources"][0]
    if change == "future":
        event["available_at"] = "2026-10-10T00:00:00+08:00"
    elif change == "new_body":
        event["title"] = "后来新增的消息"
    elif change == "unsafe_url":
        event["source_url"] = "javascript:alert(1)"
    else:
        event["title"] = "授予限制性股票核查意见"
        body["input"]["information"]["text"] = event["title"] + " 授予数量：227.69万股。"
    assert explain.event_drivers(body, []) == []


def test_financing_keeps_proposal_and_mentions_both_effects():
    body = original()
    event = body["input"]["information"]["sources"][0]
    event["title"] = "关于配售新H股及发行可转换债券的公告"
    body["input"]["information"]["text"] = (
        event["title"] + " 公司拟根据一般授权新增发行境外上市股份（H股），和发行可转换债券。"
    )
    item = explain.event_drivers(body, [])[0]
    assert "拟新增发行" in item["observation"]
    assert "补充资金" in item["implication"] and "摊薄" in item["implication"]
    assert item["assessment"] == "双向影响"


def test_revised_quote_file_is_not_used_as_original_evidence(monkeypatch):
    from app.services import fund_exposure_quotes

    body = original()
    body["input"]["information"].update(
        source_versions={"reports": {"old_report": {}}, "quotes": {"2026-10-08": "old_quote"}}
    )
    monkeypatch.setattr(
        fund_exposure_quotes,
        "reports",
        lambda: [
            {"raw": {"sha256": "old_report"}, "report_end": "2026-06-30", "available_at": "2026-08-31T00:00:00+08:00"}
        ],
    )
    monkeypatch.setattr(explain.info, "receipt_available", lambda *_: "verified")
    monkeypatch.setattr(explain.info, "read", lambda _: {"receipt": {"sha256": "new_quote"}})
    assert explain.original_holdings(body) == []
