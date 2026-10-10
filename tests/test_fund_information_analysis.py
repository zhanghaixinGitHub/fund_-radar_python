"""验证引用、方向取舍、旧行情口径与无外部调用的页面投影。"""

from contextlib import nullcontext
from copy import deepcopy
from datetime import date, datetime
from types import SimpleNamespace

import pytest
from app.schemas.fund_information_analysis import validate_analysis, validate_events
from app.services.fund_information_analysis import display_narrative, narrative
from app.services.fund_information_snapshot import facts, unique_documents


def candidate():
    return {
        "direction": "DOWN",
        "confidence": "LOW",
        "summary": "持仓偏弱，倾向下跌，但把握较低。",
        "reasons": [
            {
                "title": "持仓走弱",
                "refs": ["holdings"],
                "role": "支持下跌",
                "meaning": "已发生的走势偏弱。",
                "implication": "若短期弱势延续，可能继续拖累。",
            }
        ],
        "counterpoints": [],
        "synthesis": "主要依据是已有行情，弱势延续属于假设。",
        "change_conditions": ["持仓股票转强可能改变判断。"],
        "limitations": ["缺少最新交易日行情。"],
    }


def test_refs_and_direction_must_match():
    value = candidate()
    assert validate_analysis(value, {"holdings": {}})["direction"] == "DOWN"
    value["direction"] = "UP"
    with pytest.raises(ValueError, match="DIRECTION_UNEXPLAINED"):
        validate_analysis(value, {"holdings": {}})


@pytest.mark.parametrize("replacement", ["明日上涨80%", "必涨", "跌多了就反弹"])
def test_free_numbers_and_unsupported_promises_rejected(replacement):
    value = candidate()
    value["summary"] = replacement
    with pytest.raises(ValueError):
        validate_analysis(value, {"holdings": {}})


def test_missing_ref_rejected():
    with pytest.raises(ValueError, match="REFERENCE_INVALID"):
        validate_analysis(candidate(), {})


def test_negated_eps_warning_is_not_mistaken_for_positive_claim():
    value = candidate()
    value["reasons"][0]["implication"] = "现金分红不等于提升每股收益。"
    assert validate_analysis(value, {"holdings": {}})["direction"] == "DOWN"
    value["reasons"][0]["implication"] = "现金分红提升每股收益。"
    with pytest.raises(ValueError, match="每股收益"):
        validate_analysis(value, {"holdings": {}})


def test_quote_cannot_change_negation_or_numbers():
    doc = {"id": "a", "body": "公司尚未完成回购，不存在已经执行的回购金额。"}
    event = {
        "title": "回购进展",
        "assessment": "NEUTRAL",
        "stage": "拟议",
        "quotes": [doc["body"]],
        "meaning": "回购尚未执行",
        "mechanism": "实际效果待实施",
        "caveat": "不能视为已回购",
    }
    value = {"items": [{"id": "a", "events": [event]}]}
    assert len(validate_events(value, [doc])) == 1
    invalid = deepcopy(value)
    invalid["items"][0]["events"][0]["quotes"] = ["公司已经完成回购，存在已经执行的回购金额。"]
    with pytest.raises(ValueError, match="QUOTE_MISMATCH"):
        validate_events(invalid, [doc])


def test_weighted_return_keeps_net_asset_denominator():
    data = {
        "nav": [],
        "companies": [
            {
                "code": "1",
                "name": "公司",
                "weight_pct": 9.5,
                "quote": {"date": "2026-10-08", "change_pct": -10},
                "financials": [],
            }
        ],
        "report": {"endDate": "2026-06-30"},
        "market": {},
    }
    result = facts(data)
    assert "-0.95个百分点" in result["stock:1"]["text"]
    assert "9.50%" in result["holdings"]["text"]
    assert "不是目标日预测涨幅" in result["holdings"]["text"]


def test_narrative_uses_saved_fact_not_model_number():
    result = narrative(candidate(), {"holdings": {"category": "持仓行情", "text": "原事实", "source": None}}, [], [])
    assert result["drivers"][0]["observation"] == "原事实"
    assert result["context"] == candidate()["synthesis"]
    assert result["counterpoints"] == []


def test_announcement_company_follows_each_saved_fact_not_list_order():
    """相同标题、同日公告也按事实引用区分公司；重排 refs 不能错配，展示不覆盖原档。"""
    analysis = candidate()
    analysis["reasons"][0]["refs"] = ["cambridge", "innolight"]
    source = {"title": "董事会决议公告", "url": "https://example.com/a.pdf", "publishedDate": "2026-10-08"}
    facts = {
        "innolight": {"category": "公告", "text": "中际原文", "relation": "中际旭创披露仓位9.92%", "source": source},
        "cambridge": {"category": "公告", "text": "剑桥原文", "relation": "剑桥科技披露仓位5.57%", "source": source},
    }
    body = {"analysis": analysis, "evidence": facts, "narrative": narrative(analysis, facts, [], [])}
    original = deepcopy(body)
    driver = display_narrative(body)["drivers"][0]
    assert driver["observation"] == "【剑桥科技】剑桥原文\n【中际旭创】中际原文"
    assert [s["title"] for s in driver["sources"]] == ["剑桥科技｜董事会决议公告", "中际旭创｜董事会决议公告"]
    assert driver["meaning"] == analysis["reasons"][0]["meaning"]
    assert body == original


@pytest.mark.parametrize("relation,category,expected", [
    ("", "公告", "【关联公司待确认】原文"),
    ("甲公司披露仓位1.00%；乙公司披露仓位2.00%", "公告", "【相关公司：甲公司、乙公司】原文"),
    ("甲公司披露仓位1.00%", "新闻", "原文"),
])
def test_missing_or_multi_company_relation_does_not_guess_issuer(relation, category, expected):
    analysis = candidate()
    facts = {"holdings": {"category": category, "text": "原文", "relation": relation,
        "source": {"title": "标题", "url": "https://example.com/a", "publishedDate": "2026-10-08"}}}
    body = {"analysis": analysis, "evidence": facts, "narrative": narrative(analysis, facts, [], [])}
    assert display_narrative(body)["drivers"][0]["observation"] == expected


def test_duplicate_publication_does_not_add_input_evidence():
    doc = {"id": "a", "date": "2026-10-08", "codes": ["x"], "body": "同一份公告正文"}
    assert unique_documents([doc, {**doc, "id": "b", "body": "同一份 公告正文"}]) == [doc]
    assert len(unique_documents([doc, {**doc, "id": "b", "body": "同一计划的新进展"}])) == 2


def test_bounded_repair_keeps_validation(monkeypatch):
    from app.services import fund_information_analysis as service

    calls = []

    def request(prompt, value, stage, budget):
        calls.append(stage)
        if stage == "AUDIT":
            return {"valid": True, "issues": []}
        result = candidate()
        result["reasons"][0]["refs"] = list(value["facts"])
        result["limitations"] = ["披露持仓不是实时仓位。"]
        if stage == "SYNTHESIS":
            result["confidence"] = "MEDIUM"
        return result

    monkeypatch.setattr(service, "request", request)
    monkeypatch.setattr(service, "time_context", lambda data: {})
    data = {
        "documents": [],
        "companies": [],
        "facts": {"holdings": {}},
        "window": {},
        "as_of": "now",
        "latest_nav_date": None,
        "inventory": [],
        "overflow": 0,
    }
    result, _, _ = service.analyze(data, None)
    assert result["confidence"] == "LOW"
    assert calls == ["SYNTHESIS", "SYNTHESIS_REPAIR", "AUDIT"]


def test_wrong_semantics_cannot_be_published_after_retry(monkeypatch):
    from app.services import fund_information_analysis as service

    calls = []

    def request(prompt, value, stage, budget):
        calls.append(stage)
        result = candidate()
        result["reasons"][0]["refs"] = list(value["facts"])
        result["limitations"] = ["披露持仓不是实时仓位。"]
        return {"valid": False, "issues": ["遗漏否定条件"]} if stage == "AUDIT" else result

    monkeypatch.setattr(service, "request", request)
    monkeypatch.setattr(service, "time_context", lambda data: {})
    data = {
        "documents": [],
        "companies": [],
        "facts": {"holdings": {}},
        "window": {},
        "as_of": "now",
        "latest_nav_date": None,
        "inventory": [],
        "overflow": 0,
    }
    with pytest.raises(ValueError, match="REVIEW_FAILED"):
        service.analyze(data, None)
    assert calls.count("SYNTHESIS_REPAIR") == 2


@pytest.mark.parametrize("target_nav,expected", [("0.9", "DOWN"), ("1", "FLAT"), ("1.1", "UP")])
def test_later_published_base_keeps_original_comparison_date(monkeypatch, target_nav, expected):
    """预测时只有更早净值，公布后仍比较原定基准日；相等必须是持平，不参与训练。"""
    from app.services import fund_information_labels as service
    from app.services.direction_1d_protocol import ZONE

    now = datetime(2026, 10, 12, 21, tzinfo=ZONE)
    body = {
        "target_nav_date": "2026-10-12", "base_nav_date": "2026-10-09",
        "input": {"source_id": "same-source", "values": [{"nav_date": "2026-10-08"}]},
        "task_key": "002112-test", "input_snapshot_id": "input",
        "target_definition": "UNIT_NAV_DIRECTION", "expires_at": "2026-11-12T00:00:00+08:00",
    }
    connection = SimpleNamespace(execute=lambda *args: SimpleNamespace(
        mappings=lambda: SimpleNamespace(first=lambda: None)))
    monkeypatch.setattr(service, "get_engine", lambda: SimpleNamespace(begin=lambda: nullcontext(connection)))
    monkeypatch.setattr(service.repo, "clock", lambda: now)
    monkeypatch.setattr(service.repo, "source", lambda c: {"source_id": "same-source"})

    def navs(c, code, source, base, target):
        assert code == "002112" and base == date(2026, 10, 9) and target == date(2026, 10, 12)
        return [{"nav_date": base, "unit_nav": "1", "content_hash": "base"},
                {"nav_date": target, "unit_nav": target_nav, "content_hash": "target"}]

    monkeypatch.setattr(service.repo, "navs", navs)
    monkeypatch.setattr(service.repo, "observe", lambda c, code, source, rows, at: rows)
    monkeypatch.setattr(service.repo, "save_snapshot", lambda *args: ("label", "hash"))
    payload = service.labels(body)["payload"]
    assert payload["actual_direction"] == expected
    assert payload["base_initially_missing"] is True
    assert payload["base_revised"] is False
    assert payload["training_eligible"] is False


def test_future_target_does_not_fetch_or_guess_answer(monkeypatch):
    from app.services import fund_information_labels as service
    from app.services.direction_1d_protocol import ZONE

    monkeypatch.setattr(service.repo, "clock", lambda: datetime(2026, 10, 9, 17, tzinfo=ZONE))
    monkeypatch.setattr(service, "get_engine", lambda: pytest.fail("目标日未到，不应获取或生成答案"))
    assert service.labels({"target_nav_date": "2026-10-12", "base_nav_date": "2026-10-09"}) == {
        "status": "PENDING_TARGET"
    }
