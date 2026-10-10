"""由真实旧失败模式构造边界用例；只使用002112，不接真实服务或数据库。"""

from copy import deepcopy

import pytest
from app.integrations.fund_information_analysis import ResponseFormatError, decode_response
from app.schemas.fund_information_analysis import quote_catalog, validate_analysis, validate_events
from app.services import fund_information_analysis as engine
from app.services.fund_information_contracts import event_facts, group_events, reference_catalog, time_context


def event(identity, company="300308.SZ", title="2026年限制性股票激励计划", quote=None):
    return {
        "id": identity,
        "title": "模型不能决定身份",
        "quotes": [quote or "关于2026年限制性股票激励计划的实施安排"],
        "document": {
            "id": identity.split(":")[0],
            "codes": [company],
            "title": title,
            "date": "2026-08-25",
            "kind": "ANNOUNCEMENT",
            "url": "https://example.test/" + identity,
            "body_scope": "VERIFIED_EXCERPT",
            "body_truncated": False,
        },
    }


def candidate(ref="holdings"):
    return {
        "direction": "DOWN",
        "confidence": "LOW",
        "summary": "倾向下跌，把握较低。",
        "reasons": [
            {
                "title": "走势偏弱",
                "refs": [ref],
                "role": "支持下跌",
                "meaning": "所列持仓行情走弱。",
                "implication": "若弱势延续，可能拖累基金。",
            }
        ],
        "counterpoints": [],
        "synthesis": "以行情弱势延续为条件判断，不能视为规律。",
        "change_conditions": [],
        "limitations": ["披露持仓并非实时仓位。"],
    }


def timing_input(target="2026-08-26", base="2026-08-25"):
    return {
        "fund_code": "002112",
        "as_of": target + "T08:30:00+08:00",
        "window": {"target_nav_date": target, "base_nav_date": base},
        "nav": [{"nav_date": base}],
        "latest_nav_date": base,
        "companies": [{"code": "300308.SZ", "quote": {"date": base, "change_pct": -1}}],
        "market": {c: {"date": base, "change_pct": -1} for c in ("000300.SH", "000905.SH")},
    }


def test_company_identity_is_a_program_boundary():
    groups = group_events([event("a:0"), event("b:0", "300502.SZ")])
    assert len(groups) == 2
    assert all(len(g["company_codes"]) == 1 for g in groups)
    assert sorted(i for g in groups for i in g["event_ids"]) == ["a:0", "b:0"]


def test_same_company_same_type_different_concrete_plan_stays_separate():
    groups = group_events(
        [
            event("a:0", quote="2026年第一期限制性股票激励计划的实施安排"),
            event("b:0", quote="2026年第二期限制性股票激励计划的实施安排"),
        ]
    )
    assert len(groups) == 2
    assert groups[0]["anchor"] != groups[1]["anchor"]


def test_multifile_same_matter_is_one_fact_with_all_sources():
    groups = group_events(
        [event("a:0"), event("b:0", title="董事会决议公告", quote="审议通过关于2026年限制性股票激励计划的议案。")]
    )
    assert len(groups) == 1
    facts = event_facts(groups, [])
    assert len(facts) == 1
    fact = next(iter(facts.values()))
    assert len(fact["sources"]) == 2
    assert fact["member_event_ids"] == ["a:0", "b:0"]
    assert "审议通过关于2026年限制性股票激励计划的议案。" in fact["text"]


def test_uncertain_transactions_remain_distinct_with_one_issuer_assessment():
    groups = group_events(
        [
            event("a:0", title="对外投资公告", quote="公司拟参与投资甲项目，尚需审批。"),
            event("b:0", title="对外投资公告", quote="公司拟参与投资乙项目，尚需审批。"),
        ]
    )
    assert len(groups) == 2
    facts = reference_catalog(event_facts(groups, []))
    assert len(facts) == 1
    fact = next(iter(facts.values()))
    assert len(fact["matters"]) == 2
    assert all(m["identity_status"] == "UNRESOLVED" for m in fact["matters"])
    validate_analysis(candidate(next(iter(facts))), facts)


def test_no_missing_or_duplicate_events_in_partition():
    events = [event("a:0"), event("a:1", title="减持完成", quote="股东减持计划已经实施完毕，结果如下。"), event("b:0")]
    assert sorted(i for g in group_events(events) for i in g["event_ids"]) == ["a:0", "a:1", "b:0"]
    with pytest.raises(ValueError, match="EVENT_ID_DUPLICATE"):
        group_events([event("a:0"), event("a:0")])


def test_distinct_explanations_do_not_add_duplicate_evidence_units():
    facts = reference_catalog(event_facts(group_events([event("a:0"), event("b:0")]), []))
    value = candidate(next(iter(facts)))
    value["reasons"].append(deepcopy(value["reasons"][0]))
    result = validate_analysis(value, facts)
    assert result["evidence_units"] == [next(iter(facts))]


def test_exact_quotes_do_not_remove_spaces_or_join_fragments():
    document = {"id": "a", "body": "公司尚未完成回购， 金额为 100 万元。"}
    parsed = {
        "items": [
            {
                "id": "a",
                "events": [
                    {
                        "title": "回购计划",
                        "assessment": "NEUTRAL",
                        "stage": "拟议",
                        "quotes": [document["body"]],
                        "meaning": "尚未实施",
                        "mechanism": "待执行",
                        "caveat": "仅部分内容",
                    }
                ],
            }
        ]
    }
    validate_events(parsed, [document])
    for invalid in [
        "公司尚未完成回购，金额为100万元。",
        "公司完成回购， 金额为 100 万元。",
        "公司尚未完成回购， 金额为 200 万元。",
    ]:
        parsed["items"][0]["events"][0]["quotes"] = [invalid]
        with pytest.raises(ValueError, match=r"QUOTE_MISMATCH.*events\[0\].quotes\[0\]"):
            validate_events(parsed, [document])


def test_short_reference_is_deterministic_and_missing_prefix_is_never_guessed():
    facts = reference_catalog({"event:abc:0": {"text": "原文"}, "holdings": {"text": "行情"}})
    assert facts == reference_catalog({"holdings": {"text": "行情"}, "event:abc:0": {"text": "原文"}})
    ref = next(iter(facts))
    assert len(ref) == 13
    validate_analysis(candidate(ref), facts)
    with pytest.raises(ValueError, match="未知编号"):
        validate_analysis(candidate(ref[1:]), facts)


@pytest.mark.parametrize(
    "target,base",
    [
        ("2026-08-26", "2026-08-25"),
        ("2026-09-29", "2026-09-28"),
        ("2026-08-31", "2026-08-28"),
        ("2026-09-28", "2026-09-24"),
    ],
)
def test_real_regression_calendar_samples_have_baseline_quotes(target, base):
    result = time_context(timing_input(target, base))
    assert result["baseline_complete"] is True
    assert result["baseline_date"] == base
    assert result["target_close_status"] == "NOT_YET_OCCURRED"


def test_partial_quote_gap_is_not_hidden_by_one_latest_stock():
    value = timing_input()
    value["companies"].append({"code": "300502.SZ", "quote": {"date": "2026-08-24"}})
    result = time_context(value)
    assert result["baseline_complete"] is False
    assert result["latest_stock_quote_date"] == "2026-08-25"
    assert result["stocks"]["300502.SZ"]["status"] == "STALE"


def test_date_without_price_is_not_complete_market_data():
    value = timing_input()
    value["companies"][0]["quote"]["change_pct"] = None
    result = time_context(value)
    assert result["baseline_complete"] is False
    assert result["stocks"]["300308.SZ"]["status"] == "MISSING_VALUE"


def test_future_quote_or_guessed_holiday_baseline_rejected():
    value = timing_input("2026-09-28", "2026-09-25")
    with pytest.raises(ValueError, match="TIME_WINDOW_MISMATCH"):
        time_context(value)
    value = timing_input()
    value["companies"][0]["quote"]["date"] = "2026-08-26"
    with pytest.raises(ValueError, match="QUOTE_DATE_INVALID"):
        time_context(value)


@pytest.mark.parametrize(
    "claim", ["目标日缺少最近交易日行情。", "没有取得最新行情，不能判断。", "净值早于比较基准日。"]
)
def test_date_misreading_cannot_survive_valid_model_audit(claim):
    value = candidate()
    value["limitations"] = [claim]
    with pytest.raises(ValueError, match="DATE_CONTRADICTION"):
        validate_analysis(value, {"holdings": {}}, time_context(timing_input()))


@pytest.mark.parametrize("claim", ["截至基准日，所列持仓行情走弱。", "若目标日市场情绪转弱，持仓可能承压。"])
def test_true_baseline_and_conditional_statements_not_rejected_as_date_gaps(claim):
    value = candidate()
    value["summary"] = claim
    validate_analysis(value, {"holdings": {}}, time_context(timing_input()))


@pytest.mark.parametrize("raw", ['{"ok":true}', '```json\n{"ok":true}\n```'])
def test_complete_json_and_outer_fence_are_parsed_without_rewriting(raw):
    assert decode_response(raw) == {"ok": True}


@pytest.mark.parametrize("raw", ['{"x":1', '{"x":1,"x":2}', "[]", '{"x":NaN}', '文字 {"ok":true}'])
def test_invalid_json_is_preserved_and_never_guessed(raw):
    with pytest.raises(ResponseFormatError) as caught:
        decode_response(raw)
    assert caught.value.raw == raw


def test_format_repair_is_bounded_and_passes_original_response(monkeypatch):
    calls = []

    def request(prompt, value, stage, budget):
        calls.append(stage)
        if len(calls) == 2:
            assert value["previous_response"] == "broken"
        raise ResponseFormatError("broken")

    monkeypatch.setattr(engine, "request", request)
    with pytest.raises(ResponseFormatError):
        engine.request_json("prompt", {}, "SYNTHESIS", None)
    assert calls == ["SYNTHESIS", "SYNTHESIS_FORMAT_REPAIR"]


def test_failed_document_is_not_a_successful_direction(monkeypatch):
    value = timing_input()
    value.update(
        documents=[{"id": "a", "body": "本公司公告的实际原文。"}], facts={"holdings": {}}, inventory=[], overflow=0
    )
    monkeypatch.setattr(engine, "request", lambda *a: {"items": []})
    with pytest.raises(ValueError, match="DOCUMENT_PARSE_FAILED"):
        engine.analyze(value, None)


@pytest.mark.parametrize(
    "body", ["原文带有 100 万元与否定，尚未完成。\n原文不能去除空格或拼接。", "原" * 707, "原" * 351]
)
def test_quote_ids_cover_original_bytes_and_keep_conditions(body):
    catalog = quote_catalog(body)
    assert "".join(row["text"] for row in catalog.values()) == body
    assert all(8 <= len(row["text"]) <= 350 for row in catalog.values())
    first = next(iter(catalog))
    raw = {
        "items": [
            {
                "id": "a",
                "events": [
                    {
                        "title": "事项",
                        "assessment": "NEUTRAL",
                        "stage": "拟议",
                        "quote_ids": [first],
                        "meaning": "尚待落实",
                        "mechanism": "存在条件",
                        "caveat": "本次只是片段",
                    }
                ],
            }
        ]
    }
    events = validate_events(raw, [{"id": "a", "body": body}])
    assert events[0]["quotes"] == [catalog[first]["text"]]
    assert "quote_ids" in raw["items"][0]["events"][0]
    raw["items"][0]["events"][0]["quote_ids"] = ["Q9999"]
    with pytest.raises(ValueError, match="QUOTE_ID_INVALID"):
        validate_events(raw, [{"id": "a", "body": body}])


def test_policy_and_announcement_never_share_an_event_or_source_category():
    announcement = event("a:0")
    policy = event("b:0")
    policy["document"]["kind"] = "POLICY"
    groups = group_events([announcement, policy])
    assert len(groups) == 2 and len({g["id"] for g in groups}) == 2
    facts = event_facts(groups, [])
    assert {f["category"] for f in facts.values()} == {"公告", "政策"}
    assert all(len(f["sources"]) == 1 for f in facts.values())


def test_official_index_name_is_not_a_rewritten_numeric_fact():
    value = candidate()
    value["summary"] = "沪深300与中证500走弱，若延续可能拖累。"
    facts = {"holdings": {"text": "沪深300与中证500当日行情"}}
    validate_analysis(value, facts)
    value["summary"] = "沪深300上涨百分之五。"
    with pytest.raises(ValueError, match="UNSUPPORTED_ASSERTION"):
        validate_analysis(value, facts)


def test_summary_eps_negation_is_not_positive_claim():
    value = candidate()
    value["synthesis"] = "现金分红不等于提升每股收益。"
    validate_analysis(value, {"holdings": {}}, {})
    value["synthesis"] = "现金分红提升每股收益。"
    with pytest.raises(ValueError, match="每股收益"):
        validate_analysis(value, {"holdings": {}}, {})
