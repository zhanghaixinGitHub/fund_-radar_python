"""由真实旧失败模式构造边界用例；只使用002112，不接真实服务或数据库。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from app.integrations.fund_information_analysis import ResponseFormatError, decode_response
from app.schemas.fund_information_analysis import quote_catalog, validate_analysis, validate_events
from app.services import fund_information_analysis as engine
from app.services.fund_document_parse_cache import DocumentParseCache, digest
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


def test_invalid_format_preserves_original_response_without_paid_repair(monkeypatch):
    calls = []

    def request(prompt, value, stage, budget):
        calls.append(stage)
        raise ResponseFormatError("broken")

    monkeypatch.setattr(engine, "request", request)
    with pytest.raises(ResponseFormatError) as caught:
        engine.request_json("prompt", {}, "SYNTHESIS", None)
    assert caught.value.raw == "broken"
    assert calls == ["SYNTHESIS"]


def test_failed_document_is_not_a_successful_direction(monkeypatch, tmp_path):
    value = timing_input()
    body = "本公司公告的实际原文。"
    document = {
        "id": "a",
        "kind": "ANNOUNCEMENT",
        "source_hash": digest(body),
        "body": body,
        "body_scope": "FULL_TEXT",
        "body_truncated": False,
        "date": "2026-08-25",
        "available_at": "2026-08-25T10:00:00+08:00",
    }
    value.update(documents=[document], facts={"holdings": {}}, inventory=[], overflow=0)
    path = tmp_path / "parses.sqlite3"
    monkeypatch.setattr(
        "app.services.fund_material_parsing.get_settings", lambda: SimpleNamespace(fund_parse_cache_path=str(path))
    )
    calls = []

    def request(prompt, payload, stage, budget):
        calls.append(stage)
        return {"items": []}

    monkeypatch.setattr(engine, "request", request)
    # 单份解析失败后仍检查其余事实；综合本身返回非法结构时才拒绝该次方向。
    with pytest.raises(ValueError, match="ANALYSIS_REVIEW_FAILED"):
        engine.analyze(value, None)
    assert calls == ["EVENT", "SYNTHESIS"]
    assert DocumentParseCache(path).lookup(document, value["as_of"])["state"] == "BLOCKED"


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


def quoted_material(count):
    """生成足够长且每片不同的完整原文，末尾条件用于检查超十条后仍完整保留。"""
    body = "\n".join(f"第{i:02}项条件：" + "仍须核对主体、审批及未完成的实施条件。" * 15 for i in range(count))
    doc = {
        "id": "a",
        "kind": "ANNOUNCEMENT",
        "title": "公司回购计划公告",
        "body": body,
        "body_scope": "FULL_TEXT",
        "body_truncated": False,
        "source_hash": digest(body),
        "codes": ["300308.SZ"],
        "date": "2026-10-12",
        "available_at": "2026-10-12T10:00:00+08:00",
    }
    catalog = quote_catalog(body)
    raw = {
        "items": [
            {
                "id": "a",
                "events": [
                    {
                        "title": "回购安排",
                        "assessment": "NEUTRAL",
                        "stage": "拟议",
                        "quote_ids": list(catalog),
                        "meaning": "保留所列条款事实",
                        "mechanism": "效果取决于后续落实",
                        "caveat": "各项审批和否定条件仍适用",
                    }
                ],
            }
        ]
    }
    return doc, raw, catalog


@pytest.mark.parametrize("count", [4, 10, 11, 21])
def test_more_than_three_or_ten_quotes_keep_all_conditions_in_one_evidence_unit(count):
    doc, raw, catalog = quoted_material(count)
    original = deepcopy(raw)
    parsed = validate_events(raw, [doc])
    assert raw == original
    assert len(parsed) == 1
    record = parsed[0]
    assert record["quotes"] == [row["text"] for row in catalog.values()]
    indices = record["quote_groups"]
    assert all(1 <= len(group) <= 10 for group in indices)
    assert [index for group in indices for index in group] == list(range(len(catalog)))
    fact = next(iter(event_facts(group_events(parsed), []).values()))
    assert fact["member_event_ids"] == ["a:0"]
    assert len(fact["quote_receipts"]) == len(catalog)
    assert all(row["text"] in fact["text"] for row in catalog.values())
    for receipt in fact["quote_receipts"]:
        assert doc["body"][receipt["start"] : receipt["end"]] == receipt["quote"]


def test_repeated_quote_ids_are_deduplicated_without_mutating_first_answer():
    doc, raw, catalog = quoted_material(11)
    ids = list(catalog)
    raw["items"][0]["events"][0]["quote_ids"] = [ids[-1], *ids, ids[0]]
    original = deepcopy(raw)
    parsed = validate_events(raw, [doc])[0]
    assert raw == original
    assert parsed["quote_normalization"]["supplied_count"] == len(ids) + 2
    assert len(parsed["quotes"]) == len(ids)
    assert parsed["quote_spans"] == [{k: row[k] for k in ("start", "end")} for row in catalog.values()]


def test_invalid_reference_after_ten_valid_references_is_still_rejected():
    doc, raw, _ = quoted_material(11)
    raw["items"][0]["events"][0]["quote_ids"].append("Q9999")
    with pytest.raises(ValueError, match="QUOTE_ID_INVALID"):
        validate_events(raw, [doc])


def test_old_cached_text_quotes_over_ten_remain_source_checked():
    doc, raw, catalog = quoted_material(11)
    item = raw["items"][0]["events"][0]
    item.pop("quote_ids")
    item["quotes"] = [row["text"] for row in catalog.values()]
    assert len(validate_events(raw, [doc])[0]["quotes"]) == len(catalog)
    item["quotes"][-1] = "其他公司尚未完成回购，不能视为本份公告的真实原文。"
    with pytest.raises(ValueError, match="QUOTE_MISMATCH"):
        validate_events(raw, [doc])


def test_incremental_parser_keeps_more_than_ten_quotes_and_reuses_without_new_calls(monkeypatch, tmp_path):
    doc, raw, catalog = quoted_material(11)
    path = tmp_path / "parses.sqlite3"
    monkeypatch.setattr(
        "app.services.fund_material_parsing.get_settings", lambda: SimpleNamespace(fund_parse_cache_path=str(path))
    )
    from app.services.fund_material_parsing import parse_documents

    data = {
        "fund_code": "002112",
        "as_of": "2026-10-13T12:00:00+08:00",
        "window": {"target_nav_date": "2026-10-13"},
        "documents": [doc],
    }
    calls = []

    def provider(*args):
        calls.append(args[2])
        return deepcopy(raw)

    first = parse_documents(data, "prompt", None, provider)
    second = parse_documents(data, "prompt changed", None, provider)
    assert calls == ["EVENT"]
    assert first == second
    assert len(first[0]["quotes"]) == len(catalog)
    assert DocumentParseCache(path).lookup(doc, data["as_of"])["state"] == "REUSE"


def test_notice_over_ten_matters_selects_ten_without_rejecting_or_changing_original():
    doc, raw, _ = quoted_material(1)
    original_event = raw["items"][0]["events"][0]
    raw["items"][0]["events"] = [{**deepcopy(original_event), "title": f"公开事项{i}"} for i in range(12)]
    raw["items"][0]["events"].insert(0, deepcopy(raw["items"][0]["events"][0]))
    first = deepcopy(raw)
    events = validate_events(raw, [doc])
    assert len(events) == 10
    assert [e["title"] for e in events] == [f"公开事项{i}" for i in range(10)]
    assert events[0]["event_selection"] == {"supplied_count": 13, "retained_count": 10, "limit": 10, "omitted_count": 2}
    assert raw == first


def test_ten_matters_reuse_keeps_full_first_response_and_selection_limits(tmp_path, monkeypatch):
    from app.services.fund_material_parsing import parse_documents

    doc, raw, _ = quoted_material(1)
    original = raw["items"][0]["events"][0]
    raw["items"][0]["events"] = [{**deepcopy(original), "title": f"具体事项{i}"} for i in range(12)]
    path = tmp_path / "parses.sqlite3"
    monkeypatch.setattr(
        "app.services.fund_material_parsing.get_settings", lambda: SimpleNamespace(fund_parse_cache_path=path)
    )
    data = {
        "fund_code": "002112",
        "as_of": "2026-10-13T12:00:00+08:00",
        "documents": [doc],
        "window": {"target_nav_date": "2026-10-13"},
    }
    calls = []

    def provider(*args):
        calls.append(args[2])
        return deepcopy(raw)

    first = parse_documents(data, "prompt", None, provider)
    second = parse_documents(data, "changed prompt", None, provider)
    assert first == second and len(second) == 10 and calls == ["EVENT"]
    assert second[0]["event_selection"]["omitted_count"] == 2
    cache = DocumentParseCache(path)
    with cache.connection() as connection:
        row = connection.execute("SELECT raw_json FROM parse_result WHERE cache_key=?", (cache.key(doc),)).fetchone()
    import json

    assert json.loads(row["raw_json"])["first_response"] == raw


def test_long_event_descriptions_and_quotes_preserve_trailing_negation_and_original_spans():
    doc, raw, _ = quoted_material(4)
    item = raw["items"][0]["events"][0]
    item.pop("quote_ids")
    item["quotes"] = [doc["body"]]
    title = "本公司募投项目的实施进展说明" * 18
    description = "有关项目的实施取决于各方履约。" * 24 + "但股东会尚未批准，不能视为已经实施。"
    item.update(title=title, meaning=description, mechanism=description, caveat=description)
    first = deepcopy(raw)
    record = validate_events(raw, [doc])[0]
    assert raw == first
    assert record["title"].replace("\n", "") == title
    assert all(record[field].replace("\n", "") == description for field in ("meaning", "mechanism", "caveat"))
    assert record["caveat"].endswith("但股东会尚未批准，不能视为已经实施。")
    assert "".join(record["quotes"]) == doc["body"]
    assert all(len(q) <= 350 for q in record["quotes"])
    for quote, span in zip(record["quotes"], record["quote_spans"], strict=True):
        assert doc["body"][span["start"] : span["end"]] == quote


def test_short_but_real_zero_interest_quote_is_not_rejected_for_character_count():
    doc, raw, _ = quoted_material(1)
    doc["body"] = "零息"
    item = raw["items"][0]["events"][0]
    item["quote_ids"] = ["Q0001"]
    assert validate_events(raw, [doc])[0]["quotes"] == ["零息"]


def test_synthesis_text_and_list_targets_do_not_discard_conditions_or_references():
    value = candidate()
    description = "该判断仅以弱势延续为条件，不能视为确定事实。" * 40 + "若情况改变则需要重新评估。"
    value["summary"] = description
    value["synthesis"] = description
    value["reasons"] = [
        {
            **deepcopy(value["reasons"][0]),
            "title": "当前判断的条件性依据" * 20,
            "meaning": description,
            "implication": description,
            "refs": [f"source-{i}" for i in range(7)] + ["source-0"],
        }
        for _ in range(8)
    ]
    value["counterpoints"] = [{**deepcopy(value["reasons"][0]), "role": "支持上涨"} for _ in range(5)]
    value["change_conditions"] = [description + suffix for suffix in "甲乙丙丁"]
    value["limitations"] = [description + suffix for suffix in "甲乙丙丁戊己"]
    original = deepcopy(value)
    facts = {f"source-{i}": {} for i in range(7)}
    result = validate_analysis(value, facts)
    assert value == original
    assert len(result["reasons"]) == 8 and len(result["counterpoints"]) == 5
    assert len(result["change_conditions"]) == 4 and len(result["limitations"]) == 6
    assert result["summary"].replace("\n", "") == description
    assert result["synthesis"].replace("\n", "") == description
    assert len(result["reasons"][0]["refs"]) == 7
    assert result["reasons"][0]["implication"].endswith("若情况改变则需要重新评估。")
    assert len(result["evidence_units"]) == 7


def test_paragraph_organization_is_idempotent_and_keeps_existing_line_endings():
    from app.schemas.fund_information_analysis import organize_text

    text = "原始条件仍须批准。" * 50 + "\r\n尚未执行，不保证最终完成。"
    organized = organize_text(text, 250)
    assert organize_text(organized, 250) == organized
    assert organized.endswith("\r\n尚未执行，不保证最终完成。")
    assert organized.replace("\n", "") == text.replace("\n", "")


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
