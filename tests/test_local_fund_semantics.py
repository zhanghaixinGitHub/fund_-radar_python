"""真实错题类型及其正确边界的离线回归；不连接数据库、不调用模型。"""

import json

import pytest
from scripts.local_fund_evidence_packet import fact_packet
from scripts.local_fund_semantics import (
    check_structured,
    manager_input,
    manager_issues,
    organized_catalog,
    semantic_issues,
    structured_request,
)
from scripts.local_fund_token_budget import LocalTokenBudget


@pytest.mark.parametrize(
    "source,wrong,right,code",
    [
        (
            "债券为零息债券，不计利息。",
            "零息债券提供较高票面收益率。",
            "债券为零息，不计利息。",
            "ZERO_INTEREST_CONTRADICTION",
        ),
        (
            "截至本公告披露日，本次员工持股计划尚未购买公司股份。",
            "未提及实际购买股份情况。",
            "截至披露日尚未购买公司股份。",
            "EXPLICIT_NON_PURCHASE_LOST",
        ),
        (
            "新增某子公司为实施主体并新增相应实施地点。",
            "实施主体变更为某子公司。",
            "新增某子公司为实施主体。",
            "ADDITION_IS_NOT_REPLACEMENT",
        ),
        (
            "不存在变相改变募集资金使用用途的情形。",
            "内部决策确认资金用途变更。",
            "不存在变相改变募集资金使用用途。",
            "PURPOSE_CHANGE_CONTRADICTION",
        ),
        (
            "备案不表明对材料真实性、准确性、完整性作出保证或者认定。",
            "备案标志着监管对材料真实性的初步认可。",
            "备案不代表对材料真实性的认可。",
            "FILING_IS_NOT_TRUTH_APPROVAL",
        ),
        (
            "现金累计分红不少于该三年实现的年均可分配利润的百分之三十。",
            "现金累计分红不少于三年可分配利润的百分之三十。",
            "累计分红基数为三年年均可分配利润。",
            "ANNUAL_AVERAGE_BASIS_MISSING",
        ),
        (
            "目标值150%，触发值120%；An≤A<Am时X=A/Am。",
            "增长率达到150%方可触发归属。",
            "达到触发值后仍须按归属比例公式核对。",
            "TARGET_TRIGGER_CONFUSION",
        ),
        (
            "5名激励对象离职，名单进行调整。",
            "部分高管因离职导致名单调整。",
            "5名激励对象离职。",
            "EXECUTIVE_IDENTITY_UNSUPPORTED",
        ),
        (
            "协议受让标的公司部分股份，过户完成。",
            "标的公司主营光模块，有利于协同。",
            "股份过户已经完成。",
            "BUSINESS_DESCRIPTION_UNSUPPORTED",
        ),
        (
            "控制权发生变更的，由股东会决定是否继续执行。其他终止情况回购注销。",
            "控制权变更时股份将回购注销。",
            "控制权变更由股东会决定是否继续执行。",
            "CHANGE_OF_CONTROL_CONDITIONS_MERGED",
        ),
        (
            "研发投入强度不低于7.60%。净资产收益率且不低于同行业平均值。",
            "各指标且均需不低于同行业平均值。",
            "研发投入强度不低于7.60%。",
            "BENCHMARK_CONDITION_SCOPE_EXPANDED",
        ),
        (
            "审议核心技术人员调整。表决通过。本议案无需提交股东会。",
            "召开股东会审议核心技术人员调整等议案。",
            "核心技术人员调整无需提交股东会。",
            "APPROVAL_SCOPE_MERGED",
        ),
        (
            "2025年营业收入12.58亿元，2025年目标12亿元，2026年目标16亿元。",
            "营业收入12.58亿元，达到目标值16亿元的要求。",
            "2025年营业收入12.58亿元，达到当年目标12亿元。",
            "NUMERIC_COMPARISON_CONTRADICTION",
        ),
        (
            "代表股份18,138,687股。",
            "代表股份1.8139亿股。",
            "代表股份18,138,687股。",
            "AMOUNT_OR_SHARE_CONVERSION_UNSUPPORTED",
        ),
    ],
)
def test_known_contradictions_and_supported_counterexamples(source, wrong, right, code):
    assert code in semantic_issues(wrong, source, source)
    assert code not in semantic_issues(right, source, source)


def document(body):
    return {
        "id": "doc",
        "title": "公告",
        "body": body,
        "date": "2026-10-01",
        "kind": "ANNOUNCEMENT",
        "codes": ["000001.SZ"],
        "body_scope": "VERIFIED_EXCERPT",
        "body_truncated": True,
        "gold": "不得发送的评审答案",
    }


def test_body_and_collapsed_blank_cells_are_preserved_without_guessing():
    body = "实际投入金额\r\n实际收回本金\r\n实际收益\r\n尚未收回本金\r\n合计 85.73 8,324.58\r\n"
    source = document(body)
    catalog = organized_catalog(body)
    assert "".join(row["text"] for row in catalog.values()) == body
    request = structured_request(source)
    serialized = json.loads(request["messages"][1]["content"])
    assert serialized["documents"][0]["body_truncated"] is True
    assert "不得发送的评审答案" not in str(request)
    assert "COLLAPSED_TOTAL_COLUMNS_UNVERIFIED" in semantic_issues("实际投入85.73万元。", body, body)


def test_binding_cannot_borrow_numbers_or_subjects_from_other_quote():
    source = document("公司尚未购买股份。\n" + "原文条件保留。" * 80)
    first = next(iter(organized_catalog(source["body"])))
    event = {
        "bindings": [{"kind": "quantity", "quote_id": first, "text": "已购买五百万股"}],
        "table_status": "NOT_USED",
        "title": "员工持股",
        "assessment": "NEUTRAL",
        "stage": "进行中",
        "quote_ids": [first],
        "meaning": "截至公告日尚未购买股份。",
        "mechanism": "计划执行仍需观察。",
        "caveat": "本次仅有摘录。",
    }
    checked = check_structured({"items": [{"id": "doc", "events": [event]}]}, source)
    assert "LITERAL_BINDING_NOT_SUPPORTED" in checked["checks"][0]["issues"]
    assert checked["semantic_verified"] is False
    assert checked["events"][0]["meaning"] == event["meaning"]


def manager_source():
    interval = {
        "start": "2024-01-30",
        "end": "2026-10-09",
        "status": "AVAILABLE",
        "return_pct": 365.0838,
        "max_drawdown_pct": 45.8039,
        "positive_20d_pct": 60.5096,
    }
    return {
        "as_of": "2026-10-10T11:43:00+08:00",
        "facts": {
            "manager:change": {"text": "离任原因 个人原因\n转任本公司其他工作岗位\n的说明 -"},
            "manager:performance": {"text": "完整资料", "metrics": [{"name": "陆阳", "tenure": interval}]},
        },
    }


def test_manager_transfer_unknown_and_full_period_are_not_filled_as_subinterval():
    source = manager_source()
    packet = manager_input(source)
    assert packet["facts"] == source["facts"]
    assert len(packet["metric_scope_catalog"]) == 1
    scope = packet["metric_scope_catalog"][0]
    good = {
        "change_reason": "公开理由为个人原因，转任栏没有说明，不能确认已转任。",
        "transfer_status": "NOT_DISCLOSED",
        "performance_assessment": "参与管理全期包括共同与独立管理，缺少共同管理子区间指标。",
        "metric_scopes": [
            {
                k: scope[k]
                for k in (
                    "manager",
                    "source_field",
                    "start",
                    "end",
                    "return_pct",
                    "max_drawdown_pct",
                    "positive_20d_pct",
                )
            }
        ],
    }
    assert manager_issues(good, source) == []
    bad = {
        **good,
        "change_reason": "因个人原因解聘，转任其他岗位。",
        "transfer_status": "DISCLOSED_OTHER_ROLE",
        "performance_assessment": "共同管理期间回撤为45.80%。",
    }
    assert {
        "TRANSFER_NOT_DISCLOSED",
        "TRANSFER_ASSERTED_WITHOUT_SOURCE",
        "CO_MANAGED_SUBINTERVAL_METRICS_UNAVAILABLE",
    } <= set(manager_issues(bad, source))


def test_lossless_packet_preserves_different_documents_and_all_qualifiers():
    docs = [
        document("若满足条件，公司拟新增主体，尚需审议。"),
        {**document("该方案不改变原有主体；原文未提供数值。"), "id": "other", "title": "后续公告"},
    ]
    quotes = [docs[0]["body"], docs[1]["body"]]
    facts = {
        "F1": {
            "category": "公告",
            "text": "\n".join(f"{d['date']}《{d['title']}》：{q}" for d, q in zip(docs, quotes, strict=True)),
            "quote_receipts": [{"document_id": d["id"], "quote": q} for d, q in zip(docs, quotes, strict=True)],
            "direction_eligible": False,
            "body_truncated": True,
        }
    }
    packet = fact_packet(facts, docs)
    parts = packet["facts"]["F1"]["text_parts"]
    rebuilt = "\n".join(
        f"{packet['sources'][p['source']]['date']}《{packet['sources'][p['source']]['title']}》：{packet['quote_pool'][p['quote']]}"
        for p in parts
    )
    assert rebuilt == facts["F1"]["text"]
    assert packet["facts"]["F1"]["direction_eligible"] is False
    assert packet["facts"]["F1"]["body_truncated"] is True
    with pytest.raises(ValueError, match="SOURCE_MISSING"):
        fact_packet(facts, docs[:1])


def test_token_budget_requires_calibration_and_keeps_hard_limit():
    budget = LocalTokenBudget.__new__(LocalTokenBudget)
    budget.calibrated = False
    budget.count = len
    with pytest.raises(ValueError, match="NOT_CALIBRATED"):
        budget.request("规则", {}, {})
    request = {"messages": [{"content": "短文本"}]}
    with pytest.raises(ValueError, match="CALIBRATION_FAILED"):
        budget.calibrate([(request, {"done": True, "prompt_eval_count": 2})] * 2)
    budget.calibrate([(request, {"done": True, "prompt_eval_count": 10})] * 2)
    with pytest.raises(ValueError, match="TOO_LARGE_TOKEN_BUDGET"):
        budget.request("规则", {"body": "全文" * 40000}, {})


def test_pending_shareholder_approval_is_separate_from_completed_board_vote():
    source = "董事会已审议通过，本议案尚需提交股东会审议批准。"
    assert "PENDING_APPROVAL_ASSERTED_COMPLETED" in semantic_issues("股东会已批准该事项。", source)
    assert "PENDING_APPROVAL_ASSERTED_COMPLETED" not in semantic_issues("董事会已通过，尚需股东会审议。", source)


def test_share_units_in_printed_brackets_cannot_be_expanded_tenfold():
    source = "代表股份数为【18,138,687】股，占比【1.3355】%。"
    assert "AMOUNT_OR_SHARE_CONVERSION_UNSUPPORTED" in semantic_issues("代表股份1.8139亿股。", source)
    assert semantic_issues("代表股份18,138,687股。", source) == []
