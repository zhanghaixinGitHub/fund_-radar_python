"""生成后的独立拦截：证据与草稿不可被校验过程静默改写。"""

import pytest
from scripts.local_announcement_checks import final_checks, literal_chinese_counts


def item(text, quotes, issues=None):
    return {"answer": text, "quotes": quotes, "issues": issues or []}


def test_unsupported_causal_link_is_withheld_not_rewritten():
    raw = item("分红导致激励对象人数下降。", ["因人员离职，激励人数减少；分红后价格调整。"])
    result = final_checks("人数调整原因？", raw)
    assert result["answer"] == raw["answer"]
    assert result["safe_answer"] is None
    assert "CAUSAL_PARAPHRASE_REQUIRES_REVIEW" in result["issues"]
    assert not raw["issues"]


def test_direct_cause_and_correct_paraphrase_have_distinct_admission():
    source = "因工作内容调整，不再认定为核心技术人员，仍在职。"
    assert final_checks("为何调整？", item("因工作内容调整。", [source]))["safe_answer"]
    assert final_checks("为何调整？", item("由于工作内容变动。", [source]))["safe_answer"] is None


@pytest.mark.parametrize("amount", ["34.01", "89.72"])
def test_upper_bound_must_not_become_actual_balance(amount):
    source = f"实际担保余额不超过人民币{amount}亿元。"
    assert final_checks("担保余额是多少？", item(f"余额{amount}亿元。", [source]))["safe_answer"] is None
    assert final_checks("担保余额是多少？", item(f"余额不超过{amount}亿元。", [source]))["safe_answer"]


def test_upper_bound_question_and_unrelated_number_do_not_trigger():
    assert final_checks("额度上限是多少？", item("100万元。", ["不超过100万元。"]))["safe_answer"]
    assert final_checks("注册资本多少？", item("50万元。", ["不超过100万元；注册资本50万元。"]))["safe_answer"]


def test_chinese_explicit_counts_can_clear_false_positive_but_not_infer_names():
    assert literal_chinese_counts("九名董事，三名独立董事，一名职工董事，十二个月。") == {"9", "3", "1", "12"}
    raw = item("9名董事、3名独立董事。", ["九名董事，其中独立董事三名。"], ["NUMBER_NOT_IN_EVIDENCE:3,9"])
    assert not final_checks("董事多少？", raw)["issues"]
    raw = item("3人。", ["人员为甲、乙、丙。"], ["NUMBER_NOT_IN_EVIDENCE:3"])
    assert final_checks("人数多少？", raw)["safe_answer"] is None


def test_total_and_two_parts_must_sum_but_not_arbitrary_partial_lists():
    raw = item("总额100万元；项目投入100万元，费用5万元。", ["总额100万元，项目95万元，费用5万元。"])
    assert "COMPONENTS_DO_NOT_SUM" in final_checks("总金额及两部分是多少？", raw)["issues"]
    assert final_checks("列举部分费用？", raw)["safe_answer"]
    good = item("总额100万元；项目投入95万元，费用5万元。", raw["quotes"])
    assert final_checks("总金额及两部分是多少？", good)["safe_answer"]


def test_benchmark_condition_is_not_lost_in_numeric_threshold():
    source = "考核要求增长不低于33%，且不低于同行业平均值。"
    raw = item("考核要求增长不低于33%。", [source])
    assert "BENCHMARK_CONDITION_MISSING" in final_checks("考核要求是什么？", raw)["issues"]
    raw = item("增长不低于33%，且不低于同行业平均值。", [source])
    assert final_checks("考核要求是什么？", raw)["safe_answer"]


def test_mixed_market_rules_require_review_even_when_quotes_omit_market_names():
    body = "依据香港联交所规则及深圳证券交易所规则。年度报告前有多个窗口。"
    raw = item("窗口为30日。", ["年度报告前有多个窗口。"])
    checked = final_checks("禁止买卖窗口多久？", raw, body)
    assert "MULTI_MARKET_RULES_REQUIRES_REVIEW" in checked["issues"]
    assert checked["answer"] == raw["answer"]
    assert final_checks("公告日期？", raw, body)["safe_answer"]


def test_full_source_pending_procedure_check_can_catch_omitted_quote():
    body = "尚需签订出让合同并缴纳价款。尚需办理项目备案、规划许可及施工许可。"
    raw = item("还需签订出让合同并缴纳价款。", ["尚需签订出让合同并缴纳价款。"])
    result = final_checks("后续还需哪些手续？", raw, body)
    assert "PENDING_PROCEDURES_REQUIRES_REVIEW:项目备案,规划许可,施工许可" in result["issues"]
    assert result["safe_answer"] is None
