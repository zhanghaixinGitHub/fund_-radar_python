"""验证跨页关联只按可定位的版面条件，不因答案需要某个数字而扩展引用。"""

from copy import deepcopy

from scripts.local_announcement_continuations import link_continuations
from scripts.local_announcement_document import text_document
from scripts.local_announcement_structured_reader import check_structured, structured_request
from scripts.local_announcement_tasks import plan_question


def unit(identifier, text, page, kind="text"):
    return {"id": identifier, "text": text, "page": page, "kind": kind}


def test_number_and_unit_across_pages_keeps_bound_and_original_text():
    units = [unit("P1B", "截至公告担保余额不超过人民币18.72", 1),
             unit("P1F", "1", 1, "page_layout"), unit("P2H", "公告", 2, "page_layout"),
             unit("P2B", "亿元，无逾期担保。", 2)]
    before = deepcopy(units)
    linked = link_continuations(units)
    assert linked[0]["continuation_ids"] == ["P2B"]
    assert linked[-1]["continuation_ids"] == ["P1B"]
    assert units == before
    parsed = {"answers": [{"question_number": 1, "answer": "18.72亿元", "status": "answered", "evidence_ids": ["P2B"]}]}
    answer = check_structured(parsed, linked, ["担保余额是多少？"], "FULL_TEXT")["answers"][0]
    assert "BOUND_QUALIFIER_REQUIRES_REVIEW:18.72" in answer["issues"]
    assert answer["safe_answer"] is None
    assert answer["model_evidence_ids"] == ["P2B"]
    assert answer["context_evidence_ids"] == ["P1B"]


def test_bracket_continuation_but_not_an_arbitrary_nearby_paragraph():
    linked = link_continuations([unit("P1", "对子公司（含", 1), unit("P2", "子公司之间）不超过4亿元。", 2)])
    assert linked[0]["continuation_ids"] == ["P2"]
    for left, right in [
        (unit("P1", "授信额度17.5", 1), unit("P3", "亿元。", 3)),
        (unit("P1", "本段没有结束标点", 1), unit("P2", "下页出现17.5亿元。", 2)),
        (unit("P1", "括号（未结束", 1), unit("P2", "二、其他事项（说明）", 2)),
        (unit("P1", "17.5", 1, "table_row"), unit("P2", "亿元。", 2)),
        (unit("P1", "17.5。", 1), unit("P2", "亿元是单位。", 2)),
    ]:
        assert all("continuation_ids" not in u for u in link_continuations([left, right]))


def test_ids_first_is_opt_in_and_cannot_send_evaluation_labels():
    packet = text_document("拟投资82万元，尚需审批。", scope="FULL_TEXT")
    packet["gold"] = "不能发送给模型的标准答案"
    baseline, _ = structured_request("公告", packet, ["投资计划？"])
    candidate, _ = structured_request("公告", packet, ["投资计划？"], citation_mode="ids_first")
    fields = candidate["format"]["properties"]["answers"]["items"]["properties"]
    assert list(fields) == ["question_number", "evidence_ids", "answer", "status"]
    assert baseline["messages"] != candidate["messages"]
    assert baseline["options"]["num_predict"] == candidate["options"]["num_predict"]
    assert "不能发送给模型" not in str(candidate)


def test_balance_quote_experiment_does_not_change_existing_mode():
    question = "担保总额度调整后是多少，截至公告实际担保余额是多少，两者是否相同？"
    old = plan_question(question, focus_details=True)
    assert old["questions"][1] == (
        "截至公告实际担保余额是多少？只答本项金额，逐字保留原文金额前的限定词，不展开其他金额。"
    )
    candidate = plan_question(question, focus_details=True, quote_balance=True)
    assert "逐字引用" in candidate["questions"][1]
    assert "完整原句" in candidate["questions"][1]
    assert old["questions"][0] == candidate["questions"][0]
    assert old["questions"][2] == candidate["questions"][2]
