"""输入对照的完整覆盖、公平预算和首次失败保留边界。"""

from copy import deepcopy

import pytest
from scripts.local_fund_input_ablation import MODEL, once, request_pair, summarize
from scripts.local_fund_repair_run import save


class Budget:
    def content_tokens(self, request):
        return sum(len(message["content"]) for message in request["messages"])


def task():
    body = "备案不代表对真\n实性的认定。表头 年均利润\n（空列原样保留）"
    catalog = {
        "Q0001": {"text": body[:12], "start": 0, "end": 12},
        "Q0002": {"text": body[12:], "start": 12, "end": len(body)},
    }
    import json

    return {
        "id": "doc-27",
        "gold": "不要发送的正确答案",
        "document": {
            "id": "DOC",
            "title": "测试",
            "date": "2026-09-28",
            "body": body,
            "body_scope": "FULL_TEXT",
            "body_truncated": False,
        },
        "request": {"messages": [{}, {"content": json.dumps({"quote_catalogs": {"DOC": catalog}})}]},
    }


def setup_once(root, request, maximum=2):
    save(root / "requests.json", {"case--full": request})
    save(root / "protocol.json", {"max_calls": maximum})


def response(reason="stop"):
    return {
        "model": MODEL,
        "done": True,
        "done_reason": reason,
        "prompt_eval_count": 50,
        "eval_count": 20,
        "message": {"content": "首次答案"},
    }


def test_complete_body_common_questions_and_options_without_gold_or_output_caps():
    original = task()
    before = deepcopy(original)
    pair = request_pair(original, Budget())
    assert original == before
    assert original["document"]["body"] in pair["full"]["messages"][1]["content"]
    for value in pair.values():
        assert "不要发送的正确答案" not in str(value)
        assert "format" not in value
    assert pair["full"]["options"] == pair["segmented"]["options"]
    assert pair["full"]["messages"][0] == pair["segmented"]["messages"][0]
    assert (
        pair["full"]["messages"][1]["content"].split("<公告资料>")[0]
        == pair["segmented"]["messages"][1]["content"].split("<公告资料>")[0]
    )


def test_catalog_missing_a_character_blocks_pair():
    value = task()
    value["document"]["body"] += "新增而目录没有的字"
    with pytest.raises(ValueError, match="SOURCE_COVERAGE"):
        request_pair(value, Budget())


def test_pair_over_budget_does_not_truncate_or_call():
    value = task()

    class HugeBudget:
        def content_tokens(self, request):
            return 65536

    with pytest.raises(ValueError, match="OVER_BUDGET"):
        request_pair(value, HugeBudget())
    assert value == task()


def test_first_incomplete_response_is_reused_without_resending(tmp_path):
    request = request_pair(task(), Budget())["full"]
    setup_once(tmp_path, request)
    calls = []

    def transport(payload):
        calls.append(payload)
        return response("length")

    first = once(tmp_path, "case--full", request, transport)
    assert "LOCAL_GENERATION_INCOMPLETE" in first["error"]
    assert once(tmp_path, "case--full", request, transport) == first
    assert len(calls) == 1


def test_unknown_started_call_and_transport_error_are_never_resent(tmp_path):
    request = request_pair(task(), Budget())["full"]
    setup_once(tmp_path, request)
    save(tmp_path / "raw/case--full-started.json", {})
    with pytest.raises(ValueError, match="PREVIOUS_CALL_OUTCOME_UNKNOWN"):
        once(tmp_path, "case--full", request, lambda _: pytest.fail("不能再发送"))


def test_saved_timeout_blocks_resume(tmp_path):
    request = request_pair(task(), Budget())["full"]
    setup_once(tmp_path, request)

    def timeout(_):
        raise TimeoutError("服务端结果未知")

    with pytest.raises(ValueError, match="TRANSPORT_OUTCOME_UNKNOWN"):
        once(tmp_path, "case--full", request, timeout)
    with pytest.raises(ValueError, match="TRANSPORT_OUTCOME_UNKNOWN"):
        once(tmp_path, "case--full", request, lambda _: pytest.fail("不能再发送"))


def test_unfrozen_request_or_exceeded_call_cap_blocks_before_network(tmp_path):
    request = request_pair(task(), Budget())["full"]
    setup_once(tmp_path, request, maximum=0)
    with pytest.raises(ValueError, match="CALL_CAP_EXCEEDED"):
        once(tmp_path, "case--full", request, lambda _: pytest.fail("不能调用"))
    changed = deepcopy(request)
    changed["think"] = True
    with pytest.raises(ValueError, match="REQUEST_NOT_FROZEN"):
        once(tmp_path, "case--full", changed, lambda _: pytest.fail("不能调用"))


def test_context_exhaustion_cannot_be_marked_a_complete_answer(tmp_path):
    request = request_pair(task(), Budget())["full"]
    setup_once(tmp_path, request)
    value = response()
    value["prompt_eval_count"] = request["options"]["num_ctx"] - 10
    first = once(tmp_path, "case--full", request, lambda _: value)
    assert "LOCAL_CONTEXT_EXHAUSTED" in first["error"]
    assert first["semantic_verified"] is False


def test_progress_is_append_only_and_never_overwrites_earlier_summary(tmp_path):
    request = request_pair(task(), Budget())["full"]
    setup_once(tmp_path, request)
    save(tmp_path / "queue.json", [{"name": "case--full", "arm": "full"}])
    save(tmp_path / "summary.json", {"original_first_checkpoint": True})
    once(tmp_path, "case--full", request, lambda _: response())
    first = summarize(tmp_path)
    assert summarize(tmp_path) == first
    assert (tmp_path / "summary.json").read_text(encoding="utf-8").find("original_first_checkpoint") >= 0
    assert len(list((tmp_path / "progress").glob("*.json"))) == 1
