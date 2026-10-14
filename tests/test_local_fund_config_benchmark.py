"""四组对照的公平性、首次响应保护和未知调用结果边界。"""

from copy import deepcopy

import pytest
from scripts.local_fund_config_benchmark import CONFIGS, configured_request, once
from scripts.local_fund_repair_run import save


class Budget:
    def content_tokens(self, request):
        return sum(len(m["content"]) for m in request["messages"])

    def count(self, text):
        return len(text)


def original():
    return {
        "model": "qwen3.5:4b",
        "stream": False,
        "think": False,
        "messages": [{"role": "user", "content": "年均可分配利润；完整资料不裁剪。"}],
        "format": {"type": "object"},
        "options": {"temperature": 0, "num_predict": 4096},
    }


def test_only_frozen_configuration_changes_and_source_is_unchanged():
    source = original()
    before = deepcopy(source)
    requests = {c: configured_request(source, c, Budget()) for c in CONFIGS}
    assert source == before
    for request in requests.values():
        assert request["messages"] == source["messages"]
        assert request["format"] == source["format"]
        assert request["options"]["num_predict"] == 8192
    off, on = requests["off_zero"], requests["on_zero"]
    assert off["options"] == on["options"]
    assert off["think"] is False and on["think"] is True
    assert len({r["options"]["num_ctx"] for r in requests.values()}) == 1


def test_over_budget_fails_without_truncating_source():
    source = original()
    source["messages"][0]["content"] = "字" * 65536
    with pytest.raises(ValueError, match="OVER_BUDGET"):
        configured_request(source, "on_recommended", Budget())
    assert len(source["messages"][0]["content"]) == 65536


def test_incomplete_response_is_retained_and_never_retried(tmp_path):
    calls = []

    def transport(request):
        calls.append(request)
        return {"done_reason": "length", "message": {"content": "{"}}

    request = configured_request(original(), "on_zero", Budget())
    first = once(tmp_path, "case--on_zero", request, transport)
    assert "INCOMPLETE" in first["error"]
    assert once(tmp_path, "case--on_zero", request, transport) == first
    assert len(calls) == 1
    changed = deepcopy(request)
    changed["think"] = False
    with pytest.raises(ValueError, match="FROZEN_REQUEST_CHANGED"):
        once(tmp_path, "case--on_zero", changed, transport)


def test_unknown_previous_outcome_blocks_before_transport(tmp_path):
    save(tmp_path / "raw" / "case-started.json", {})
    with pytest.raises(ValueError, match="PREVIOUS_CALL_OUTCOME_UNKNOWN"):
        once(tmp_path, "case", original(), lambda _: pytest.fail("不能重发"))


def test_transport_error_is_saved_and_resume_stays_blocked(tmp_path):
    def fail(_):
        raise TimeoutError("运行器可能仍在处理")

    with pytest.raises(ValueError, match="TRANSPORT_OUTCOME_UNKNOWN"):
        once(tmp_path, "case", original(), fail)
    with pytest.raises(ValueError, match="TRANSPORT_OUTCOME_UNKNOWN"):
        once(tmp_path, "case", original(), lambda _: pytest.fail("不能重发"))
