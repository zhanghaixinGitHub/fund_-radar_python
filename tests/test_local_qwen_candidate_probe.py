"""验证9B旧题可比性、全文不截断及首次调用边界；不证明模型理解正确。"""

import urllib.error

import pytest
from scripts.local_fund_repair_run import read, save
from scripts.local_qwen_candidate_probe import MODEL, clone_old, natural_request, once


def setup_run(tmp_path):
    req = {"model": MODEL, "messages": [], "options": {"num_ctx": 16384}}
    save(tmp_path / "requests.json", {"test": req})
    save(tmp_path / "protocol.json", {"max_calls": 1})
    return req


def test_old_task_changes_only_model():
    request = {
        "model": "qwen3.5:4b",
        "messages": [{"role": "user", "content": "完整原文\n空列和条件"}],
        "options": {"seed": 42, "num_ctx": 32768},
        "think": False,
    }
    revised = clone_old(request)
    assert request["model"] == "qwen3.5:4b"
    assert revised["model"] == MODEL
    assert {k: v for k, v in revised.items() if k != "model"} == {k: v for k, v in request.items() if k != "model"}


def test_full_body_is_not_truncated_and_no_fact_copy_contract():
    class Budget:
        def count(self, text):
            return len(text)

    document = {
        "id": "case",
        "title": "题名",
        "date": "2026-04-01",
        "body_scope": "FULL_TEXT",
        "body_truncated": False,
        "body": "条件且条件\n最后一行原文",
    }
    request = natural_request(document, ["请保留条件"], Budget(), {"options": {"seed": 42}})
    assert document["body"] in request["messages"][1]["content"]
    assert "data_json" not in str(request)


def test_completed_first_response_is_reused_without_calling_again(tmp_path):
    setup_run(tmp_path)
    calls = []

    def transport(request):
        calls.append(request)
        return {
            "model": MODEL,
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 10,
            "message": {"content": "首次答案"},
        }

    first = once(tmp_path, "test", transport)
    assert once(tmp_path, "test", transport) == first
    assert len(calls) == 1


def test_unknown_first_transport_is_never_retried(tmp_path):
    setup_run(tmp_path)

    def transport(_):
        raise TimeoutError("unknown")

    with pytest.raises(ValueError, match="UNKNOWN_NO_RETRY"):
        once(tmp_path, "test", transport)
    with pytest.raises(ValueError, match="OUTCOME_UNKNOWN"):
        once(tmp_path, "test", transport)
    assert read(tmp_path / "raw/test.json")["transport_unknown"] is True


def test_explicit_server_rejection_is_retained_as_runtime_failure(tmp_path):
    import io

    setup_run(tmp_path)

    def transport(_):
        raise urllib.error.HTTPError(
            "http://127.0.0.1:11434/api/chat", 500, "OOM", None, io.BytesIO(b"not enough memory")
        )

    raw = once(tmp_path, "test", transport)
    assert raw["server_error"]["status"] == 500
    assert "response" not in raw
    assert raw["errors"] == ["SERVER_REJECTED_FIRST_REQUEST"]
