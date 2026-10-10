"""最终版本预算、原件保护和解释复核的离线边界。"""

import json
from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from app.integrations.fund_information_analysis import Budget, ResponseFormatError
from scripts import fund_002112_analysis_replay_v5 as replay


def test_total_budget_counts_engineering_diagnostics():
    assert sum(replay.LIMITS.values()) + 172 == 420


def test_selection_remains_fixed_despite_label_changes():
    targets = [str(date(2026, 6, 1) + timedelta(days=i)) for i in range(80)]
    rows = [{"target": target, "phase": "MORNING_0830", "label": "UP"} for target in targets]
    before = replay.select_targets(rows, targets[-20:])
    for row in rows:
        row["label"] = "DOWN"
    assert replay.select_targets(rows, targets[-20:]) == before
    assert before["expanded"] == targets[-60:-20]


def test_no_scoring_or_separate_review_before_all_results(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "OUT", tmp_path)
    monkeypatch.setattr(replay, "checked_protocol", lambda: {"targets": ["2026-08-26"]})
    monkeypatch.setattr(replay.data, "load_sources", lambda: pytest.fail("不得提前评分"))
    with pytest.raises(ValueError, match="ALL_PREDICTIONS_REQUIRED"):
        replay.score()
    with pytest.raises(ValueError, match="ALL_PREDICTIONS_REQUIRED"):
        replay.review()


def test_invalid_inner_json_preserves_raw_response_and_counts_call(tmp_path, monkeypatch):
    """测试替身使用公开假配置，不连接数据库，不请求任何外部服务。"""
    from datetime import datetime, timedelta

    monkeypatch.setattr(replay, "OUT", tmp_path)
    raw = {"choices": [{"finish_reason": "stop", "message": {"content": '{"broken":'}}], "usage": {"total_tokens": 5}}

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            text = json.dumps(raw)
            return SimpleNamespace(status_code=200, content=text.encode(), text=text, json=lambda: raw)

    monkeypatch.setattr(replay.httpx, "Client", Client)
    requests = object.__new__(replay.LocalRequests)
    requests.settings = SimpleNamespace(
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="test",
        deepseek_api_key=SimpleNamespace(get_secret_value=lambda: "test-not-a-secret"),
    )
    requests.calls = requests.cohort_calls = 0
    requests.cohort = "regression"
    requests.target = "2026-08-26"
    with pytest.raises(ResponseFormatError):
        requests.direct(
            "test", {"fund_code": "002112"}, "SYNTHESIS", Budget(datetime.now().astimezone() + timedelta(minutes=1))
        )
    assert requests.calls == 1 and requests.cohort_calls == 1
    folder = next((tmp_path / "calls").iterdir())
    assert replay.data.read(folder / "raw-response.json") == raw
    assert (folder / "raw-http-response.txt").read_text(encoding="utf-8") == json.dumps(raw)
    assert replay.data.read(folder / "failure.json")["type"] == "ResponseFormatError"
    assert "Authorization" not in (folder / "request.json").read_text(encoding="utf-8")
