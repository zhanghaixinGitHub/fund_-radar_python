"""三年回放的时间范围、失败分母和有限调用边界，不请求服务或其他基金。"""

from datetime import datetime, timedelta

import pytest
from app.integrations.fund_information_analysis import Budget
from scripts import fund_002112_analysis_three_year_v1 as replay


def test_window_is_complete_and_not_selected_by_old_accuracy():
    sessions = ["2023-09-28", "2023-10-09", "2024-01-02", "2025-01-02", "2026-09-30", "2026-10-09"]
    assert replay.selected_sessions(sessions) == sessions[1:5]


def test_failed_and_missing_dates_remain_in_planned_denominator():
    rows = [{"new": "UP", "actual": "UP"}, {"new": None, "actual": "DOWN"}, {"new": None, "actual": "UP"}]
    metric = replay.metrics(rows, "new")
    assert metric["planned"] == 3 and metric["judged"] == 1 and metric["accuracy"] == 1
    assert metric["correct_per_planned"] == 1 / 3


def test_missing_old_model_does_not_become_wrong_or_guessed_prediction():
    m = replay.metrics([{"A_NAV": None, "actual": "DOWN"}], "A_NAV")
    assert m["accuracy"] is None and m["judged"] == 0


def test_separate_direction_counts_include_flat_without_turning_it_down():
    m = replay.metrics(
        [{"new": "UP", "actual": "UP"}, {"new": "DOWN", "actual": "FLAT"}, {"new": "UP", "actual": "DOWN"}], "new"
    )
    assert m["correct"] == 1
    assert m["by_actual"]["FLAT"] == {"total": 1, "correct": 0}
    assert m["by_actual"]["DOWN"] == {"total": 1, "correct": 0}


def test_no_score_or_review_before_all_dates_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "OUT", tmp_path)
    monkeypatch.setattr(replay, "checked_protocol", lambda: {"targets": ["2024-01-02"]})
    monkeypatch.setattr(replay.data, "load_sources", lambda: pytest.fail("不能提前读取答案"))
    with pytest.raises(ValueError, match="ALL_PREDICTIONS_REQUIRED"):
        replay.score()
    with pytest.raises(ValueError, match="ALL_PREDICTIONS_REQUIRED"):
        replay.review("2024")


def test_year_budgets_sum_to_frozen_cap_and_reserve_review():
    assert sum(replay.LIMITS.values()) == 1800
    requests = object.__new__(replay.Requests)
    requests.year = "2024"
    requests.protocol = {"review_reserve": {"2024": 49}}
    requests.calls = replay.LIMITS["2024"] - 49
    # 在接触配置或网络前触发预算，保证生成不会吃掉逐日复核份额。
    with pytest.raises(ValueError, match="CALL_LIMIT"):
        requests.direct("unused", {}, "SYNTHESIS", Budget(datetime.now().astimezone() + timedelta(minutes=1)))
