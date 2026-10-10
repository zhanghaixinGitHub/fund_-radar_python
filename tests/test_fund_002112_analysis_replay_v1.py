"""回放边界测试：不把目标答案传给分析，不把失败样本从总数中隐藏。"""

from copy import deepcopy

import pytest
from scripts.fund_002112_analysis_replay_v1 import metrics, validate_input


@pytest.fixture
def data():
    return {
        "as_of": "2026-09-29T08:30:00+08:00",
        "window": {"base_nav_date": "2026-09-28", "target_nav_date": "2026-09-29"},
        "latest_nav_date": "2026-09-28",
        "nav": [{"nav_date": "2026-09-28", "unit_nav": "1.2"}],
        "report": {"available_at": "2026-09-01T08:00:00+08:00"},
        "time_proof": {
            "nav_max_available_at": "2026-09-29T08:00:00+08:00",
            "quote_available_at": "2026-09-28T18:00:00+08:00",
        },
        "companies": [{"quote": {"date": "2026-09-28"}}],
        "market": {"index": {"date": "2026-09-28"}},
        "documents": [{"date": "2026-09-28", "available_at": "2026-09-29T00:00:00+08:00"}],
    }


def test_known_past_input_is_accepted(data):
    validate_input(data)


@pytest.mark.parametrize("where", ["nav", "report", "document", "market", "answer"])
def test_future_and_answer_are_rejected(data, where):
    value = deepcopy(data)
    if where == "nav":
        value["nav"].append({"nav_date": "2026-09-29", "unit_nav": "1.3"})
    elif where == "report":
        value["report"]["available_at"] = "2026-09-29T09:00:00+08:00"
    elif where == "document":
        value["documents"][0]["available_at"] = "2026-09-29T09:00:00+08:00"
    elif where == "market":
        value["market"]["index"]["date"] = "2026-09-29"
    else:
        value["facts"] = {"answer": {"actual_direction": "UP"}}
    with pytest.raises(AssertionError):
        validate_input(value)


def test_failed_and_flat_targets_are_not_erased():
    result = metrics(
        [{"actual": "DOWN", "new": "DOWN"}, {"actual": "UP", "new": None}, {"actual": "FLAT", "new": "UP"}], "new"
    )
    assert result["planned"] == 3
    assert result["judged"] == 2
    assert result["accuracy"] == 0.5
    assert result["correct_per_planned"] == 1 / 3
    assert result["by_actual"]["FLAT"] == {"total": 1, "correct": 0}
