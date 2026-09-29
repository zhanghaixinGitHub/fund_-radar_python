"""策略文字只增输入，不改变基金目标、日期、数值或答案。"""

import pytest
from scripts.fund_002112_strategy_candidate import TAGS, append_strategy


@pytest.fixture
def row():
    return {
        "fund_code": "002112",
        "target": "2023-04-18",
        "x": list(range(20)),
        "report_sha256": "a" * 64,
        "actual_direction": "DOWN",
    }


@pytest.mark.parametrize("tag", TAGS)
def test_preserve_inputs_and_encode_explicit_unknown(row, tag):
    r = append_strategy(row, {"tag": tag, "published_date": "2023-04-17", "report_sha256": "a" * 64})
    assert r["x"][:20] == row["x"] and r["actual_direction"] == "DOWN"
    assert sum(r["x"][20:]) == 1 and r["x"][20 + TAGS.index(tag)] == 1
    assert len(row["x"]) == 20


@pytest.mark.parametrize("published", ["2023-04-18", "2023-04-19"])
def test_same_day_and_future_reports_cannot_leak(row, published):
    with pytest.raises(ValueError, match="PUBLICATION"):
        append_strategy(row, {"tag": "MEDICINE_FOCUS", "published_date": published, "report_sha256": "a" * 64})


def test_cannot_substitute_another_report(row):
    with pytest.raises(ValueError, match="IDENTITY"):
        append_strategy(row, {"tag": "MEDICINE_FOCUS", "published_date": "2023-04-17", "report_sha256": "b" * 64})
