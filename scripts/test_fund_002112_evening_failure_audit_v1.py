"""排查统计口径检查：不把改错方向、名称变更或未披露持仓误当成原因。"""

import numpy as np
import pytest

from scripts import fund_002112_evening_failure_audit_v1 as m


def test_removal_renormalizes_remaining_branches():
    result = m.probabilities([0.9, 0, 0.1], [0.3, 0, 0.7], [0.6, 0, 0.4])
    assert np.allclose(result["full"], [0.6, 0, 0.4])
    assert np.allclose(result["without_information"], [0.4, 0, 0.6])
    assert all(np.isclose(v.sum(), 1) for v in result.values())


def test_changed_wrong_guess_does_not_count_as_spoiled_correct_guess():
    rows = [
        {"label": "UP", "probabilities": {"before": [1, 0, 0], "after": [0, 1, 0]}},
        {"label": "UP", "probabilities": {"before": [1, 0, 0], "after": [0, 0, 1]}},
        {"label": "UP", "probabilities": {"before": [0, 0, 1], "after": [1, 0, 0]}},
    ]
    result = m.flips(rows, "before", "after")
    assert result["both_wrong_changed"] == 1
    assert result["corrected"] == result["spoiled"] == 1 and result["net_correct"] == 0


def test_unreported_holdings_remain_unknown():
    assert m.overlap(None, {"A": 1}) is None
    assert m.overlap({"A": 1}, {"B": 1}) == 0
    assert m.overlap({"A": 0.7, "B": 0.3}, {"A": 0.4, "C": 0.6}) == pytest.approx(0.4)


def test_top_ten_uses_code_and_not_full_report_tail_or_name():
    report = {
        "holdings": [
            {"stock_code": str(i), "reported_rank": i, "stock_name": "旧名", "nav_weight_pct": "2"}
            for i in range(1, 12)
        ]
    }
    vector = m.top_ten(report)
    report["holdings"][0]["stock_name"] = "更名"
    report["holdings"][-1]["nav_weight_pct"] = "80"
    assert vector == m.top_ten(report) and "11" not in vector
    assert sum(vector.values()) == pytest.approx(1)


def test_training_guard_rejects_fit():
    with pytest.raises(RuntimeError, match="NO_TRAINING"):
        m.forbid_fit()
