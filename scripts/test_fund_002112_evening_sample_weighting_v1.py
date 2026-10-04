"""检验历史时点、未知持仓和比较统计边界，不额外拟合模型。"""

import numpy as np
import pytest

from scripts import fund_002112_evening_sample_weighting_v1 as m


def row(day, as_of=None, mature=None):
    return {
        "target": day,
        "as_of": as_of or day + "T00:00:00+08:00",
        "label_mature_at": mature or day + "T20:00:00+08:00",
        "label": "UP",
        "groups": {"history": [1]},
    }


def report(day, code):
    return {
        "report_end": day,
        "available_at": day + "T00:00:00+08:00",
        "fund_code": "002112",
        "raw": {"sha256": code},
        "holdings": [{"reported_rank": 1, "stock_code": code, "nav_weight_pct": 10}],
    }


def test_labels_not_mature_at_cutoff_are_excluded():
    cutoff = "2025-01-03T23:00:00+08:00"
    rows = [row("2025-01-01"), row("2025-01-02", mature=cutoff), row("2025-01-03")]
    assert m.training_rows(rows, "history", cutoff) == rows[:1]
    with pytest.raises(ValueError, match="TIME_LEAKAGE"):
        m.sample_weights("RECENT", rows, cutoff, {})


def test_recency_halves_at_126_sessions_and_does_not_drop_rows():
    sessions = [f"d{i:04d}" for i in range(300)]
    rows = [row(sessions[i]) for i in [1, 127, 253]]
    weights, proof = m.sample_weights("RECENT", rows, "d0299T23:00:00+08:00", {"sessions": sessions, "reports": []})
    assert len(weights) == 3 and np.all(weights > 0)
    assert weights[1] / weights[0] == pytest.approx(2)
    assert weights[2] / weights[1] == pytest.approx(2)
    assert weights.mean() == pytest.approx(1)
    assert all(r["top10_overlap"] is None for r in proof["rows"])


def test_future_report_cannot_affect_weights_and_unknown_is_neutral():
    rows = [row("2025-01-01"), row("2025-01-02"), row("2025-01-03")]
    source = {
        "sessions": [f"2025-01-0{i}" for i in range(1, 7)],
        "reports": [report("2025-01-02", "A"), report("2025-01-03", "B")],
    }
    cutoff = "2025-01-05T23:00:00+08:00"
    weights, proof = m.sample_weights("SIMILAR", rows, cutoff, source)
    assert proof["rows"][0]["top10_overlap"] is None and weights[0] == pytest.approx(1)
    assert proof["rows"][1]["top10_overlap"] == 0
    assert proof["rows"][2]["top10_overlap"] == 1
    assert weights[2] / weights[1] == pytest.approx(4)
    source["reports"].append(report("2025-01-06", "C"))
    changed, changed_proof = m.sample_weights("SIMILAR", rows, cutoff, source)
    assert np.array_equal(weights, changed) and proof == changed_proof


def test_no_holdings_does_not_invent_similarity_or_change_weights():
    rows = [row("2025-01-01"), row("2025-01-02")]
    weights, proof = m.sample_weights(
        "SIMILAR", rows, "2025-01-03T23:00:00+08:00", {"sessions": [r["target"] for r in rows], "reports": []}
    )
    assert np.array_equal(weights, [1, 1])
    assert proof["summary"]["unknown_holdings_count"] == 2


def test_paired_changes_are_per_day_and_block_interval_is_zero_for_identical_predictions():
    rows = [
        {"policy": p, "target": f"d{i}", "label": "UP", "probabilities": [0.2, 0, 0.8]}
        for p in ("ALL", "RECENT")
        for i in range(42)
    ]
    result = m.paired_comparison(rows, "RECENT")
    assert result["corrected"] == result["spoiled"] == result["net_correct"] == 0
    assert result["descriptive_block_interval95"] == [0, 0]


def test_no_lookahead_report_reassignment_to_old_sample():
    rows = [row("2025-01-01"), row("2025-01-02")]
    source = {
        "sessions": [f"2025-01-0{i}" for i in range(1, 6)],
        "reports": [report("2025-01-01", "A"), report("2025-01-03", "B")],
    }
    _, proof = m.sample_weights("SIMILAR", rows, "2025-01-04T23:00:00+08:00", source)
    assert proof["current_report_end"] == "2025-01-03"
    assert all(r["sample_report_end"] == "2025-01-01" and r["top10_overlap"] == 0 for r in proof["rows"])
