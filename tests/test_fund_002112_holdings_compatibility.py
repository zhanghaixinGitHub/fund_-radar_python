"""只校验持仓口径、未披露边界和按路径读取，不加载真实模型或标签。"""

import json

import pytest
from app.services.fund_002112_holdings_compatibility import compare_holdings, holding_vector, json_subtree, summarize


def report(weights, total, full=False, end="2023-06-30"):
    return {
        "stock_nav_pct": total,
        "full_stock_disclosure": full,
        "report_end": end,
        "holdings": [{"stock_code": k, "stock_name": k, "nav_weight_pct": v} for k, v in weights.items()],
    }


def test_partial_zero_overlap_keeps_hidden_possibility():
    result = compare_holdings(report({"A": "10"}, "90"), report({"B": "20"}, "80"))
    assert result["visible_common_nav_pct"] == 0
    assert result["possible_common_upper_nav_pct"] == 80
    assert not result["both_full"]


def test_weighted_intersection_uses_smaller_nav_weight():
    result = compare_holdings(report({"A": "40", "B": "30"}, "70", True), report({"A": "5", "C": "20"}, "25", True))
    assert result["visible_common_nav_pct"] == result["possible_common_upper_nav_pct"] == 5
    assert result["common_target_nav_pct"] == 40 and result["common_peer_nav_pct"] == 5


def test_full_disclosure_does_not_invent_hidden_stock_from_rounding():
    _, meta = holding_vector(report({"A": "40.00", "B": "39.99"}, "80.00", True))
    assert meta["rounding_residual_pct"] == 0.01 and meta["undisclosed_stock_nav_pct"] == 0


@pytest.mark.parametrize("bad", [None, "NaN", "-1"])
def test_bad_weight_never_becomes_zero(bad):
    with pytest.raises((ValueError, ArithmeticError)):
        holding_vector(report({"A": bad}, "50"))


def test_full_disclosure_gap_fails():
    with pytest.raises(ValueError, match="FULL_DISCLOSURE"):
        holding_vector(report({"A": "10"}, "90", True))


def test_duplicate_security_fails():
    r = report({"A": "10"}, "90")
    r["holdings"].append(r["holdings"][0].copy())
    with pytest.raises(ValueError, match="DUPLICATE"):
        holding_vector(r)


def test_cross_period_is_explicit():
    result = compare_holdings(report({"A": "10"}, "10", True), report({"A": "10"}, "10", True, "2024-03-31"))
    assert result["visible_common_nav_pct"] == 10 and not result["same_report_end"]


def test_json_selection_handles_escaped_braces_and_skips_unneeded_branch():
    text = json.dumps(
        {
            "ignored": [{"text": 'escaped \\" quote [ { } ]', "numbers": [1, 2]}],
            "payload": {"funds": {"002112": {"nav": [999], "reports": [{"x": 7}]}}},
        }
    )
    assert json_subtree(text, ["payload", "funds", "002112", "reports", 0, "x"]) == 7
    with pytest.raises(KeyError):
        json_subtree(text, ["missing"])


def test_json_array_route_skips_exam_without_decoding():
    text = '[{"exam":[{"ignored":"value"}],"name":"Q2","train_ids":[["A","2023-01-01"]]}]'
    assert json_subtree(text, [0, "train_ids"]) == [["A", "2023-01-01"]]


def test_dates_are_not_independent_report_pairs():
    comp = compare_holdings(report({"A": "10"}, "10", True), report({"A": "5"}, "5", True))
    rows = [
        {"status": "MATCHED", "pair_id": "a", "target": "2023-09-01"},
        {"status": "MATCHED", "pair_id": "a", "target": "2023-09-04"},
        {"status": "NO_EXACT_TARGET_INPUT_ROW", "target": "2023-09-05"},
    ]
    s = summarize(rows, {"a": {"comparison": comp}})
    assert s["rows"] == 3 and s["matched_rows"] == 2 and s["unmatched_rows"] == 1
    assert s["unique_report_pairs"] == 1
