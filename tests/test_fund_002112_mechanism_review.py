"""合成来源和纯算术检查；测试不拟合任何模型。"""

from types import SimpleNamespace

import numpy as np
import pytest
from app.services import fund_002112_mechanism_review as r
from sklearn.linear_model import LogisticRegression


def nav_window(values, publications=None):
    days = [f"2023-04-{i:02d}" for i in range(1, len(values) + 1)]
    nav = {
        d: {"nav": str(v), "ann_date": publications[i] if publications else d}
        for i, (d, v) in enumerate(zip(days, values, strict=True))
    }
    return {"target": "2023-04-10", "nav": {"nav_dates": days, "S": days[-1]}}, nav


@pytest.mark.parametrize(
    "values,expected", [([1, 2, 3], 0), ([1, 2, 2], 1), ([1, 2, 2, 2], 2), ([1, 1, 1], 2), ([1, 1, 1.00001], 0)]
)
def test_public_prior_equal_run(values, expected):
    row, nav = nav_window(values)
    assert r.prior_equal_run(row, nav) == expected


def test_future_publication_forbidden():
    row, nav = nav_window([1, 1], ["2023-04-01", "2023-04-10"])
    with pytest.raises(ValueError, match="NOT_KNOWN"):
        r.prior_equal_run(row, nav)


def test_missing_nav_never_filled():
    row, nav = nav_window([1, 1])
    nav.pop("2023-04-01")
    with pytest.raises(KeyError):
        r.prior_equal_run(row, nav)


def test_target_cannot_enter_prior_window():
    row, nav = nav_window([1, 1])
    row["target"] = "2023-04-02"
    with pytest.raises(ValueError, match="IDENTITY"):
        r.prior_equal_run(row, nav)


def test_no_fit_guard_restores_method():
    before = LogisticRegression.fit
    with r.no_fits(), pytest.raises(RuntimeError, match="FORBIDS"):
        LogisticRegression().fit([[0]], [0])
    assert LogisticRegression.fit is before


def test_exact_linear_margin_decomposition():
    mean = np.arange(20, dtype=float)
    scale = np.arange(1, 21, dtype=float)
    coef = np.vstack([np.arange(20) * 0.1, np.zeros(20), np.arange(20) * -0.2])
    bias = np.array([0.2, -0.4, 0.8])
    scaler = SimpleNamespace(mean_=mean, scale_=scale, transform=lambda x: (x - mean) / scale)
    classifier = SimpleNamespace(
        classes_=np.array(r.CLASSES), coef_=coef, intercept_=bias, decision_function=lambda x: x @ coef.T + bias
    )
    rows = [{"x": np.arange(20).tolist()}, {"x": np.arange(3, 23).tolist()}]
    offset, terms, margin = r.margin_parts({"scaler": scaler, "classifier": classifier}, rows, mean + 2)
    assert np.allclose(terms.sum(1) + offset, margin)


def report(holdings, full=False):
    return {
        "holdings": [{"stock_code": k, "nav_weight_pct": str(v)} for k, v in holdings],
        "stock_nav_pct": "80",
        "full_stock_disclosure": full,
    }


def test_overlap_uses_net_asset_weight_not_normalized_stock_weight():
    overlap = r.stock_overlap(report([("a", 10), ("b", 20)]), report([("a", 3), ("c", 40)]))
    assert overlap["shared_nav_pct_lower_bound"] == 3 and overlap["count"] == 1
    assert not overlap["both_full_disclosure"]


def test_missing_overlap_is_disclosed_zero_not_full_holdings_claim():
    overlap = r.stock_overlap(report([("a", 10)]), report([("b", 20)]))
    assert overlap["shared_nav_pct_lower_bound"] == 0 and not overlap["both_full_disclosure"]


@pytest.mark.parametrize("weights", [[("a", -1)], [("a", 10), ("a", 20)], [("a", "NaN")]])
def test_overlap_rejects_invalid_weights(weights):
    with pytest.raises((ValueError, ArithmeticError)):
        r.stock_overlap(report(weights), report([("a", 20)]))


def test_flat_weight_stat_is_class_specific():
    rows = [
        {"fund_code": "002112", "target": "2023-04-01", "actual_direction": "FLAT"},
        {"fund_code": "160323", "target": "2023-04-01", "actual_direction": "UP"},
    ]
    pred = [{"input_hash": r.digest(row), "direction": "UP", "scores": [0.3, 0.1, 0.6]} for row in rows]
    result = r.flat_learning(rows, pred, [1, 9])
    assert result["all"]["flat_share_of_total_saved_weight"] == 0.1
    assert result["own"]["correct_flat"] == 0


def test_model_prediction_identity_required():
    with pytest.raises(ValueError, match="IDENTITY"):
        r.flat_learning([{"target": "2023-01-01"}], [{"input_hash": "bad"}], [1])


def test_gate_union_is_explicitly_oracle():
    controls = {
        "a": [
            {"target": "2023-01-01", "actual_direction": "UP", "direction": "UP"},
            {"target": "2023-01-02", "actual_direction": "DOWN", "direction": "UP"},
        ],
        "b": [
            {"target": "2023-01-01", "actual_direction": "UP", "direction": "DOWN"},
            {"target": "2023-01-02", "actual_direction": "DOWN", "direction": "DOWN"},
        ],
    }
    result = r.gate_diagnostic(controls, controls["a"])
    assert result["minimum_total_from_class_gates"] == 2 and result["same_day_oracle_union_correct"] == 2
    assert result["both_controls_correct"] == 0 and "answers" in result["warning"]


def test_nonfinite_summary_rejected():
    with pytest.raises(ValueError):
        r.stats([float("nan")])
    assert r.stats([])["median"] is None


def test_report_binding_uses_manifest_not_filename(monkeypatch):
    candidate = "C:/frozen/public-pdf-probe/report-candidate.json"
    monkeypatch.setattr(
        r, "read_json", lambda p: {"reports": [{"parsed_file": candidate}]} if "worklist" in str(p) else []
    )
    assert r.parsed_report_paths({"files": {candidate: "hash"}}) == [candidate]
    with pytest.raises(ValueError, match="NOT_FROZEN"):
        r.parsed_report_paths({"files": {}})
