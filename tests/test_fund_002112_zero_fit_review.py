"""只用合成行验证计数、配对、时间块与日期边界，不读取真实研究资料。"""

import copy

import numpy as np
import pytest
from app.services import fund_002112_zero_fit_review as review


def protocol():
    return {
        "bins": {
            "stock_nav_weight": [0.5, 0.8],
            "disclosed_stock_fraction": [0.5, 0.8],
            "report_age_days": [90, 180],
            "absolute_return_fraction": [0, 0.001, 0.005],
        }
    }


def row():
    return {
        "fund_code": "002112",
        "base": "2024-01-02",
        "target": "2024-01-03",
        "base_unit_nav": "1.0000",
        "target_unit_nav": "1.0010",
        "actual_direction": "UP",
        "x": [0.0] * 12 + [0.4, 0.8, 90, 1] + [0.01, 0.02, 0.01, 0.02],
        "nav": {"lag_sessions": 1},
    }


def test_exact_boundaries_keep_small_up_as_up():
    value = row()
    review.validate_row(value)
    result = review.groups(value, protocol())
    assert result["report_age"] == "0-90"
    assert result["stock_weight"] == ">=80%"
    assert result["disclosure_fraction"] == "50%-<80%"
    assert result["absolute_return"] == "0-0.1%"
    assert value["actual_direction"] == "UP"
    value["target_unit_nav"], value["actual_direction"] = "1.0000", "FLAT"
    assert review.groups(value, protocol())["absolute_return"] == "EXACT_FLAT"


def test_zero_stock_weight_is_unknown_coverage_not_zero_imputation():
    value = row()
    value["x"][13] = 0
    assert review.groups(value, protocol())["disclosure_fraction"] == "NOT_APPLICABLE"
    del value["nav"]
    assert review.groups(value, protocol())["nav_lag"] == "UNKNOWN_OLD_CONTRACT"


@pytest.mark.parametrize("change", ["future", "label", "missing", "nan", "negative_nav", "duplicate"])
def test_invalid_rows_rejected(change):
    value = row()
    if change == "future":
        value["target"] = "2025-01-02"
    if change == "label":
        value["actual_direction"] = "FLAT"
    if change == "missing":
        value["x"].pop()
    if change == "nan":
        value["x"][0] = float("nan")
    if change == "negative_nav":
        value["base_unit_nav"] = "-1"
    with pytest.raises(ValueError):
        review.index_rows([value, copy.deepcopy(value)] if change == "duplicate" else [value])


def test_join_uses_dates_and_checks_labels():
    inputs = review.index_rows([row()])
    raw = [{"target": "2024-01-03", "actual": "UP", "N7": "DOWN", "L20": "UP"}]
    result = review.join_daily(raw, inputs, ("N7", "L20"), protocol())
    assert result[0]["actual"] == "UP"
    raw[0]["actual"] = "DOWN"
    with pytest.raises(ValueError, match="MISMATCH"):
        review.join_daily(raw, inputs, ("N7", "L20"), protocol())


def test_pairs_count_rescued_and_lost_days_not_just_new_hits():
    rows = [
        {"target": f"2024-01-0{i}", "actual": actual, "directions": {"A": a, "B": b}}
        for i, (actual, a, b) in enumerate([("UP", "UP", "DOWN"), ("DOWN", "UP", "DOWN"), ("FLAT", "FLAT", "FLAT")], 2)
    ]
    indices = np.tile(np.arange(3), (20, 1))
    result = review.pair_result(rows, "A", "B", indices)
    assert (result["gained"], result["lost"], result["net_correct"]) == (1, 1, 0)
    assert result["diagnostic_interval_pp"] == [0, 0]
    assert result["class_net"] == {"DOWN": -1, "FLAT": 0, "UP": 1}


def test_blocks_preserve_quarter_sizes_and_circular_order():
    days = [f"2023-06-{d:02d}" for d in range(20, 30)] + [f"2023-07-{d:02d}" for d in range(1, 8)]
    settings = {"seed": 7, "replications": 100, "block_length": 3}
    indices = review.block_indices(days, settings)
    assert indices.shape == (100, 17)
    assert np.all(indices[:, :10] < 10) and np.all(indices[:, 10:] >= 10)
    assert np.all((indices[:, 1] - indices[:, 0]) % 10 == 1)
    assert np.array_equal(indices, review.block_indices(days, settings))
    with pytest.raises(ValueError):
        review.block_indices(days[::-1], settings)


def test_empty_group_unknown_accuracy_and_constant_control():
    assert review.summary([], ("DOWN",))["models"]["DOWN"]["accuracy"] is None
    rows = [{"actual": "DOWN", "directions": {"A": "UP"}}, {"actual": "FLAT", "directions": {"A": "FLAT"}}]
    result = review.summary(rows, ("A", "DOWN"))
    assert result["models"]["A"]["correct_by_class"]["FLAT"] == 1
    assert result["models"]["DOWN"]["correct"] == 1


def test_immutable_output_and_envelope_validation(tmp_path):
    p = tmp_path / "result.json"
    review.save_once(p, {"count": 3})
    review.save_once(p, {"count": 3})
    with pytest.raises(ValueError):
        review.save_once(p, {"count": 4})
    p.write_text('{"hash":"bad","payload":{"count":3}}', encoding="utf-8")
    with pytest.raises(ValueError, match="CONTENT_HASH"):
        review.read_json(p)
