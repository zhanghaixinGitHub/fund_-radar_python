"""用小型预测替身检查原性能门槛，不做任何真实拟合。"""

from app.services.fund_002112_information_experiment import comparison


def rows(directions):
    dates = ["2023-04-17", "2023-04-18", "2023-07-03", "2023-07-04", "2023-10-09", "2023-10-10"]
    actual = ["DOWN", "UP", "DOWN", "UP", "FLAT", "UP"]
    return [
        {"fund_code": "002112", "target": d, "actual_direction": a, "direction": p}
        for d, a, p in zip(dates, actual, directions, strict=True)
    ]


def test_higher_total_still_fails_if_up_class_worse():
    values = {
        "CANDIDATE": rows(["DOWN", "DOWN", "DOWN", "UP", "FLAT", "UP"]),
        "L20_ORIGINAL": rows(["UP", "UP", "UP", "UP", "UP", "UP"]),
        "N7_ORIGINAL": rows(["UP", "UP", "UP", "UP", "UP", "UP"]),
    }
    result = comparison(values, True)
    assert result["checks"]["total_strictly_above_controls"]
    assert not result["checks"]["class_not_worse_UP"]
    assert not result["passed"]


def test_full_does_not_add_historical_quarter_gate():
    values = {
        "CANDIDATE": rows(["DOWN", "UP", "DOWN", "UP", "FLAT", "UP"]),
        "L20_ORIGINAL": rows(["UP"] * 6),
        "N7_ORIGINAL": rows(["DOWN"] * 6),
    }
    result = comparison(values, False)
    assert result["passed"]
    assert "two_quarters_not_worse_L20" not in result["checks"]
    assert result["pairs"]["L20_ORIGINAL"]["net"] == 3
