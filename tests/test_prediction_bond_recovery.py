import json

import pytest
from app.services.fund_prediction_bond_recovery import closed_rows


def test_only_actual_closed_baseline_and_previous_session_are_used():
    rows = [
        {"day": day, "close": price}
        for day, price in [("2026-11-09", "112.977"), ("2026-11-10", "112.889"), ("2026-11-11", "113.5")]
    ]
    raw = ("var _113052=(" + json.dumps(rows) + ");").encode()
    actual = closed_rows(raw, "113052.SH", "2026-11-10")
    assert list(actual) == ["2026-11-09", "2026-11-10"]
    assert actual["2026-11-10"]["close"] == "112.889"
    with pytest.raises(ValueError, match="BASE_MISSING"):
        closed_rows(raw, "113052.SH", "2026-11-12")
    with pytest.raises(ValueError, match="SOURCE_FORMAT"):
        closed_rows(raw, "113043.SH", "2026-11-10")
