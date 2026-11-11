import pytest
from app.services.fund_prediction_hk_recovery import closed_rows


def test_public_daily_history_selects_real_closed_rows_and_refuses_wrong_year_or_missing_base():
    raw = (
        b'daily_data_26="\\n\\\r\n261109 15.420 15.160 15.600 15.060 12190398\\n\\\r\n'
        b'261110 15.230 15.130 15.490 14.970 11353000\\n\\\r\n";'
    )
    rows = closed_rows(raw, "2026-11-10")
    assert rows["2026-11-10"]["close"] == "15.130"
    with pytest.raises(ValueError, match="YEAR_OR_FORMAT"):
        closed_rows(raw, "2027-11-10")
    with pytest.raises(ValueError, match="MISSING"):
        closed_rows(raw, "2026-11-11")
