from decimal import Decimal

import pytest
from app.services.fund_prediction_overseas_recovery import jp_close, official_fx, us_close


def test_us_regular_close_does_not_use_after_hours_or_extension_prices():
    fields = ["name", "725.1100", "-1.10", "2026-11-11 09:14:55", "-8.0400"] + ["0"] * 31
    fields[21], fields[24], fields[25] = "736.0000", "Nov 10 07:59PM EST", "Nov 10 04:00PM EST"
    fields[26], fields[29], fields[34] = "733.1500", "2026", "724.6000"
    raw = ('var hq_str_gb_mu="' + ",".join(fields) + '";').encode()
    value = us_close(raw, "MU.US", "2026-11-10")
    assert value["close"] == "725.1100" and value["previous_close"] == "733.1500"
    assert value["closed_at"] == "2026-11-10T16:00:00-05:00"
    with pytest.raises(ValueError, match="CLOSE_DATE"):
        us_close(raw, "MU.US", "2026-11-11")


def test_japan_history_uses_past_closed_rows_and_not_live_header():
    raw = b"""<h1>8035.T live 99999</h1><table>
    <tr><td>2026/11/11</td><td>100</td><td>110</td><td>90</td><td>105</td></tr>
    <tr><td>2026/11/10</td><td>90</td><td>95</td><td>89</td><td>94</td></tr>
    <tr><td>2026/11/9</td><td>80</td><td>90</td><td>79</td><td>88</td></tr></table>"""
    quote, rows = jp_close(raw, "8035.JP", "2026-11-10")
    assert quote["close"] == "94" and quote["previous_close"] == "88"
    assert "2026-11-11" not in rows


def test_official_midpoint_keeps_units_and_rejects_changed_column_order():
    raw = (
        "<table><tr><td>日期</td><td>美元</td><td>欧元</td><td>日元</td><td>港元</td></tr>"
        "<tr><td>2026-11-10</td><td>671.67</td><td>750.71</td><td>4.2355</td><td>85.642</td></tr></table>"
    )
    result = official_fx(raw.encode(), "2026-11-10")
    rates = {r["currency"]: Decimal(r["cny_per_unit"]) for r in result["rates"]}
    assert rates == {"USD": Decimal("6.7167"), "JPY": Decimal(".042355"), "HKD": Decimal(".85642")}
    with pytest.raises(ValueError, match="FX_BASE_MISSING"):
        official_fx(raw.replace("美元", "外币").encode(), "2026-11-10")
