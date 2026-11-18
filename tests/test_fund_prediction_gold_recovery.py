import pytest
from app.services.fund_prediction_gold_recovery import gold_rows


def page(day, code="Au99.99", close="101", change="1", pct="1.00%"):
    cells = [day, code, "100", "102", "99", close, change, pct]
    return ("<table><tr>" + "".join("<td>" + cell + "</td>" for cell in cells) + "</tr></table>").encode()


def test_old_day_or_different_contract_never_substitutes_for_current_spot():
    with pytest.raises(ValueError, match="REQUIRED_DATE_NOT_RETURNED"):
        gold_rows(page("2026-11-13"), "2026-11-16")
    with pytest.raises(ValueError, match="REQUIRED_DATE_NOT_RETURNED"):
        gold_rows(page("2026-11-16", code="Au(T+D)"), "2026-11-16")
    row, _ = gold_rows(page("2026-11-16"), "2026-11-16")
    assert row[5] == "101"


def test_source_arithmetic_conflict_is_not_silently_repaired():
    with pytest.raises(ValueError, match="RETURN_MISMATCH"):
        gold_rows(page("2026-11-16", pct="5%"), "2026-11-16")
