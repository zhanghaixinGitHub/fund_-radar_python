"""数字片段恢复的守恒与缺失边界；这些临时数值不是独立校准产品。"""

from decimal import Decimal

import pytest
from app.services.fund_quality_v2_restricted_layout import _unique_financial_cells, fixed_width_stock_restrictions


def test_every_visible_fragment_is_consumed_in_unique_five_columns_and_both_identities_hold():
    fragments = [["3.2", "5.8", "12", "39.", "69."], ["5", "0", "00", "60"]]
    assert _unique_financial_cells(fragments, quantity_limit=Decimal(20), fair_limit=Decimal(100)) == (
        Decimal("3.25"),
        Decimal("5.80"),
        Decimal(12),
        Decimal("39.00"),
        Decimal("69.60"),
    )


@pytest.mark.parametrize("problem", ["missing", "price_conflict", "quantity_limit", "value_limit"])
def test_unknown_cells_and_conflicting_originals_cannot_be_zero_filled_or_forced_to_fit(problem):
    fragments = [["3.25", "5.80", "12", "39.00", "69.60"]]
    quantity, fair = Decimal(20), Decimal(100)
    if problem == "missing":
        fragments[0].pop()
    elif problem == "price_conflict":
        fragments[0][-1] = "68.60"
    elif problem == "quantity_limit":
        quantity = Decimal(11)
    else:
        fair = Decimal(69)
    with pytest.raises(ValueError, match="UNRESOLVED"):
        _unique_financial_cells(fragments, quantity_limit=quantity, fair_limit=fair)


def test_missing_explicit_table_header_is_not_inferred_from_fund_type_or_row_values():
    with pytest.raises(ValueError, match="HEADER_UNSUPPORTED"):
        fixed_width_stock_restrictions("金额单位：人民币元\n600001 3.25 5.80 12 39.00 69.60", [])
