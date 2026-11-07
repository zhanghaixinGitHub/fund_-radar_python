"""官网复权净值按原明示精度核对，不能把上游另一条曲线当作舍入。"""

from decimal import Decimal

import pytest
from app.services.fund_rating_direct_index import th_own_nav_check


def check(*, literal="2.5843", adjusted="2.58432857", unit="2.4544"):
    day = "2026-01-05"
    return th_own_nav_check(
        [{"fundDate": "20260105", "fundAdjustedNav": literal}],
        {day: {"unit_nav": Decimal("2.4544"), "publisher_nav": Decimal(literal)}},
        {day: Decimal(adjusted)}, {day: Decimal(unit)}, [day],
    )


def test_explicit_four_decimal_original_accepts_only_half_quantum_rounding():
    result = check()
    assert result["own_share_observations_checked"] == 1
    assert result["four_decimal_adjusted_rows_checked"] == 1
    assert Decimal(result["maximum_adjusted_nav_difference"]) == Decimal(".00002857")
    assert check(adjusted="2.5843")["four_decimal_adjusted_rows_checked"] == 0


@pytest.mark.parametrize("changes", [
    {"adjusted": "2.4544"},  # 单位值被当作复权值，禁止用宽容差掩盖上游断点。
    {"adjusted": "2.58435001"},
    {"literal": "2.584", "adjusted": "2.58402857"},  # 原字面精度不符。
    {"unit": "2.4543"},  # 单位净值始终要求逐值一致。
])
def test_conflict_or_unverified_original_precision_is_rejected(changes):
    with pytest.raises(ValueError, match="TH_OWN_NAV_CONFLICT"):
        check(**changes)
