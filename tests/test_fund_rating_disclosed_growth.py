"""冻结披露增长不能混成除息日现金再投资；旧历史版本保留单独重放。"""

from decimal import Decimal, localcontext

import pytest
from app.services.fund_rating_metrics import total_return
from app.services.fund_rating_passive_performance import disclosed_growth


def test_cash_with_ex_day_market_move_differs_from_cash_reinvestment():
    with localcontext() as context:
        context.prec = 70
        values = disclosed_growth(["1", ".8", ".88"], ["0", ".19", "0"])
        assert values[1] == Decimal(".8") / Decimal(".81")
        assert values[-1] == Decimal(".88") / Decimal(".81")
        assert values != total_return(["1", ".8", ".88"], ["0", ".19", "0"], ["1"] * 3)


@pytest.mark.parametrize(
    "units,cash",
    [([], []), (["1"], []), (["1", "0"], ["0", "0"]), (["1", ".8"], ["0", "1"]), (["1", ".8"], ["0", "-.1"])],
)
def test_invalid_observations_never_become_a_growth_point(units, cash):
    with pytest.raises(ValueError):
        disclosed_growth(units, cash)
