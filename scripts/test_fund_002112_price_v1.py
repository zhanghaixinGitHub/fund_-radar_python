"""量价窗口与单位边界，不用训练分数断言实现正确。"""

import numpy as np
import pytest

from scripts import fund_002112_price_sources_v1 as source


def test_nav_known_window_return_and_downside():
    values = [1.01**i for i in range(61)]
    x = source.nav_extra(values)
    assert np.isclose(x[0], 0.01)
    assert np.isclose(x[3], 1.01**40 - 1)
    assert x[-1] == 0
    with pytest.raises(ValueError, match="WINDOW"):
        source.nav_extra(values[1:])


def test_market_activity_is_dimensionless_and_excludes_today_from_denominator():
    rows = [{"pct_chg": 0, "amount": 100, "vol": 200} for _ in range(20)] + [{"pct_chg": 0, "amount": 200, "vol": 400}]
    assert source.market_extra(rows) == [0, 0, 0, 1, 1]
    scaled = [dict(r, amount=r["amount"] * 1000, vol=r["vol"] * 100) for r in rows]
    assert source.market_extra(scaled) == source.market_extra(rows)


def test_zero_activity_base_is_unknown_not_zero_feature():
    with pytest.raises(ValueError, match="ZERO_MARKET"):
        source.market_extra([{"pct_chg": 0, "amount": 0, "vol": 0} for _ in range(21)])
