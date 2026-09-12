"""一日协议的模拟/纯计算验证；这些用例不是真实提前预测。"""

from datetime import date, datetime

import numpy as np
import pytest
from app.schemas.direction_1d import ForecastRequest, Scope
from app.services.direction_1d_data import classify
from app.services.direction_1d_protocol import ZONE, features, input_days, label, score, window
from app.services.direction_1d_training import build_samples, select_fit, weights
from pydantic import ValidationError


@pytest.mark.parametrize(
    ("time", "base", "target", "status"),
    [
        ("2026-09-11T18:00:00", "2026-09-11", "2026-09-14", "OPEN"),
        ("2026-09-13T12:00:00", "2026-09-11", "2026-09-14", "OPEN"),
        ("2026-09-14T08:29:59", "2026-09-11", "2026-09-14", "OPEN"),
        ("2026-09-14T08:30:00", "2026-09-11", "2026-09-14", "MISSED_DEADLINE"),
        ("2026-09-14T17:59:59", "2026-09-11", "2026-09-14", "MISSED_DEADLINE"),
        ("2026-09-30T18:00:00", "2026-09-30", "2026-10-08", "OPEN"),
        ("2025-12-31T18:00:00", "2025-12-31", "2026-01-05", "OPEN"),
    ],
)
def test_exact_windows(time, base, target, status):
    w = window(datetime.fromisoformat(time).replace(tzinfo=ZONE))
    assert (w["base_nav_date"], w["target_nav_date"], w["status"]) == (base, target, status)


def test_calendar_fails_closed():
    with pytest.raises(ValueError, match="CALENDAR"):
        window(datetime(2027, 1, 1, tzinfo=ZONE))
    assert len(input_days(date(2026, 1, 5))) == 61


def test_seven_features_include_t():
    v = [float(i) for i in range(1, 62)]
    x = features(v)
    assert x[0] == 61 / 56 - 1
    assert x[1] == 61 / 41 - 1
    assert x[2] == 60
    assert x[3] == pytest.approx(np.std([v[i] / v[i - 1] - 1 for i in range(41, 61)], ddof=0))
    assert x[4:] == [0, 1, 0]
    assert features(list(reversed(v)))[6] == 60


@pytest.mark.parametrize("v", [[1] * 61, [1] * 60, [1] * 60 + [0], [1] * 60 + [float("nan")]])
def test_bad_features(v):
    with pytest.raises(ValueError):
        features(v)


@pytest.mark.parametrize(
    ("a", "b", "direction", "y"), [("1", "1", "FLAT", 0), ("1", "1.00000001", "UP", 1), ("1", ".99999999", "DOWN", 0)]
)
def test_decimal_direction(a, b, direction, y):
    result = label(a, b)
    assert result["actual_direction"] == direction and result["y"] == y


@pytest.mark.parametrize("bad", ["0", "-1", "NaN", "Infinity"])
def test_invalid_label(bad):
    with pytest.raises(ValueError):
        label("1", bad)


def test_no_20d_model_or_extra_identity():
    with pytest.raises(ValueError, match="MODEL_PROTOCOL"):
        score({"horizon": 20}, [0] * 7)
    with pytest.raises(ValidationError):
        ForecastRequest(fund_code="001632", user_id="admin")
    with pytest.raises(ValidationError):
        Scope(fund_codes=["001632"] * 101)


def test_family_weights_do_not_inflate_duplicate_shares():
    rows = [{"family": "A", "t": "1"}, {"family": "B", "t": "1"}, {"family": "B", "t": "2"}]
    w = weights(rows)
    duplicated = rows + [{"family": "A", "t": "1"}]
    w2 = weights(duplicated)
    assert sum(w) == pytest.approx(3)
    assert sum(w2) == pytest.approx(3)
    assert w[0] == pytest.approx(w2[0] + w2[-1])


def test_late_answers_not_in_fit():
    r = {"mature_at": "2024-01-02T08:00:00+08:00", "t": "2023-12-29"}
    assert select_fit([r], datetime(2024, 1, 1, tzinfo=ZONE)) == []


def test_special_types_and_unknown_mapping_not_forced():
    base = {
        "fund_type": "STOCK",
        "fund_master_id": "family",
        "master_name": "product",
        "profile_hash": "hash",
        "source_code": "TUSHARE_PRO_FUND",
        "status": "ACTIVE",
        "benchmark": "沪深300指数95%",
    }
    assert classify(base)["group_id"] == "CN_EQUITY"
    assert (
        classify({**base, "benchmark": "沪深300指数80%+恒生指数20%"})["classification_reason"]
        == "SPECIAL_POLICY_REQUIRED"
    )
    assert classify({**base, "benchmark": "存款利率"})["classification_reason"] == "GROUP_UNVERIFIED"
    assert classify({**base, "fund_type": "BOND", "benchmark": "iBoxx亚债基金中国指数×100%"})["group_id"] == "CN_BOND"


def test_missing_date_not_compressed_or_filled():
    from app.services.direction_1d_protocol import calendar

    days, _ = calendar()
    fund = {
        "fund_code": "123456",
        "product_family_id": "f",
        "group_id": "CN_EQUITY",
        "rows": [
            {"date": str(d), "nav": str(1 + i / 1000), "ann_date": None, "source_hash": "h"}
            for i, d in enumerate(days[:63])
            if i != 30
        ],
    }
    assert build_samples({"funds": [fund]}) == []
