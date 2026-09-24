"""境内四周期共用净值门槛；合成数据验证时点，不作为真实预测成绩。"""

import json
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest
from app.services.direction_1d_protocol import window
from app.services.prediction_contract import POLICY_FILE, PredictionFailure, policy_scope, target_dates
from app.services.prediction_features import build_features
from tests.test_prediction_contract import calendar


def data_at(now, *, include_current=False):
    cal = calendar()
    days = [d for d in cal.sessions if d < now.date() or (include_current and d == now.date())][-65:]
    return {
        "calendar": cal,
        "fund": {"fund_code": "000001", "fund_master_id": "family", "fund_type": "STOCK"},
        "navs": [
            {
                "nav_date": d,
                "unit_nav": Decimal(1) + Decimal(i) / 1000,
                "updated_at": now - timedelta(seconds=1),
                "source_published_at": None,
            }
            for i, d in enumerate(days)
        ],
        "dividends": [],
        "facts": [],
        "dividendWatermark": now,
    }


@pytest.mark.parametrize(
    "stamp,base,target",
    [
        ("2026-09-24T08:30:00+08:00", "2026-09-23", "2026-09-24"),
        ("2026-09-24T14:59:59+08:00", "2026-09-23", "2026-09-24"),
        ("2026-09-24T15:00:00+08:00", "2026-09-24", "2026-09-28"),
        ("2026-09-24T23:50:00+08:00", "2026-09-24", "2026-09-28"),
        ("2026-10-01T10:00:00+08:00", "2026-09-30", "2026-10-08"),
    ],
)
def test_all_horizons_agree_on_required_nav_and_first_target(stamp, base, target):
    now = datetime.fromisoformat(stamp)
    daily = window(now)
    assert (daily["base_nav_date"], daily["target_nav_date"], daily["status"]) == (base, target, "OPEN")
    for horizon in ("T5_V1", "T20_V1", "M6_V1"):
        value = target_dates(calendar(), now, horizon)
        assert (value["baseNavDate"], value["startDate"]) == (base, target)


def test_close_waits_for_current_nav_and_uses_it_once_received():
    now = datetime.fromisoformat("2026-09-24T16:00:00+08:00")
    with pytest.raises(PredictionFailure) as error:
        build_features(data_at(now), now, 20, persist=False)
    assert error.value.payload["code"] == "NAV_CURRENT_NOT_READY"
    result = build_features(data_at(now, include_current=True), now, 20, persist=False)
    assert result["dataAsOf"] == "2026-09-24"
    # 来源声明已经公布但本服务截止后才收到，也不能倒填为已知输入。
    data = data_at(now, include_current=True)
    data["navs"][-1]["updated_at"] = now + timedelta(seconds=1)
    with pytest.raises(PredictionFailure) as late:
        build_features(data, now, 20, persist=False)
    assert late.value.payload["code"] == "NAV_CURRENT_NOT_READY"


def test_before_close_requires_previous_day_and_never_consumes_target_nav():
    now = datetime.fromisoformat("2026-09-24T10:00:00+08:00")
    data = data_at(now, include_current=True)
    assert build_features(data, now, 20, persist=False)["dataAsOf"] == "2026-09-23"
    data["navs"] = [r for r in data["navs"] if r["nav_date"] != date(2026, 9, 23)]
    with pytest.raises(PredictionFailure) as error:
        build_features(data, now, 20, persist=False)
    assert error.value.payload["code"] == "NAV_LATEST_NOT_READY"


def test_old_policy_keeps_original_target_dates():
    old = json.loads(POLICY_FILE.with_name("prediction_policy_v2.json").read_text(encoding="utf-8"))
    now = datetime.fromisoformat("2026-09-22T10:00:00+08:00")
    with policy_scope(old):
        original = target_dates(calendar(), now, "T5_V1")
    assert original["endDate"] == "2026-09-30" and "baseNavDate" not in original
    assert target_dates(calendar(), now, "T5_V1")["endDate"] == "2026-09-29"
