"""新候选的反证准入与封存门禁，使用替身，不暴露新测试答案。"""

from copy import deepcopy
from datetime import date, timedelta
from math import sin

import pytest
from scripts import fund_002112_codex_replay_v2 as replay


def packet():
    return {
        "as_of": "2026-09-30T10:00:00+08:00",
        "window": {"base_nav_date": "2026-09-29"},
        "nav": [
            {
                "nav_date": "2026-09-29",
                "unit_nav": "1",
                "received_at": "2026-09-30T09:00:00+08:00",
                "stored_at": "2026-09-30T09:00:00+08:00",
            }
        ],
        "report": {"available_at": "2026-09-01T08:00:00+08:00", "raw": {"received_at": "2026-09-24T22:00:00+08:00"}},
        "quote_proofs": [{"business_date": "2026-09-29", "received_at": "2026-09-30T09:24:00+08:00"}],
    }


@pytest.mark.parametrize("kind", ["nav", "quote", "answer", "report"])
def test_future_sources_rejected(kind):
    value = deepcopy(packet())
    if kind == "nav":
        value["nav"][0]["nav_date"] = "2026-09-30"
    elif kind == "quote":
        value["quote_proofs"][0]["received_at"] = "2026-09-30T10:01:00+08:00"
    elif kind == "report":
        value["report"]["raw"]["received_at"] = "2026-10-01T08:00:00+08:00"
    else:
        value["nested"] = {"actual_direction": "UP"}
    with pytest.raises(ValueError):
        replay.validate_packet(value)


def test_valid_source_window():
    replay.validate_packet(packet())


def test_scoring_before_seal_never_reads_database(monkeypatch, tmp_path):
    monkeypatch.setattr(replay, "OUT", tmp_path)
    monkeypatch.setattr(replay, "checked", lambda: {"targets": ["2026-09-30"]})
    monkeypatch.setattr(replay, "get_engine", lambda: pytest.fail("封存前不能读取答案"))
    with pytest.raises(ValueError, match="ALL_FIRST_PREDICTIONS"):
        replay.score()


def test_external_network_rejected():
    with pytest.raises(RuntimeError, match="EXTERNAL_NETWORK"):
        replay.network_guard(None, ("api.deepseek.com", 443))


def test_noise_sign_is_not_direction_vote():
    assert replay.sign(0.1, 0.25) == 0
    assert replay.sign(-0.1, 0.25) == 0


def trend_packet():
    """合成稳定正漂移，噪声小但非零；它不能仅因均值正而被误判异常。"""
    value = packet()
    dates = [str(date(2026, 9, 29) - timedelta(days=60 - i)) for i in range(61)]
    prices = [100.0]
    for i in range(60):
        prices.append(prices[-1] * (1 + (0.5 + 0.05 * sin(i)) / 100))
    value["nav"] = [
        {
            "nav_date": day,
            "unit_nav": str(price),
            "received_at": "2026-09-30T09:00:00+08:00",
            "stored_at": "2026-09-30T09:00:00+08:00",
        }
        for day, price in zip(dates, prices, strict=True)
    ]
    value["expected_nav_dates"] = dates
    value["companies"] = [{"weight_pct": 90, "quote": {"date": "2026-09-29", "change_pct": 0.5}}]
    value["market"] = {code: {"date": "2026-09-29", "change_pct": 1.0} for code in ["a", "b"]}
    return value


def test_stable_positive_drift_is_not_extreme_shock():
    prediction = replay.predict(trend_packet())
    assert prediction["state"] == "PREDICTED"
    assert prediction["direction"] == "UP"


def test_extreme_move_triggers_abstention_without_flipping_direction():
    value = trend_packet()
    value["nav"][-1]["unit_nav"] = str(float(value["nav"][-2]["unit_nav"]) * 0.9)
    prediction = replay.predict(value)
    assert prediction["state"] == "ABSTAIN"
    assert "EXTREME_PREVIOUS_DAY_MOVE" in prediction["reason_codes"]
    assert prediction["direction"] is None


def test_missing_market_is_not_bullish_evidence():
    value = trend_packet()
    value["market"] = {}
    prediction = replay.predict(value)
    assert prediction["state"] == "ABSTAIN"
    assert "MARKET_SOURCE_MISSING_AT_CUTOFF" in prediction["reason_codes"]
