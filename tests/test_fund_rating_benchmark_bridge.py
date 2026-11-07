"""构造基准的真实报价闭合、整体精度和明确缺日边界。"""

from copy import deepcopy
from decimal import Decimal

import pytest
from app.services.fund_rating_benchmark_bridge import reconstruct, reconstruct_normalized_windows
from app.services.fund_rating_public_etf import _benchmark_series


def test_contract_reference_is_distinguished_from_missing_issuer_original():
    dates = ["2023-09-28", "2023-10-09", "2023-10-10", "2023-10-11"]
    result = reconstruct(dates, dict.fromkeys(dates, "100"), dict.fromkeys(dates[1:], "500"),
                         annual_cash_rate="0", expected_missing_dates=dates[:1])
    assert all(Decimal(v) == 1 for v in result["levels"])
    assert result["official_overlap"] == 3 and result["missing_publisher_dates"] == dates[:1]
    assert result["scope"] == "CONTRACT_DERIVED_REFERENCE_NOT_UNPUBLISHED_ISSUER_OBSERVATIONS"
    with pytest.raises(ValueError, match="INDEX_CALENDAR_INCOMPLETE"):
        reconstruct(dates, dict.fromkeys(dates[1:], "100"), dict.fromkeys(dates[1:], "500"),
                    annual_cash_rate="0", expected_missing_dates=dates[:1])


def test_publisher_internal_hole_or_different_series_is_not_patched_or_tolerance_tuned():
    dates = ["2023-09-28", "2023-10-09", "2023-10-10", "2023-10-11"]
    prices = dict.fromkeys(dates, "100")
    with pytest.raises(ValueError, match="CROSS_CHECK_INCOMPLETE"):
        reconstruct(dates, prices, {dates[0]: "500", dates[-1]: "500"}, annual_cash_rate="0",
                    expected_missing_dates=dates[1:3])
    with pytest.raises(ValueError, match="PRECISION_CONFLICT"):
        reconstruct(dates, prices, dict(zip(dates, ["500", "500", "500", "501"], strict=True)),
                    annual_cash_rate="0", expected_missing_dates=[])


def test_published_precision_is_explicit_and_cannot_be_an_arbitrary_tolerance():
    dates = ['2026-09-28', '2026-09-29', '2026-09-30']
    prices = dict.fromkeys(dates, '100')
    published = dict(zip(dates, ['1', '1.00004', '1'], strict=True))
    with pytest.raises(ValueError, match='PRECISION_CONFLICT'):
        reconstruct(dates, prices, published, annual_cash_rate='0', expected_missing_dates=[])
    result = reconstruct(dates, prices, published, annual_cash_rate='0', expected_missing_dates=[],
                         publisher_level_half_quantum='.00005')
    assert result['precision_fit'] and result['publisher_level_half_quantum'] == '.00005'
    with pytest.raises(ValueError, match='PRECISION_UNSUPPORTED'):
        reconstruct(dates, prices, published, annual_cash_rate='0', expected_missing_dates=[],
                    publisher_level_half_quantum='.0001')


def test_th_inner_success_code_and_own_identity_remain_strict():
    raw = {"retCode": 0, "payload": {"retCode": "000000", "benchMark": "隔离指数*95%+活期税后*5%",
        "fundHistoryRecord": {"fundCode": "123451", "fundDayRecordList": [
            {"fundDate": "20260929", "fundNetValue": "1", "fundAdjustedNav": "1"}]},
        "indexHistoryRecord": {"fundDayRecordList": [{"fundDate": "20260929", "fundNetValue": "1000"}]}}}
    ref = {"format": "ISSUER_DAILY_BENCHMARK_TH"}
    result = _benchmark_series(raw, ref, "123451", "隔离指数*95%+活期税后*5%")
    assert result["rows"][0]["navDate"] == "2026-09-29"
    broken = deepcopy(raw)
    broken["payload"]["retCode"] = "000001"
    with pytest.raises(ValueError, match="RESPONSE"):
        _benchmark_series(broken, ref, "123451", "隔离指数*95%+活期税后*5%")
    with pytest.raises(ValueError, match="PRODUCT"):
        _benchmark_series(raw, ref, "123452", "隔离指数*95%+活期税后*5%")


def test_normalized_publisher_windows_require_daily_union_and_every_long_window_point():
    dates = ["2026-09-25", "2026-09-28", "2026-09-29", "2026-09-30"]
    prices = dict.fromkeys(dates, "100")
    windows = [{"first": dates[0], "last": dates[2], "levels": dict.fromkeys(dates[:3], "1")},
               {"first": dates[2], "last": dates[3], "levels": dict.fromkeys(dates[2:], "1")},
               {"first": dates[0], "last": dates[3], "levels": {dates[0]: "1", dates[3]: "1"}}]
    result = reconstruct_normalized_windows(dates, prices, windows, annual_cash_rate=".0035")
    assert result["published_valuation_dates"] == 4 and len(result["windows"]) == 3
    assert all(Decimal(v) > 1 for v in result["levels"][1:])
    assert "NOT_STITCHED" in result["scope"]
    missing = deepcopy(windows)
    missing[0]["levels"].pop(dates[1])
    with pytest.raises(ValueError, match="DAILY_COVERAGE"):
        reconstruct_normalized_windows(dates, prices, missing, annual_cash_rate=".0035")
    conflict = deepcopy(windows)
    conflict[-1]["levels"][dates[-1]] = "1.01"
    with pytest.raises(ValueError, match="PRECISION_CONFLICT"):
        reconstruct_normalized_windows(dates, prices, conflict, annual_cash_rate=".0035")
    conflict = deepcopy(windows)
    conflict[1]["levels"][dates[2]] = "1.0001"
    with pytest.raises(ValueError, match="WINDOW_SCOPE"):
        reconstruct_normalized_windows(dates, prices, conflict, annual_cash_rate=".0035")
