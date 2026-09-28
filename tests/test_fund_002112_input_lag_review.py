"""输入时间差核查的边界测试；只使用合成数据，无拟合或历史预测生成。"""

from datetime import date, timedelta

import pytest
from scripts import fund_002112_input_lag_review as review


def sample_report(end="2023-12-31", public="2024-03-30", full=True, sha="annual"):
    return {"report_end": end, "published_date": public, "report_type": "ANNUAL",
            "full_stock_disclosure": full, "raw": {"sha256": sha},
            "holdings": [{"stock_code": "ONE", "nav_weight_pct": 50}],
            "disclosed_nav_pct": 50, "stock_nav_pct": 60}


def quote_sample():
    # 合成的 21 个连续日期只用于算式；真实运行还必须匹配冻结交易日历。
    days = [(date(2024, 3, 1) + timedelta(days=i)).isoformat() for i in range(21)]
    quotes = {d: {"ONE": {"pct_chg": 1, "amount": 100}} for d in days}
    indices = {c: {"rows": {d: {"pct_chg": 0} for d in days}} for c in ("000300.SH", "000905.SH")}
    return days, quotes, indices


def test_publication_day_is_not_usable_until_following_day():
    old = sample_report()
    new = sample_report("2024-03-31", "2024-04-16", False, "quarter")
    assert review.legal_report([old, new], [old, new], "2024-04-16") == old
    assert review.legal_report([old, new], [old, new], "2024-04-17") == new


def test_revision_date_does_not_get_advanced():
    report = {**sample_report(), "revised_at": "2024-04-20T12:00:00+08:00"}
    assert review.publication(report) == "2024-04-20"


def test_missing_latest_report_prevents_fallback():
    old = sample_report()
    new = sample_report("2024-03-31", "2024-04-16", False, "quarter")
    with pytest.raises(ValueError, match="LATEST_REPORT_MISSING"):
        review.legal_report([old], [old, new], "2024-04-17")


def test_unknown_publication_is_not_inferred():
    report = sample_report()
    del report["published_date"]
    with pytest.raises(ValueError, match="PUBLICATION_UNKNOWN"):
        review.publication(report)


def test_independent_weight_formula_uses_nav_denominator():
    x = review.recompute_exposure(sample_report(), *quote_sample())
    assert x[0] == pytest.approx(0.005)
    assert x[1] == pytest.approx(0.5 * (1.01**5 - 1))
    assert x[2:7] == [0.5, 0, 0.25, 0.5, 0.6]
    assert x[8:] == [1, 0, 0, 0, 0]


def test_missing_quote_is_not_zero_or_forward_filled():
    days, quotes, indices = quote_sample()
    del quotes[days[5]]["ONE"]
    with pytest.raises(ValueError, match="POSITIVE_HOLDING_QUOTE_MISSING"):
        review.recompute_exposure(sample_report(), days, quotes, indices)


def test_shortened_window_is_rejected():
    days, quotes, indices = quote_sample()
    with pytest.raises(ValueError, match="COMPLETE_21_SESSION_WINDOW_REQUIRED"):
        review.recompute_exposure(sample_report(), days[1:], quotes, indices)


def test_explicit_zero_weight_does_not_require_fake_quote():
    report = sample_report()
    report["holdings"].append({"stock_code": "ROUNDED_ZERO", "nav_weight_pct": 0})
    assert review.recompute_exposure(report, *quote_sample())[5] == 0.5


def test_zero_turnover_baseline_is_rejected():
    days, quotes, indices = quote_sample()
    for d in days:
        quotes[d]["ONE"]["amount"] = 0
    with pytest.raises(ValueError, match="AMOUNT_BASE_NOT_POSITIVE"):
        review.recompute_exposure(sample_report(), days, quotes, indices)


def test_predictions_must_be_bound_to_exact_original_inputs():
    original = {"2024-04-01": {"actual_direction": "DOWN", "base": "2024-03-29", "x": [1]}}
    prediction = {"target": "2024-04-01", "actual_direction": "DOWN", "base": "2024-03-29",
                  "fund_code": "002112", "direction": "UP", "input_hash": "wrong"}
    with pytest.raises(ValueError, match="PREDICTION_INPUT_IDENTITY_MISMATCH"):
        review.prediction_index([prediction], original)
    prediction["input_hash"] = review.digest(original["2024-04-01"])
    assert len(review.prediction_index([prediction], original)) == 1
    with pytest.raises(ValueError, match="PREDICTION_DATE_MISMATCH"):
        review.prediction_index([prediction, prediction], original)
    with pytest.raises(ValueError, match="PREDICTION_DATE_MISMATCH"):
        review.prediction_index([], original)


def test_pairs_keep_gains_losses_and_flat_separate():
    rows = [
        {"target": "one", "actual_direction": "DOWN", "directions": {"A7": "UP", "B16": "DOWN", "C20": "DOWN"}},
        {"target": "two", "actual_direction": "UP", "directions": {"A7": "UP", "B16": "DOWN", "C20": "UP"}},
        {"target": "three", "actual_direction": "FLAT", "directions": {"A7": "UP", "B16": "DOWN", "C20": "UP"}},
    ]
    result = review.summarize(rows)
    assert result["pairs"]["B16_vs_A7"] == {
        "gained": 1, "lost": 1, "net": 0, "gained_dates": ["one"], "lost_dates": ["two"]}
    assert result["models"]["B16"]["class_correct"] == {"DOWN": 1, "FLAT": 0, "UP": 0}
    assert result["constant_correct"] == {"DOWN": 1, "FLAT": 1, "UP": 1}


def test_sources_cannot_be_silently_replaced(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("{}", encoding="utf-8")
    protocol = {"sources": {"data": {"path": str(path), "sha256": review.file_hash(path)}}}
    path.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        review.source_path(protocol, "data")
