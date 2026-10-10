"""各类原始输入适配与时间边界：纯隔离数据，不导入日常评级表。"""

import calendar
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from app.services.fund_rating_inputs import build_metrics
from app.services.fund_rating_metrics import stability
from app.services.fund_rating_rules import candidate
from app.services.fund_rating_specialized import distance, distribution_index

NOW = datetime(2026, 10, 10, tzinfo=UTC)
AS_OF = date(2026, 9, 30)


def test_classification_matches_actual_valuation_day(monkeypatch):
    """月末不是估值日时仍按原件日期查分类，避免把合法资料归入未核验类别。"""
    import app.services.fund_rating_inputs as inputs

    payload = {"found_date": "2020-01-01"}
    evidence = SimpleNamespace(
        fund_code="002112",
        classification_id="class",
        as_of_date=date(2026, 9, 29),
        payload=payload,
        evidence_hash=inputs.digest(payload),
    )
    classification = SimpleNamespace(
        classification_id="class",
        effective_from=date(2020, 1, 1),
        effective_to=date(2026, 9, 29),
        category_code="verified",
        product_id="product",
        family="ACTIVE_EQUITY",
    )

    class ScalarRows(list):
        def all(self):
            return self

    answers = iter([[evidence], [classification]])
    session = SimpleNamespace(scalars=lambda _: ScalarRows(next(answers)))
    monkeypatch.setattr(inputs, "validate_sources", lambda *_: None)
    monkeypatch.setattr(inputs, "build_metrics", lambda *_: ({}, {}, NOW + timedelta(days=1)))
    groups = inputs.prepare(session, [{"fund_code": "002112", "fund_name": "测试", "fund_type": "MIXED"}], AS_OF, NOW)
    assert groups["verified"][0]["admitted"] is True
    assert groups["verified"][0]["valuation_date"] == "2026-09-29"


def bundle(family="ACTIVE_EQUITY"):
    ends = []
    for n in range(37):
        serial = 2023 * 12 + 8 + n
        year, month0 = divmod(serial, 12)
        d = date(year, month0 + 1, calendar.monthrange(year, month0 + 1)[1])
        while d.weekday() > 4:
            d -= timedelta(days=1)
        ends.append(d)
    dates = [ends[0] + timedelta(days=i) for i in range((AS_OF - ends[0]).days + 1)]
    dates = [d for d in dates if d.weekday() < 5]
    currency = "USD" if family.startswith("QDII") else "CNY"
    history = {
        "dates": list(map(str, dates)),
        "valuation_calendar": list(map(str, dates)),
        "as_of_month_calendar": [str(d) for d in dates if d.month == 9 and d.year == 2026],
        "month_ends": list(map(str, ends)),
        "annual_days": 365 if family == "MONEY" else 252,
        "currency": currency,
        "risk_free_currency": currency,
        "unit_nav": [str(Decimal(100) + Decimal(i) / 20 + Decimal(i % 7) / 10) for i in range(len(dates))],
        "cash_dividend_per_share": [0] * len(dates),
        "split_factor": [1] * len(dates),
        "risk_free_interval_return": [".00001"] * (len(dates) - 1),
        "distribution_per_unit": [".00005"] * (len(dates) - 1),
        "distribution_base_unit": [1] * (len(dates) - 1),
    }
    position = {
        "amount": 80,
        "saleable_amount": 80,
        "issuer": "A",
        "industry": "I",
        "turnover_20d": [100] * 20,
        "stress_default_probability": ".01",
        "loss_given_default": ".6",
        "modified_duration": 3,
        "convexity": 5,
        "lower_credit": False,
        "equity_delta": ".5",
        "conversion_premium": ".1",
        "maturing_or_redeemable_5d": 20,
        "redeemable_5d": 40,
        "underlying_weights": {"A": ".7", "B": ".3"},
        "underlying_cost_1y": ".01",
        "waived_cost_1y": ".003",
        "cost_already_in_outer": False,
    }
    value = {
        "family": family,
        "as_of_date": str(AS_OF),
        "currency": currency,
        "history": history,
        "holdings": {
            "report_date": "2026-06-30",
            "currency": currency,
            "amount_unit": "YUAN" if currency == "CNY" else "CURRENCY_UNIT",
            "positions": [position],
            "net_assets": 100,
            "cash": 20,
            "payable_5d": 0,
            "largest_holder": ".1",
        },
        "manager": {
            "checked_at": NOW.isoformat(),
            "effective_date": "2020-01-01",
            "teams": [{"effective_date": "2020-01-01", "members": ["团队甲"]}],
        },
        "fees": {
            "checked_at": NOW.isoformat(),
            "effective_date": "2020-01-01",
            "trading_form": "OTC",
            "redemption_holding_days": 365,
            "management": ".01",
            "custody": ".001",
            "sales": 0,
            "subscription": 0,
            "redemption": 0,
            "subscription_mode": "EXTERNAL",
        },
        "next_update_calendar": ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06", "2026-11-09"],
        "classification": {"peer": {"target_index": "TEST"}},
        "specialized": {
            "actual_index_weights": {"A": 1},
            "target_index_weights": {"A": ".8", "B": ".2"},
            "actual_allocation": {"stock": ".6", "bond": ".4"},
            "target_allocation": {"stock": ".5", "bond": ".5"},
            "roll_cost_amount": ".5",
            "fx_basis_verified": True,
            "timezone_verified": True,
            "overseas_calendar_verified": True,
            "redemption_limits_verified": True,
            "benchmark": {
                "dates": history["dates"],
                "currency": currency,
                "total_return_verified": True,
                "target_index": "TEST",
                "index": history["unit_nav"],
            },
        },
    }
    for key in (
        "events_complete",
        "calendar_verified",
        "full_holdings",
        "manager_history_complete",
        "fees_complete",
        "strategy_verified",
        "latest_report_verified",
        "retail_eligible",
    ):
        value[key] = True
    return value


@pytest.mark.parametrize(
    "family",
    [
        "ACTIVE_EQUITY",
        "BOND",
        "SHORT_BOND",
        "MIXED_BOND",
        "CONVERTIBLE",
        "INDEX",
        "MONEY",
        "QDII_EQUITY",
        "QDII_BOND",
        "QDII_COMMODITY",
        "FOF",
    ],
)
def test_each_family_supplies_all_eight_specific_metrics(family):
    metrics, dates, expiry = build_metrics(bundle(family), AS_OF, NOW)
    member = {"family": family, "metrics": metrics, "representative": True}
    stability([member])
    for terms in candidate(family)["formula"].values():
        assert all(metrics[key] is not None for key, _, _ in terms)
    assert dates["holdings"] == "2026-06-30" and expiry > NOW


@pytest.mark.parametrize(
    "key",
    [
        "events_complete",
        "calendar_verified",
        "full_holdings",
        "manager_history_complete",
        "fees_complete",
        "strategy_verified",
        "latest_report_verified",
        "retail_eligible",
    ],
)
def test_missing_required_verification_blocks(key):
    value = bundle()
    value[key] = False
    with pytest.raises(ValueError, match="REQUIRED_VERIFICATION"):
        build_metrics(value, AS_OF, NOW)


def test_effective_dates_currency_and_calendar_gaps():
    value = bundle()
    value["fees"]["effective_date"] = "2026-10-01"
    with pytest.raises(ValueError, match="FUTURE_EFFECTIVE"):
        build_metrics(value, AS_OF, NOW)
    value = bundle()
    value["history"]["risk_free_currency"] = "USD"
    with pytest.raises(ValueError, match="CURRENCY"):
        build_metrics(value, AS_OF, NOW)
    value = bundle()
    value["history"]["dates"].pop(100)
    with pytest.raises(ValueError, match="CALENDAR_GAP"):
        build_metrics(value, AS_OF, NOW)
    value = bundle()
    value["holdings"]["report_date"] = "2025-12-31"
    with pytest.raises(ValueError, match="DATE_UNIT"):
        build_metrics(value, AS_OF, NOW)


def test_money_uses_distribution_and_fof_waivers_are_not_double_counted():
    value = bundle("MONEY")
    value["history"]["unit_nav"] = [1] * len(value["history"]["dates"])
    metrics, _, _ = build_metrics(value, AS_OF, NOW)
    assert metrics["sharpe"] is None and metrics["distribution_return_1y"] > 0
    assert distribution_index({"dates": ["a", "b"], "distribution_per_unit": [1], "distribution_base_unit": [100]}) == [
        1,
        Decimal("1.01"),
    ]
    value = bundle("FOF")
    metrics, _, _ = build_metrics(value, AS_OF, NOW)
    assert metrics["lookthrough_cost_1y"] == Decimal(".018")
    value["holdings"]["positions"][0]["cost_already_in_outer"] = True
    with pytest.raises(ValueError, match="DOUBLE_COUNT"):
        build_metrics(value, AS_OF, NOW)


def test_index_target_mismatch_and_unknown_foreign_basis_never_fallback():
    value = bundle("INDEX")
    value["specialized"]["benchmark"]["target_index"] = "DIFFERENT"
    with pytest.raises(ValueError, match="TARGET"):
        build_metrics(value, AS_OF, NOW)
    value = bundle("QDII_EQUITY")
    del value["specialized"]["fx_basis_verified"]
    with pytest.raises(ValueError, match="OVERSEAS"):
        build_metrics(value, AS_OF, NOW)
    assert distance({"A": 1}, {"A": ".5", "B": ".5"}) == Decimal(".5")
