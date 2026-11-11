from datetime import date

from app.services.fund_rating_pipeline import _period_covered


def test_closed_calendar_month_end_accepts_only_verified_final_session():
    bundle = {"family": "INDEX", "identity": {"legal_name": "境内股票指数基金"}}
    assert _period_covered(bundle, date(2026, 10, 30), date(2026, 10, 31))
    assert not _period_covered(bundle, date(2026, 10, 29), date(2026, 10, 31))
    assert not _period_covered(bundle, date(2026, 9, 30), date(2026, 10, 31))
    assert not _period_covered(bundle, date(2027, 1, 29), date(2027, 1, 31))


def test_newer_published_period_does_not_require_an_older_replacement():
    bundle = {"family": "ACTIVE_EQUITY", "identity": {"legal_name": "境内股票基金"}}
    assert _period_covered(bundle, date(2026, 11, 6), date(2026, 10, 31))
    foreign = {"family": "INDEX", "identity": {"legal_name": "纳斯达克QDII基金"}}
    assert not _period_covered(foreign, date(2026, 10, 30), date(2026, 10, 31))
