from copy import deepcopy

import pytest
from app.services.fund_prediction_market_recovery import FIELDS, quote_row, refresh


def response(code="000300.SH", day="20261110", close=101, pct=1):
    return {"code": 0, "data": {"fields": FIELDS, "items": [[code, day, close, 100, pct, 12]]}}


@pytest.mark.parametrize("raw", [response(day="20261109"), response(code="000905.SH"), response(pct=2)])
def test_old_wrong_or_inconsistent_quote_is_rejected(raw):
    with pytest.raises(ValueError):
        quote_row(raw, "000300.SH", "2026-11-10", "index_daily")


def test_source_failure_preserves_old_quote_and_success_binds_real_raw_receipt():
    material = {
        "fund_code": "123456",
        "market": {"000300.SH": {"date": "2026-11-04", "close": 9}},
        "documents": [],
        "facts": {},
        "comprehensive_coverage": {},
    }

    class Provider:
        failed = True

        def query(self, *args):
            if self.failed:
                return response(day="20261109"), {}
            return response(), {
                "sha256": "a" * 64,
                "file": "sources/test.json",
                "received_at": "2026-11-11T09:00:00+08:00",
            }

    provider = Provider()
    original = deepcopy(material)
    assert refresh(material, "2026-11-10", provider=provider)[0]["status"] == "ACTION_REQUIRED"
    assert material == original
    provider.failed = False
    assert refresh(material, "2026-11-10", provider=provider)[0]["status"] == "SUCCEEDED"
    assert material["market"]["000300.SH"]["date"] == "2026-11-10"
    assert material["documents"][0]["raw_sha256"] == "a" * 64
    assert "20261110" in material["facts"]["current_reference_quote:000300.SH"]["quotes"][0]


def test_bond_index_missing_turnover_is_preserved_as_unknown():
    raw = response()
    raw["data"]["items"][0][-1] = None
    assert quote_row(raw, "000300.SH", "2026-11-10", "index_daily")["amount"] is None
