from datetime import date

import pytest
from app.services.fund_rating_monthly_typed import prepare_typed_monthly
from app.services.fund_rating_qdii_disclosure import own_history


def test_roll_requires_actual_new_period_review(tmp_path):
    with pytest.raises(ValueError, match="ACTUAL_PERIOD_REVIEW"):
        prepare_typed_monthly(
            "019449",
            {"fund_code": "019449"},
            "2026-10-30",
            tmp_path,
            tmp_path,
            replacements={},
            new_sources=[],
            period_review=None,
        )


@pytest.mark.parametrize("conflict", [False, True])
def test_japan_derived_auxiliary_still_matches_actual_issuer(monkeypatch, conflict):
    from app.services import fund_rating_qdii_disclosure as qdii

    days = [
        f"{year}-{month:02}-28"
        for year in range(2023, 2027)
        for month in range(1, 13)
        if "2023-10" <= f"{year}-{month:02}" <= "2026-10"
    ]
    days[0], days[-1] = "2023-10-31", "2026-10-30"
    rows = [{"nav_date": d, "unit_nav": "1", "adjusted_nav": "1", "content_hash": "real-observation"} for d in days]
    provider = {
        "fund_code": "019449",
        "source": {"enabled": True, "authorization_verified_at": "verified"},
        "history": {"rows": rows},
    }
    detail = {
        "code": 0,
        "data": {
            "product_info": {
                "product_code": "019449",
                "product_id": "id",
                "product_legal_name": "摩根日本精选股票型证券投资基金(QDII)",
                "money_type": "0",
                "found_date": "2023-09-08",
            }
        },
    }
    official = [
        {
            "nav_date": d.replace("-", ""),
            "product_id": "id",
            "relate_price": "1",
            "cumulative_net": "1",
            "adjust_nav": "1",
        }
        for d in days
    ]
    if conflict:
        official[-1]["relate_price"] = "2"
    nav = {
        "code": 0,
        "data": {
            "product_nav": {
                "id": {"pages": 1, "pageNum": 1, "pageSize": len(official), "total": len(official), "list": official}
            }
        },
    }

    class Source:
        root = None

        def json(self, ref):
            return {"detail": detail, "nav": nav}[ref]

    monkeypatch.setattr(qdii, "auxiliary_json", lambda *args: provider)
    inputs = {
        "adapter": "CIFM_JAPAN",
        "fund_code": "019449",
        "as_of_date": "2026-10-30",
        "auxiliary_archives": {"provider_crosscheck": {}},
        "issuer_detail_ref": "detail",
        "issuer_nav_ref": "nav",
    }
    if conflict:
        with pytest.raises(ValueError, match="NAV_SOURCE_CONFLICT"):
            own_history(Source(), inputs)
    else:
        values, proof = own_history(Source(), inputs)
        assert max(values) == str(date(2026, 10, 30))
        assert proof["all_unit_navs_matched"] is True
