import json
from copy import deepcopy
from datetime import date
from hashlib import sha256
from urllib.parse import urlencode

import pytest
from app.services import fund_rating_month_end as month_end
from app.services.fund_rating_pipeline import _period_covered


def test_bosera_requires_next_actual_valuation_and_same_share(monkeypatch, tmp_path):
    raw = {"retCode": "0", "data": {"title": "单位净值", "resultList": [
        {"20261030": "1.2"}, {"20261102": "1.3"}]}}
    class Source:
        def __init__(self, *args, **kwargs):
            pass
        def json(self, key):
            return raw
    monkeypatch.setattr(month_end, "Originals", Source)
    bundle = {"fund_code": "018853", "share_class": "C", "as_of_date": "2026-10-30",
        "monthly_valuation_proof": {"format": "BOSERA_NAV_YEAR_RANGE_V1", "fund_code": "018853",
            "as_of_date": "2026-10-30", "month_end": "2026-10-31", "next_valuation_date": "2026-11-02",
            "issuer_nav_source_ref": {"sha256": "raw", "status": 200,
                "url": "https://www.bosera.com/fund/fundNetValue.json", "params": {
                    "fundCode": "018853", "startDate": "20261001", "endDate": "20261116",
                    "netValueType": "NETVALUEPER"}}}}
    args = (bundle, date(2026, 10, 30), date(2026, 10, 31))
    assert month_end.official_month_end(*args, root=tmp_path)
    raw["data"]["resultList"].pop()
    with pytest.raises(ValueError, match="TERMINAL_OR_NEXT"):
        month_end.official_month_end(*args, root=tmp_path)


@pytest.fixture
def official(monkeypatch):
    product = "actual-product"
    info = {
        "code": 0,
        "data": {
            "product_info": {"product_code": "019449", "product_id": product, "product_legal_name": "日本基金(QDII)"}
        },
    }
    rows = [{"nav_date": day, "product_id": product} for day in ["20261029", "20261030", "20261102"]]
    page = {"pages": 1, "pageNum": 1, "pageSize": 3, "total": 3, "list": rows}
    nav = {"code": 0, "data": {"product_nav": {product: page}}}
    sources = {"nav": nav, "identity": info}

    class Source:
        def __init__(self, *args, **kwargs):
            pass

        def json(self, key):
            return sources[key]

    monkeypatch.setattr(month_end, "Originals", Source)
    proof = {
        "format": "CIFM_PRODUCT_NAV_COMPLETE_V1",
        "fund_code": "019449",
        "as_of_date": "2026-10-30",
        "month_end": "2026-10-31",
        "next_valuation_date": "2026-11-02",
        "issuer_nav_source_ref": {
            "sha256": "nav",
            "url": "https://ecmob.cifm.com/mall-data-server/product/queryProductNav",
            "params": {"product_id": product, "start_date": "20230928", "end_date": "20261102"},
        },
        "issuer_identity_source_ref": {
            "sha256": "identity",
            "url": "https://ecmob.cifm.com/mall-data-server/fundinfo/queryFundDetail",
            "params": {"product_code": "019449"},
        },
        "sequence": {
            "path": ["data", "product_nav", product, "list"],
            "date_field": "nav_date",
            "identity_field": "product_id",
        },
    }
    bundle = {
        "fund_code": "019449",
        "family": "QDII_EQUITY",
        "as_of_date": "2026-10-30",
        "identity": {"legal_name": "日本基金(QDII)"},
        "monthly_valuation_proof": proof,
    }
    return bundle, page, info


def test_actual_official_terminal_and_next_replay(official, tmp_path):
    bundle, _, _ = official
    assert month_end.official_month_end(bundle, date(2026, 10, 30), date(2026, 10, 31), root=tmp_path)


@pytest.mark.parametrize("failure", ["missing_next", "duplicate", "truncated", "wrong_share", "claimed_date"])
def test_official_sequence_rejects_candidate_claims(official, tmp_path, failure):
    bundle, page, identity = official
    if failure == "missing_next":
        page["list"].pop()
        page["total"] = 2
    elif failure == "duplicate":
        page["list"][0] = deepcopy(page["list"][1])
    elif failure == "truncated":
        page["pages"] = 2
    elif failure == "wrong_share":
        identity["data"]["product_info"]["product_code"] = "007280"
    else:
        bundle["monthly_valuation_proof"]["next_valuation_date"] = "2026-11-03"
    with pytest.raises(ValueError):
        month_end.official_month_end(bundle, date(2026, 10, 30), date(2026, 10, 31), root=tmp_path)


@pytest.mark.parametrize("failure", [None, "wrong_share", "truncated", "missing_next", "unofficial"])
def test_csrc_replays_exact_share_and_complete_cross_month_original(tmp_path, failure):
    code = "008764"
    args = {
        "fundCode": code, "iDisplayStart": "0", "iDisplayLength": "20",
        "startDate": "2026-10-26", "endDate": "2026-11-04",
    }
    rows = [
        {"code": code, "shortName": "天弘越南市场股票发起（QDII）C", "valuationDate": day, "shareNetValue": value}
        for day, value in [("2026-10-30", "1.5083"), ("2026-11-02", "1.4766")]
    ]
    if failure == "wrong_share":
        rows[0]["code"] = "008763"
    if failure == "missing_next":
        rows.pop()
    raw = json.dumps({"iTotalDisplayRecords": len(rows) + (failure == "truncated"), "aaData": rows}).encode()
    source = tmp_path / "sources" / "csrc.json"
    source.parent.mkdir()
    source.write_bytes(raw)
    host = "example.com" if failure == "unofficial" else "eid.csrc.gov.cn"
    url = f"http://{host}{month_end.CSRC_NAV_PATH}?" + urlencode({
        "aoData": json.dumps([{"name": key, "value": value} for key, value in args.items()]),
    })
    bundle = {
        "fund_code": code,
        "as_of_date": "2026-10-30",
        "monthly_valuation_proof": {
            "format": "CSRC_EID_SHARE_NAV_V1",
            "fund_code": code,
            "as_of_date": "2026-10-30",
            "month_end": "2026-10-31",
            "next_valuation_date": "2026-11-02",
            "official_nav_source_ref": {
                "file": "sources/csrc.json", "sha256": sha256(raw).hexdigest(), "url": url, "http_status": 200,
            },
        },
    }
    dates = (date(2026, 10, 30), date(2026, 10, 31))
    if failure:
        with pytest.raises(ValueError, match="MONTH_END_CSRC"):
            month_end.official_month_end(bundle, *dates, root=tmp_path)
    else:
        assert month_end.official_month_end(bundle, *dates, root=tmp_path)


@pytest.mark.parametrize(
    "profile,family", [("MIXED", "MIXED"), ("CONVERTIBLE", "CONVERTIBLE"), ("BOND_INDEX", "INDEX")]
)
def test_validated_domestic_types_use_actual_cn_month_end(monkeypatch, profile, family):
    from app.services import direction_1d_protocol

    monkeypatch.setattr(direction_1d_protocol, "calendar", lambda: ([date(2026, 10, 29), date(2026, 10, 30)], "test"))
    bundle = {
        "family": family,
        "identity": {"legal_name": "境内基金"},
        "typed_disclosure_calculation": {"profile": profile, "inputs": {"currency": "CNY"}},
    }
    assert _period_covered(bundle, date(2026, 10, 30), date(2026, 10, 31))
    assert not _period_covered(bundle, date(2026, 10, 29), date(2026, 10, 31))
    assert not _period_covered(bundle, date(2026, 10, 30), date(2026, 11, 30))
    bundle["identity"]["legal_name"] = "日本基金(QDII)"
    assert not _period_covered(bundle, date(2026, 10, 30), date(2026, 10, 31))
