"""指数联接公开评级的隔离测试；算例不是016008真实资料或实际评级。"""

from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace

import pytest
from app.services.fund_rating_inputs import digest
from app.services.fund_rating_public_etf import _amount, build_etf_metrics, etf_policy
from app.services.fund_rating_public_inputs import validate_public_sources
from app.services.fund_rating_public_policy import ETF_BASIS, calculate_public, policy
from app.services.fund_rating_public_validation import independent_score, sensitivity
from app.services.fund_rating_rules import DIMENSIONS

from test_fund_rating_inputs import AS_OF, NOW
from test_fund_rating_public import public_bundle, sample_metrics


def etf_bundle(monkeypatch):
    """造完整隔离结构以验证日期、资产对账及跟踪边界，不写入任何业务表。"""
    b = public_bundle(monkeypatch)
    b.update(basis=ETF_BASIS, family="INDEX", fund_code="016008")
    b["key_information"]["fund_code"] = "016008"
    b["classification"] = {
        "scope": {
            "basis": ETF_BASIS,
            "asset": "DOMESTIC_EQUITY_ETF_FEEDER",
            "currency": "CNY",
            "target_fund": "159779",
            "target_index": "中证消费电子主题指数",
            "share_class": "C",
        }
    }
    positions = b["holdings"]["positions"]
    total = sum(Decimal(p["amount"]) for p in positions)
    target_balance = {
        "assets": {"其中：股票投资": str(total), "货币资金": "20"},
        "total_assets": str(total + 20),
        "liabilities": "0",
        "net_assets": str(total + 20),
    }
    link_balance = {
        "assets": {"基金投资": "900", "货币资金": "100"},
        "total_assets": "1000",
        "liabilities": "0",
        "net_assets": "1000",
    }
    b["holdings"] = {
        "positions": positions,
        "industries": [{"code": "C", "amount": str(total)}],
        "balance": target_balance,
        "link_balance": link_balance,
        "report_date": str(AS_OF),
        "net_assets": "1000",
        "target_fund_amount": "900",
        "restricted_stock_amount": "10",
    }
    b["history"]["benchmark_index"] = b["history"]["unit_nav"][:]
    return b


def test_equity_method_is_unchanged_and_index_scope_is_distinct():
    assert digest(policy()) == "bef1081767f026c53759dee757f756bfd1dcd67aefeb0de519af8318bc4fc5d7"
    p = etf_policy()
    assert p["family"] == "INDEX" and set(p["weights"]) == set(DIMENSIONS)
    assert sum(p["weights"].values()) == 100 and p["weights"]["management"] == 25
    assert p["scope"]["target_fund"] == "159779"


@pytest.mark.parametrize("dimension", list(DIMENSIONS))
def test_each_index_dimension_remains_mandatory(dimension):
    p = etf_policy()
    m = {**sample_metrics(), "tracking_error_ratio": ".5", "tracking_bias_ratio": ".2"}
    del m[p["terms"][dimension][0][0]]
    with pytest.raises(KeyError):
        calculate_public(m, config=p)


def test_tracking_contract_is_used_without_changing_equity_policy():
    p = etf_policy()
    m = {**sample_metrics(), "tracking_error_ratio": ".5", "tracking_bias_ratio": ".2"}
    result = calculate_public(m, config=p)
    total, _ = independent_score(m, p)
    assert abs(total - float(result["score"])) < 1e-10
    worse = {**m, "tracking_error_ratio": "2"}
    assert Decimal(calculate_public(worse, config=p)["score"]) < Decimal(result["score"])
    assert len(sensitivity(m, result, config=p)["weight_20_percent"]) == 16


def test_tracking_and_lookthrough_are_not_stock_or_holder_placeholders(monkeypatch):
    b = etf_bundle(monkeypatch)
    m, _, _ = build_etf_metrics(b, AS_OF, NOW)
    assert m["tracking_error"] == 0 and m["tracking_bias"] == 0
    assert m["industry_hhi"] == 1 and "holder_risk_band" not in m
    assert m["restricted_ratio"] > 0 and m["cash_ratio"] == Decimal(".1")
    assert m["net_assets"] == 1000 and set(calculate_public(m, config=etf_policy())["dimensions"]) == set(DIMENSIONS)


@pytest.mark.parametrize(
    "change", ["wrong_fund", "wrong_family", "missing_benchmark", "industry_gap", "unbalanced_assets"]
)
def test_ambiguous_or_incomplete_etf_inputs_cannot_be_scored(monkeypatch, change):
    b = etf_bundle(monkeypatch)
    if change == "wrong_fund":
        b["fund_code"] = "016007"
    elif change == "wrong_family":
        b["family"] = "ACTIVE_EQUITY"
    elif change == "missing_benchmark":
        b["history"]["benchmark_index"] = []
    elif change == "industry_gap":
        b["holdings"]["industries"][0]["amount"] = "0"
    else:
        b["holdings"]["link_balance"]["assets"]["货币资金"] = "101"
    with pytest.raises(ValueError):
        build_etf_metrics(b, AS_OF, NOW)


def test_statement_current_column_and_explicit_dash():
    assert _amount("货币资金 6.4.7.1 12.34 56.78", "货币资金") == "12.34"
    assert _amount("货币资金 - 56.78", "货币资金") == "0"
    with pytest.raises(ValueError):
        _amount("货币资金 未披露 56.78", "货币资金")


def test_public_archive_aliases_preserve_both_empty_source_receipts(tmp_path, monkeypatch):
    import hashlib

    from app.services import fund_rating_public_inputs as inputs

    monkeypatch.setattr(inputs, "get_settings", lambda: SimpleNamespace(app_environment="local"))
    raw = b'{"code":0,"data":[]}'
    key = hashlib.sha256(raw).hexdigest()
    path = tmp_path / "sources" / (key + ".json")
    path.parent.mkdir()
    path.write_bytes(raw)
    ref = {
        "sha256": key,
        "file": "sources/" + path.name,
        "url": "https://www.cmfchina.com/ws-business-server/fund/getFundMelonmd",
        "acquired_at": NOW.isoformat(),
    }
    ref["aliases"] = [{**ref, "url": "https://www.cmfchina.com/ws-business-server/fund/getFundSplit"}]
    b = {
        "basis": ETF_BASIS,
        "fund_code": "016008",
        "source_scope": "PUBLIC_DISCLOSURE_LOCAL_ANALYSIS",
        "report_hash": key,
        "summary_hash": key,
        "classification": {"source_refs": [key]},
        "raw_archive": [ref],
    }
    validate_public_sources(b, NOW, root=tmp_path)
    bad = deepcopy(b)
    bad["raw_archive"][0]["aliases"][0]["url"] = "https://unverified.example/report"
    with pytest.raises(ValueError, match="SOURCE_HOST"):
        validate_public_sources(bad, NOW, root=tmp_path)
    bad = deepcopy(b)
    bad["raw_archive"][0]["aliases"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SOURCE_ALIAS"):
        validate_public_sources(bad, NOW, root=tmp_path)
