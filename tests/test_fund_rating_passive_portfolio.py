"""ETF实际金额穿透与受限披露范围，不以一般正常描述充当全范围证明。"""

from decimal import Decimal

import pytest
from app.services.fund_rating_passive_portfolio import lookthrough


def test_etf_stock_risk_budget_keeps_underlying_cash_and_substitution_outside_stocks():
    link = {"balance": {"fund_amount": "900"}, "positions": [{"stock_code": "a", "market_value_cny": "100"}]}
    target = {"balance": {"net_assets": "1000", "fund_amount": "0"}, "positions": [
        {"stock_code": "a", "market_value_cny": "500"}, {"stock_code": "b", "market_value_cny": "400"}]}
    result = lookthrough(link, target, fund_holding_cny="900")
    assert Decimal(result["stock_total"]) == 910
    assert {r["stock_code"]: Decimal(r["market_value_cny"]) for r in result["positions"]} == {
        "a": Decimal(550), "b": Decimal(360)
    }
    with pytest.raises(ValueError, match="NOT_RECONCILED"):
        lookthrough(link, target, fund_holding_cny="800")
    target["balance"]["fund_amount"] = "10"
    with pytest.raises(ValueError, match="FURTHER_LOOKTHROUGH_REQUIRED"):
        lookthrough(link, target, fund_holding_cny="900")


def test_general_most_assets_liquid_statement_does_not_close_all_asset_restrictions(monkeypatch):
    from app.services import fund_rating_passive_portfolio as module

    monkeypatch.setattr(module, "periodic_financial_view", lambda pages: (pages, {}))
    monkeypatch.setattr(module, "strict_balance_values", lambda *a, **k: {
        "point_book": {"stock_amount": "0", "fund_amount": "100", "net_assets": "100", "bond_amount": "0"}
    })
    def unrelated_stock_parser(*_args):
        raise AssertionError("原文未闭合全资产范围，不应先用股票表错误掩盖真正缺项")

    monkeypatch.setattr(module, "gross_asset_book", unrelated_stock_parser)
    with pytest.raises(ValueError, match="FULL_ASSET_RESTRICTION_SCOPE_REQUIRED"):
        module.portfolio(["除6.4.12之外，本基金所持大部分投资标的具有良好流动性。"],
                         holding_date="2026-06-30", include_restrictions=True)
