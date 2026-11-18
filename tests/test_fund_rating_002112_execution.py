from decimal import Decimal

import pytest

from app.services.fund_rating_002112_execution import combined_issuer_exposure, qualified_cash, compile_observations, positive_compliance


def test_stock_and_corporate_bond_are_combined_under_same_legal_issuer():
    rows = [dict(security="600001", amount_cny="8"), dict(security="110001", amount_cny="4")]
    result = combined_issuer_exposure(rows, {"600001": "甲公司", "110001": "甲公司"}, "100")
    assert result["maximum_pct"] == "12.00"
    assert result["amounts"] == {"甲公司": "12"}


def test_same_controller_is_not_same_legal_issuer():
    rows = [dict(security="600001", amount_cny="8"), dict(security="600002", amount_cny="4")]
    result = combined_issuer_exposure(rows, {"600001": "甲公司", "600002": "乙公司"}, "100")
    assert Decimal(result["maximum_pct"]) == 8


def test_unknown_bond_issuer_cannot_disappear_behind_stock_only_check():
    result = combined_issuer_exposure([dict(security="110001", amount_cny="40")], {}, "100")
    assert result["missing_securities"] == ["110001"]


def test_explicit_government_is_not_a_company_but_unidentified_bond_is_unknown():
    result = combined_issuer_exposure([
        dict(security="019709", amount_cny="30", issuer_kind="CENTRAL_GOVERNMENT"),
        dict(security="123456", amount_cny="20")], {}, "100")
    assert result["missing_securities"] == ["123456"]


def test_cash_uses_actual_maturity_and_principal_not_all_bond_market_value():
    value, missing = qualified_cash("3", [
        dict(security="1", amount_cny="2", maturity="2024-07-25"),
        dict(security="2", amount_cny="80", maturity="2026-07-25")], "2024-06-30")
    assert value == 5 and not missing


def test_unknown_or_matured_bond_cannot_be_cash():
    value, missing = qualified_cash("3", [dict(security="1", amount_cny="20"),
        dict(security="2", amount_cny="30", maturity="2024-06-30")], "2024-06-30")
    assert value == 3 and missing == ["1"]


def test_other_fund_or_external_state_cannot_open_adapter(tmp_path):
    with pytest.raises(ValueError, match="002112_EXEC_SCOPE"):
        compile_observations(dict(fund_code="004605", execution_originals=dict(version="manual75")), root=tmp_path)


def test_normal_statement_does_not_hide_material_exception():
    positive = "依据基金合同，基金运作合法合规，无损害基金份额持有人利益的行为。"
    assert positive_compliance(positive)
    assert not positive_compliance(positive + "但是已见超限尚未整改。")
    assert not positive_compliance("依据基金合同，基金运作不合法合规。")
