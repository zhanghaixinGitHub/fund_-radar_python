"""域边界验证：未知用途不填0，任何银行限制先扣；不拿样例充实际T/S。"""

import sys
from copy import deepcopy
from decimal import Decimal as D

import pytest
from app.services.fund_rating_r5_rules import payment_score
from app.services.fund_rating_payment_domain import DOMAIN, payment_domain_score


def node(**changes):
    return {
        "kind": "purpose_covered_payment",
        "coverage_domain": DOMAIN,
        "scope_closed": True,
        "restriction_amount_pending": False,
        "currency": "CNY",
        "as_of": "2026-06-30",
        "bank_principal": "120",
        "all_book_liabilities": "100",
        **changes,
    }


@pytest.mark.parametrize("payable", ["0", "100"])
def test_domain_matches_original_formula_every_sample_without_actual_T_S(payable):
    raw = node(all_book_liabilities=payable)
    point, proof = payment_domain_score(raw)
    p = D(payable)
    for t, s in [(D(0), D(0)), (D(0), p), (p / 2, p), (p * 2, p), (p, p / 2)]:
        scalar = {**raw, "broker_settlement_principal": str(t), "settlement_liabilities": str(s)}
        del scalar["coverage_domain"]
        assert payment_score(scalar)[0] == point == 75
    assert proof["broker_settlement_principal"] is None and proof["settlement_liabilities"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"bank_principal": "99.99"},
        {"bank_principal": None},
        {"bank_principal": True},
        {"all_book_liabilities": "NaN"},
        {"scope_closed": False},
        {"restriction_amount_pending": True},
        {"currency": "USD"},
        {"coverage_domain": "OTHER"},
        {"broker_settlement_principal": "0"},
        {"settlement_liabilities": "0"},
        {"disclosed_restrictions": [{"account": "bank", "amount": "21"}]},
        {"disclosed_restrictions": [{"account": "bank", "amount": None}]},
        {"disclosed_restrictions": [{"account": "unknown", "amount": "0"}]},
    ],
)
def test_incomplete_or_nonconstant_domains_are_rejected(changes):
    with pytest.raises((ValueError, KeyError)):
        payment_domain_score(node(**changes))


def test_disclosed_bank_restrictions_are_deducted_before_testing_dominance():
    raw = node(disclosed_restrictions=[{"account": "bank", "amount": "20"}])
    value, proof = payment_domain_score(raw)
    assert value == 75 and proof["bank_principal_after_disclosed_restrictions"] == "100"
    changed = deepcopy(raw)
    changed["disclosed_restrictions"][0]["amount"] = "20.01"
    with pytest.raises(ValueError, match="AVAILABLE_BANK_DOES_NOT_COVER"):
        payment_domain_score(changed)


def test_proposed_shared_dispatch_accepts_domain_without_changing_original_scalar_formula(monkeypatch):
    from app.services import fund_rating_payment_domain as helper
    from app.services import fund_rating_r5_rules as candidate

    monkeypatch.setitem(sys.modules, "app.services.fund_rating_payment_domain", helper)
    assert candidate.payment_score(node())[0] == 75
    raw = node(bank_principal="30", broker_settlement_principal="20", settlement_liabilities="50")
    raw.pop("coverage_domain")
    assert candidate.payment_score(raw)[0] == payment_score(raw)[0] == D("37.5")


def test_proposed_dispatch_does_not_ignore_unknown_domain(monkeypatch):
    from app.services import fund_rating_payment_domain as helper
    from app.services import fund_rating_r5_rules as candidate

    monkeypatch.setitem(sys.modules, "app.services.fund_rating_payment_domain", helper)
    with pytest.raises(ValueError, match="DOMAIN_KIND"):
        candidate.payment_score(node(coverage_domain="arbitrary-point-75"))
