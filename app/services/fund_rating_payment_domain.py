"""付款用途域的等价计算建议：银行可用本金覆盖全部账面负债，无需虚填T/S。

只能由同日报表、现金受限范围及真实本金编译。它不认证未来赎回安全，
也不改变原R5的75锚点或付款公式；未知账户用途保持未知。
"""

import re
from datetime import date
from decimal import Decimal as D
from pathlib import Path

from app.services.fund_rating_r5_rules import amount

DOMAIN = "BANK_PRINCIPAL_DOMINATES_ALL_BOOK_LIABILITIES_V1"


def require(ok, code):
    if not ok:
        raise ValueError("R5_PAYMENT_DOMAIN_" + code)


def payment_domain_score(node):
    """在全部合法T≥0、0≤S≤P上求原付款分；不创造实际证券账户余额或用途。

    B先扣所有已披露银行限制；若扣后B<P，域结果不唯一，必须拒绝。
    证券账户T指扣除已披露限制后的可用专用本金，本证明不需要使用它。
    """
    require(node.get("kind") == "purpose_covered_payment", "NODE_KIND")
    require(node.get("coverage_domain") == DOMAIN, "DOMAIN_KIND")
    require(
        node.get("scope_closed") is True
        and node.get("restriction_amount_pending") is False
        and node.get("currency") == "CNY"
        and isinstance(node.get("as_of"), str),
        "SCOPE_OR_DATE",
    )
    date.fromisoformat(node["as_of"])
    require(
        all(node.get(key) is None for key in ["broker_settlement_principal", "settlement_liabilities"]),
        "UNKNOWN_SCALAR_MUST_NOT_BE_ASSIGNED",
    )
    bank, payable = amount(node.get("bank_principal")), amount(node.get("all_book_liabilities"))
    restrictions = node.get("disclosed_restrictions", [])
    require(isinstance(restrictions, list), "RESTRICTION_SCHEMA")
    for restriction in restrictions:
        locked = amount(restriction.get("amount"))
        account = restriction.get("account")
        require(account in {"bank", "broker"}, "UNKNOWN_RESTRICTION_ACCOUNT")
        if account == "bank":
            bank -= locked
    require(bank >= payable >= 0, "AVAILABLE_BANK_DOES_NOT_COVER_ALL_PAYABLES")
    return D(75), {
        "bank_principal_after_disclosed_restrictions": str(bank),
        "all_book_liabilities": str(payable),
        "covered_total": str(payable),
        "covered_via_broker": None,
        "covered_via_bank": None,
        "broker_settlement_principal": None,
        "settlement_liabilities": None,
        "unknown_T_S_not_assigned": True,
        "T_available_domain": {"lower": "0", "upper": None},
        "S_domain": {"lower": "0", "upper": str(payable)},
        "proof": "v=min(T,S),0<=v<=P;B>=P implies min(B,P-v)=P-v;covered=v+P-v=P.",
        "covered_ratio": "1",
        "point": "75",
        "not_payment_failure_inferred": True,
        "not_future_redemption_safety": True,
    }


def compile_payment_from_primary(inputs, *, root):
    """从018590本C份额的2026H1完整原PDF编译域节点，不接受派生金额或正常state。"""
    from app.services.fund_quality_v2_disclosures import report_period
    from app.services.fund_rating_active_execution import MONEY, Originals, _plain, _section, _share_binding

    require(
        inputs.get("fund_code") == "018590"
        and inputs.get("share_class") == "C"
        and inputs.get("holding_date") == "2026-06-30"
        and inputs.get("as_of_date") == "2026-09-30",
        "COMPONENT_SCOPE",
    )
    require(
        not any(k in inputs for k in ["score", "state", "bank_principal", "all_book_liabilities"]), "RAW_INPUT_REQUIRED"
    )
    digest = inputs["primary_ref"]
    source = Originals({**inputs, "sources": {digest: inputs["sources"][digest]}}, Path(root))
    text = source.body(digest)
    plain = _plain(text)
    require(
        _plain(inputs["legal_name"]) in plain[:18000]
        and _plain(inputs["manager_name"]) in plain[:18000]
        and str(report_period(text)[1]) == inputs["holding_date"],
        "PRIMARY_IDENTITY_OR_ECONOMIC_PERIOD",
    )
    _share_binding(source, inputs)
    balance = _section(text, r"6\.1\s*资产负债表", r"6\.2\s*利润表")
    note = _section(text, r"6\.4\.7\.1\s*货币资金", r"6\.4\.7\.2\s*交易性金融资产")
    liabilities = re.search(r"负债合计\s+(" + MONEY + r")", balance)
    bank = re.search(r"等于[：:]本金\s+(" + MONEY + r")", note)
    require(liabilities is not None and bank is not None, "ORIGINAL_BALANCE_ROWS")
    restricted = _section(text, r"6\.4\.12\s*期末", r"6\.4\.13\s*金融工具")
    headings = ["6.4.12.1", "6.4.12.2", "6.4.12.3.1", "6.4.12.3.2", "6.4.12.4"]
    offsets = [restricted.rfind(heading) for heading in headings]
    require(all(i >= 0 for i in offsets) and offsets == sorted(set(offsets)), "RESTRICTION_SECTIONS")
    for i, start in enumerate(offsets):
        end = offsets[i + 1] if i + 1 < len(offsets) else len(restricted)
        section = _plain(restricted[start:end])
        # “6.4.12.3”等章节编号不是金额；保留所有真实金额再检查非零受限表。
        section = re.sub(r"\d+(?:\.\d+){2,}", "", section)
        require("无。" in section and re.search(MONEY, section) is None, "ACTUAL_RESTRICTIONS_REQUIRE_ADAPTER")
    whole_asset_clause = "除在附注6.4.12中列示的部分基金资产流通暂时受限制外(如有),在正常市场条件下其余均能及时变现"
    require(whole_asset_clause in plain, "WHOLE_ASSET_RESTRICTION_SCOPE")
    node = {
        "kind": "purpose_covered_payment",
        "coverage_domain": DOMAIN,
        "bank_principal": str(amount(bank[1].replace(",", ""))),
        "all_book_liabilities": str(amount(liabilities[1].replace(",", ""))),
        "currency": "CNY",
        "as_of": inputs["holding_date"],
        "scope_closed": True,
        "restriction_amount_pending": False,
        "disclosed_restrictions": [],
        "source_refs": [digest],
    }
    value, proof = payment_domain_score(node)
    return {
        "node": node,
        "point": str(value),
        "proof": proof,
        "source_hash": digest,
        "bank_original_row": bank[0],
        "liabilities_original_row": liabilities[0],
        "restriction_original_sections": restricted,
        "whole_asset_restriction_clause": whole_asset_clause,
    }
