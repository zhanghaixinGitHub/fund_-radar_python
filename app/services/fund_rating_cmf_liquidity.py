"""招商消费电子联接C的同日完整流动性原件适配。

基础结构、实际申赎、普通/大额赎回机制及当期履职均须逐项核准。
单个持有人集中程度没有公开确认时保持未知；仅在所有相容结构均通过
原分档前提且得到相同点值时识别分值，不把未知结构写成分散或零。
本模块不联网、不写库、不接收预先计算的维度分或总分。
"""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import date
from decimal import Decimal, localcontext

from app.services.fund_quality_v2_accounting import accounting_observations
from app.services.fund_quality_v2_balance_proofs import identify_nonnegative_parts
from app.services.fund_quality_v2_stock_book import periodic_financial_view
from app.services.fund_rating_direct_index import (
    LIABILITIES,
    Originals,
    _flows,
    _liabilities,
    holder_execution_evidence,
)
from app.services.fund_rating_direct_index import compile_observations as standard_compile
from app.services.fund_rating_market_review import normalized_text
from app.services.fund_rating_passive_portfolio import portfolio

PRIMARY_SHA256 = "19277ca7387b21fb4e6d7c645749615612071688b5e396ad3c71d8562794bcca"


def verified_pages(source: Originals, primary_hash: str) -> list[str]:
    """只处理本基金本期已由Originals逐字节核准的公开原件。"""
    if primary_hash != PRIMARY_SHA256 or primary_hash not in source.raw:
        raise ValueError("R5_CMF_016008_PRIMARY_SOURCE_MISMATCH")
    pages = source.pages(primary_hash)
    plain = normalized_text("\n".join(pages))
    if not all(
        text in plain
        for text in ["016008", "016007", "2026年6月30日", "招商中证消费电子主题交易型开放式指数证券投资基金联接基金"]
    ):
        raise ValueError("R5_CMF_016008_IDENTITY_OR_PERIOD_MISMATCH")
    return pages


def liability_signed_comparison(source: Originals, primary_hash: str, total: str) -> dict:
    """本期非负负债与上年可有符号比较列分开读取，保留全部原始单元格。"""
    pages = verified_pages(source, primary_hash)
    view, _ = periodic_financial_view(pages)
    try:
        return {**_liabilities(view, total), "layout_adapter": None}
    except ValueError as error:
        if str(error) != "R5_DIRECT_INDEX_LIABILITY_ROW:卖出回购金融资产款":
            raise
    body = "\n".join(view)
    start, stop = body.find("短期借款"), body.find("负债合计")
    if start < 0 or stop <= start:
        raise ValueError("R5_CMF_LIABILITY_BODY")
    body = body[start:stop]
    slot = r"(?:[\d,]+\.\d{2}|[-－—])"
    comparison_slot = r"(?:-?[\d,]+\.\d{2}|[-－—])"
    parts, cells, comparisons = {}, {}, {}
    for name in LIABILITIES:
        matches = re.findall(
            rf"(?m)^\s*{name}\s+(?:[567]\.4\.\d+(?:\.\d+)*\s+)?({slot})\s+({comparison_slot})\s*$", body
        )
        if len(matches) != 1:
            raise ValueError("R5_CMF_LIABILITY_ROW:" + name)
        cell, comparison = matches[0]
        parts[name] = None if cell in "-－—" else cell.replace(",", "")
        cells[name], comparisons[name] = cell, comparison
    proof = identify_nonnegative_parts(total, parts)
    if proof["state"] != "POINT_IDENTIFIED":
        raise ValueError("R5_CMF_LIABILITY_AMBIGUOUS")
    return {
        **proof,
        "original_cells": cells,
        "comparison_cells_not_current_observations": comparisons,
        "layout_adapter": "SIGNED_COMPARISON_COLUMN_CURRENT_VALUES_REMAIN_NONNEGATIVE",
        "source_hash": primary_hash,
    }


def flows_with_split_explanation(source: Originals, primary_hash: str) -> dict:
    """仅识别拆分行完整原注释，继续以原A/C列和逐份额守恒核准申赎。"""
    pages = verified_pages(source, primary_hash)
    try:
        rows, proof = _flows(pages)
        return {"rows": rows, "proof": proof, "layout_adapter": None}
    except ValueError as error:
        if str(error) != "R5_DIRECT_INDEX_SHARE_FLOW_ROW:split":
            raise
    compact = normalized_text("\n".join(pages))
    original = normalized_text('本报告期基金拆分变动份额（份额减少以"-"填列）')
    replacement = "本报告期基金拆分变动份额"
    if compact.count(original) != 1:
        raise ValueError("R5_CMF_SPLIT_EXPLANATION_NOT_UNIQUE")
    # 不删负号、横杠单元格或其他括号内容；完整原文和改动轨迹均保留。
    rows, proof = _flows([compact.replace(original, replacement)])
    return {
        "rows": rows,
        "proof": proof,
        "layout_adapter": "EXACT_SPLIT_ROW_EXPLANATION_ONLY",
        "layout_change": {"original": original, "replacement": replacement},
        "source_hash": primary_hash,
    }


def section(text: str, start: str, stop: str) -> str:
    """取正文最后一个标题至后继节，避免目录及跨节片段冒充完整原文。"""
    begin = text.rfind(start)
    end = text.find(stop, begin + len(start))
    if begin < 0 or end <= begin:
        raise ValueError("R5_CMF_HOLDER_SECTION_MISSING:" + start)
    return text[begin:end]


PROTOCOL = "R5_CMF_REPORTED_FOUNDATION_STRUCTURE_CONDITIONS_20261029_1"
COMPATIBLE_TYPES = ["DISPERSED", "CONCENTRATED_OPEN_END"]
FOUNDATION_GATES = [
    "legal_open_end_cash_product",
    "complete_A_C_category_table",
    "same_product_share_flow",
    "ordinary_and_large_holder_mechanisms",
    "current_manager_and_trustee_execution",
    "pressure_effect_scope",
]


def foundation(source: Originals, inputs: dict) -> tuple[dict, dict]:
    """从真实同产品原文核准必要基础结构；不要求确定集中/分散标签。"""
    primary, contract_hash = inputs["target_refs"]["half"], inputs["target_refs"]["prospectus"]
    if (
        inputs["fund_code"] != "016008"
        or primary != PRIMARY_SHA256
        or inputs["holding_date"] != "2026-06-30"
        or inputs["as_of_date"] != "2026-09-30"
    ):
        raise ValueError("R5_CMF_HOLDER_PRODUCT_OR_PERIOD")
    review = inputs.get("holder_condition_review")
    if not review or review.get("actual_effect_review_complete") is not True:
        raise ValueError("R5_CMF_HOLDER_EFFECT_REVIEW_UNKNOWN")
    if review.get("confirmed_pressure_effects") is None:
        raise ValueError("R5_HOLDER_EFFECT_SCOPE_UNKNOWN")
    if review.get("actual_structure_type") is not None or review.get("compatible_structure_types") != COMPATIBLE_TYPES:
        raise ValueError("R5_CMF_ACTUAL_STRUCTURE_MUST_REMAIN_UNKNOWN")
    required_claims = [
        "product",
        "structure",
        "ordinary_mechanism",
        "large_holder_mechanism",
        "manager_execution",
        "execution",
    ]
    if any(name not in review for name in required_claims):
        raise ValueError("R5_CMF_HOLDER_BASIC_CLAIM_MISSING")
    for name in required_claims:
        source.claim(review[name])
        expected_hash = contract_hash if name in {"ordinary_mechanism", "large_holder_mechanism"} else primary
        if review[name]["source_hash"] != expected_hash:
            raise ValueError("R5_CMF_HOLDER_CROSS_PRODUCT_CLAIM")
    primary_text, contract = source.plain(primary), source.plain(contract_hash)
    expected = {
        "structure": section(primary_text, "8.1期末基金份额持有人户数及持有人结构", "8.2"),
        "manager_execution": section(primary_text, "4.2管理人对报告期内本基金运作遵规守信情况的说明", "4.3"),
        "execution": section(primary_text, "5.2托管人对报告期内本基金投资运作", "5.3"),
        "ordinary_mechanism": section(contract, "7.5申购与赎回的程序", "7.6"),
        "large_holder_mechanism": section(contract, "7.10巨额赎回的情形及处理方式", "7.11"),
    }
    if any(review[name]["quote"] != quote for name, quote in expected.items()):
        raise ValueError("R5_CMF_HOLDER_COMPLETE_SECTION_REQUIRED")
    if not all(value in review["product"]["quote"] for value in ["016007", "016008", "契约型开放式"]):
        raise ValueError("R5_CMF_HOLDER_LEGAL_OPEN_END_MODE_REQUIRED")
    if not all(value in expected["ordinary_mechanism"] for value in ["T+7", "赎回款项", "基金登记机构", "巨额赎回"]):
        raise ValueError("R5_CMF_HOLDER_ORDINARY_MECHANISM_REQUIRED")
    if not all(
        value in expected["large_holder_mechanism"]
        for value in ["单个开放日", "单个基金份额持有人", "上一日基金总份额10%", "延期办理", "无优先权"]
    ):
        raise ValueError("R5_CMF_HOLDER_CONCENTRATED_MECHANISM_REQUIRED")
    if "本报告期内,基金运作整体合法合规,无损害基金持有人利益的行为" not in expected["manager_execution"]:
        raise ValueError("R5_CMF_HOLDER_CURRENT_MANAGER_EXECUTION_REQUIRED")
    holder_execution_evidence(source, {"execution": review["execution"]}, primary)
    if "未发现基金管理人有损害基金份额持有人利益的行为" not in expected["execution"]:
        raise ValueError("R5_CMF_HOLDER_CURRENT_TRUSTEE_EXECUTION_REQUIRED")
    # 原表逐行核对列序和数值，不把相同数字在别处出现当作同一单元格。
    raw = "\n".join(source.pages(primary))
    amount = r"[\d,]+\.\d{2}"
    slot = rf"(?:{amount}|[-－—])"
    pattern = (
        rf"招商中证消费电子主题\s*ETF\s*联接\s*([AC])\s+([\d,]+)\s+({amount})\s+"
        rf"({slot})\s+({slot}%?)\s+({amount})\s+({amount})%"
    )
    matches = re.findall(pattern, raw)
    if len(matches) != 2 or [r[0] for r in matches] != ["A", "C"]:
        raise ValueError("R5_CMF_HOLDER_COMPLETE_CATEGORY_ROWS_REQUIRED")
    if not all(s in expected["structure"] for s in ["持有人户数", "户均持有", "机构投资者", "个人投资者", "占总份额"]):
        raise ValueError("R5_CMF_HOLDER_CATEGORY_HEADERS_REQUIRED")
    flows = flows_with_split_explanation(source, primary)
    rows = []
    for cells, flow in zip(matches, flows["rows"], strict=True):
        share, households, average, institution, institution_percent, individual, individual_percent = cells
        values = {
            "institution": None if institution in "-－—" else institution.replace(",", ""),
            "individual": individual.replace(",", ""),
        }
        proof = identify_nonnegative_parts(flow["closing"], values)
        if proof["state"] != "POINT_IDENTIFIED" or int(households.replace(",", "")) <= 0:
            raise ValueError("R5_CMF_HOLDER_CATEGORY_TOTAL_NOT_CLOSED")
        rows.append(
            {
                "share": share,
                "reported_households": households,
                "reported_average_units": average,
                "institution_original_cell": institution,
                "institution_percent_original_cell": institution_percent,
                "individual_units": individual,
                "individual_percent": individual_percent,
                "total_units": flow["closing"],
                "category_reconciliation": proof,
            }
        )
    if sum(Decimal(r["total_units"]) for r in rows) != Decimal("57719331.06"):
        raise ValueError("R5_CMF_HOLDER_LEGAL_TOTAL_NOT_CLOSED")
    # 结构闭合仅指上述已证法律方式、完整分类及份额流，不伪造集中标签。
    node = {
        "kind": "holder_execution_r5",
        "structure_type": None,
        "structure_condition_protocol": PROTOCOL,
        "compatible_structure_types": deepcopy(COMPATIBLE_TYPES),
        "structure_closed": True,
        "structure_basis": "OPEN_END_CASH_LEGAL_PRODUCT_AND_COMPLETE_REPORTED_CATEGORY_TABLE",
        "foundation_gate_checks": {name: True for name in FOUNDATION_GATES},
        "share_flow_closed": True,
        "redemption_mechanism_verified": True,
        "actual_execution_verified": True,
        "actual_redemption_shares": str(sum(Decimal(r["redeemed"]) for r in flows["rows"])),
        "confirmed_pressure_effects": deepcopy(review["confirmed_pressure_effects"]),
        "source_refs": sorted({primary, contract_hash}),
        "concentration_disclosure": "实际集中标签及最大个人持有比例未确认；不把空表横杠解释成零或无。",
    }
    proof = {
        "category_rows": rows,
        "share_flows": flows,
        "claims": deepcopy(review),
        "no_individual_max_ratio_fabricated": True,
        "no_xbrl_dash_converted_to_zero": True,
    }
    return node, proof


def invariant_holders_score(node: dict, *, evaluator):
    """全部相容结构均须通过原门槛且点值相同；不以同分豁免基础证据。"""
    if node.get("structure_type") is not None or node.get("compatible_structure_types") != COMPATIBLE_TYPES:
        raise ValueError("R5_CMF_HOLDER_TYPES_NOT_EXHAUSTIVE_OR_ACTUAL_LABEL_FABRICATED")
    if node.get("structure_basis") != "OPEN_END_CASH_LEGAL_PRODUCT_AND_COMPLETE_REPORTED_CATEGORY_TABLE" or node.get(
        "foundation_gate_checks"
    ) != {name: True for name in FOUNDATION_GATES}:
        raise ValueError("R5_CMF_HOLDER_FOUNDATION_NOT_PROVEN")
    outcomes = []
    for kind in COMPATIBLE_TYPES:
        # 分支只改变条件标签；实际节点保持未知，不递归触发条件适配。
        case = {**node, "structure_type": kind}
        case.pop("structure_condition_protocol", None)
        point, detail = evaluator(case)
        outcomes.append({"condition": kind, "point": str(point), "detail": detail})
    if len({row["point"] for row in outcomes}) != 1:
        raise ValueError("R5_CMF_HOLDER_STRUCTURE_CONDITIONS_NOT_POINT_IDENTIFIED")
    return Decimal(outcomes[0]["point"]), {
        "actual_structure_type": None,
        "compatible_structure_types": deepcopy(COMPATIBLE_TYPES),
        "structure_condition_outcomes": outcomes,
        "basic_evidence_not_waived": True,
        "not_future_pressure_test": True,
        "concentration_still_disclosed": True,
    }


def compile_observations(inputs: dict, *, root) -> dict:
    """重放原七维编译，并从同日真实原件独立编译全部四个流动性子项。"""
    if inputs.get("holder_structure_condition_protocol") != PROTOCOL:
        return standard_compile(inputs, root=root)
    source = Originals(inputs, root)
    holder, holder_proof = foundation(source, inputs)
    seven_inputs = deepcopy(inputs)
    seven_inputs.pop("holder_structure_condition_protocol", None)
    seven_inputs["component_scope"] = "NON_LIQUIDITY_SEVEN_DIMENSIONS"
    compiled = standard_compile(seven_inputs, root=root)
    observations = deepcopy(compiled["observations"])
    all_refs = sorted(inputs["sources"])
    values = compiled["audit"]["performance"]
    peers = inputs["cohort"]["peers"]
    observations["market_capacity"] = {
        "kind": "rank",
        "target": values["016008"]["market_capacity"],
        "direction": "lower",
        "expected_products": ["CN_PUBLIC_FUND:" + c for c in peers],
        "peers": [{"product_id": "CN_PUBLIC_FUND:" + c, "value": values[c]["market_capacity"]} for c in peers],
        "source_refs": all_refs,
    }
    primary = inputs["target_refs"]["half"]
    own_pages = source.pages(primary)
    own = portfolio(own_pages, holding_date=inputs["holding_date"], include_restrictions=True)
    etf = inputs["etf_refs"]["016008"]
    underlying = portfolio(
        source.pages(etf["source_hash"]), holding_date=inputs["holding_date"], include_restrictions=True
    )
    with localcontext() as context:
        context.prec = 70
        restricted = Decimal(own["restricted_assets"]["restricted_amount"])
        restricted += Decimal(compiled["audit"]["feeder_budgets"]["016008"]["etf_scale"]) * Decimal(
            underlying["restricted_assets"]["restricted_amount"]
        )
    observations["restricted"] = {
        "kind": "restricted",
        "restricted_amount": str(restricted),
        "net_assets": own["balance"]["net_assets"],
        "scope_closed": True,
        "source_refs": sorted({primary, etf["source_hash"]}),
    }
    view, _ = periodic_financial_view(own_pages)
    accounting = accounting_observations(view, date.fromisoformat(inputs["holding_date"]))["observations"]
    if Decimal(accounting["money_funds"]["value"]) != Decimal(accounting["demand_deposits_total"]["value"]):
        raise ValueError("R5_DIRECT_INDEX_NON_DEMAND_CASH_REQUIRES_PURPOSE_ADAPTER")
    liabilities = liability_signed_comparison(source, primary, own["balance"]["liabilities"])
    scope = own["restriction_scope_excerpt"]
    source.claim({"source_hash": primary, "quote": scope})
    observations["known_payment"] = {
        "kind": "purpose_covered_payment",
        "as_of": inputs["holding_date"],
        "currency": "CNY",
        "bank_principal": accounting["demand_deposits_principal"]["value"],
        "broker_settlement_principal": "0",
        "settlement_liabilities": liabilities["point_values"]["应付清算款"],
        "all_book_liabilities": own["balance"]["liabilities"],
        "scope_closed": True,
        "restriction_amount_pending": False,
        "disclosed_restrictions": [],
        "source_refs": [primary],
        "scope_note": "货币资金全部为活期存款，取原件本金；备付金、保证金、应收款及底层ETF现金不计通用付款本金。",
    }
    observations["holders"] = holder
    compiled["observations"] = observations
    compiled["audit"].update(
        holder_conditions=holder_proof,
        liability_proof=liabilities,
        share_flow_rows=holder_proof["share_flows"]["rows"],
        share_flow_proof=holder_proof["share_flows"]["proof"],
        lookthrough_restricted_cny=str(restricted),
        cash_restriction_scope=scope,
    )
    return compiled
