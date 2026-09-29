"""核验明示中国准则的上半年归母利润，区分预告、审阅与审计。

独立于已经冻结的全年核验器；不借用国际准则行、净资产行或上年列，
也不把半年报与全年预告比较。公司身份、版本冲突由上层原件核验。
"""

import re
from decimal import Decimal

from app.services.fund_earnings_cas_parent_v1 import BASIS, LABEL, METRIC, NUMBER
from app.services.fund_earnings_evidence_v1 import review_claim
from app.services.fund_earnings_yoy_amount_v1 import unique
from app.services.fund_information_history_v1 import normalize


def review_cas_interim(spec, pages):
    """核对连续的准则/归母/单位/期间原文，只接收已登记的两种半年布局。

    归属口径指中国准则合并归母利润，不表示各期子公司名单完全相同。
    当期报告中的上年比较列仅用于校验列位置，不能倒填为更早披露。
    """
    year = spec["year"]
    if (
        spec["period_start"] != f"{year}-01-01"
        or spec["period_end"] != f"{year}-06-30"
        or spec["metric"] != METRIC
        or spec["basis"] != BASIS
        or spec["currency"] != "CNY"
    ):
        raise ValueError("INTERIM_CAS_PARENT_SCOPE_MISMATCH")
    quote = normalize(spec["quote"])
    if spec["kind"] == "FORECAST":
        prefix = "按照中国企业会计准则，" + re.escape(spec["legal_name"]) + "（以下简称“公司”或“本公司”）"
        expression = prefix + rf"{year}年中期{LABEL}预计为人民币({NUMBER})亿元到({NUMBER})亿元"
        match = re.fullmatch(expression, quote)
        if spec["unit"] != "亿元" or not match or list(match.groups()) != spec["values"]:
            raise ValueError("INTERIM_FORECAST_LITERAL_MISMATCH")
        proof = {
            "period": unique(pages, spec["period_page"], f"{year}年1月1日到{year}年6月30日"),
            "audit": unique(pages, spec["audit_page"], "本次预计的业绩未经审计。"),
        }
        audit_status = "FORECAST_UNAUDITED"
    elif spec["kind"] == "REPORTED_RESULT":
        if spec["unit"] != "百万元" or len(spec["table_cells"]) != 2:
            raise ValueError("INTERIM_CAS_TABLE_UNIT_OR_COLUMNS")
        prefix = (
            rf"人民币百万元{LABEL}截至{year}年6月30日止6个月期间"
            rf"截至{year - 1}年6月30日止6个月期间"
        )
        raw = spec["quote"]
        marker = "按中国企业会计准则"
        if marker not in raw:
            raise ValueError("INTERIM_CAS_TABLE_STANDARD_MISSING")
        start = raw.index(marker)
        expression = prefix + marker + rf"\s+({NUMBER})\s+({NUMBER})"
        match = re.fullmatch(expression, normalize(raw[:start]) + raw[start:].strip())
        if not match or list(match.groups()) != spec["table_cells"] or spec["values"] != [match[1]]:
            raise ValueError("INTERIM_CAS_TABLE_LITERAL_OR_CURRENT_COLUMN")
        proof = {
            "current_year": year,
            "comparative_year": year - 1,
            "all_cells": spec["table_cells"],
            "unaudited": unique(pages, spec["audit_page"], "本半年度报告中的财务报告未经审计。"),
            "review_scope": unique(
                pages,
                spec["review_page"],
                f"自{year}年1月1日至6月30日止期间的合并及公司利润表、股东权益变动表和现金流量表",
            ),
            "review_not_audit": unique(pages, spec["review_page"], "我们没有实施审计，因而不发表审计意见。"),
        }
        audit_status = "INTERIM_REVIEWED_NOT_AUDITED"
    else:
        raise ValueError("INTERIM_CAS_STAGE_UNSUPPORTED")
    fact = review_claim(spec, pages)
    return {
        **fact,
        "stage_proof": proof,
        "audit_status": audit_status,
        "accounting_standard_verified": True,
        "accounting_basis_anchor": fact["anchor"],
        "consolidation_scope_verified": True,
        "consolidation_scope_anchor": fact["anchor"],
        "scope_interpretation": "CAS consolidated profit attributable to parent, not parent-only profit",
        "unchanged_consolidated_entity_population_verified": False,
        "comparative_is_independent_earlier_disclosure": False,
        "reported_resolution_in_source_unit": [
            str(Decimal(1).scaleb(Decimal(v.replace(",", "")).as_tuple().exponent)) for v in spec["values"]
        ],
        "training_ready": False,
    }
