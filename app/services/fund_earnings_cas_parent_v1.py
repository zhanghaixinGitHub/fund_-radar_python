"""核验明示中国准则的归母净利润；仅处理预告句和准则差异表的指定原行。

这里的范围是合并归母指标，与母公司单体净利润区分；不由相同的未知口径
推断一致，也不宣称逐家子公司的构成没有变动。未明示准则和归母指标的
资料不能使用此入口。同比增加额、净资产及扣非净利润均不在本入口范围。
"""

import re
from decimal import Decimal

from app.services.fund_earnings_evidence_v1 import review_claim
from app.services.fund_earnings_identity_v1 import occurrence
from app.services.fund_information_history_v1 import normalize

METRIC = "NET_PROFIT_ATTRIBUTABLE_TO_PARENT"
BASIS = "CAS_CONSOLIDATED_PROFIT_ATTRIBUTABLE_TO_PARENT"
LABEL = "归属于母公司股东的净利润"
NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"


def review_cas_parent(spec, pages):
    """核对同页连续证据并返回金额、单位、报告精度与会计/归属范围锚点。

    FORECAST 仅接收明示准则、全年、归母、人民币的完整预告句；实际数仅
    接收差异表内中国准则一行，固定选当前年度第一列。全部列值都需匹配，
    不能从国际准则行或净资产区块取数。证券身份及修订由上层原件重放核验。
    """
    year = spec["year"]
    if (
        spec["period_start"] != f"{year}-01-01"
        or spec["period_end"] != f"{year}-12-31"
        or spec["metric"] != METRIC
        or spec["basis"] != BASIS
        or spec["currency"] != "CNY"
    ):
        raise ValueError("CAS_PARENT_SCOPE_MISMATCH")
    quote = normalize(spec["quote"])
    if spec["kind"] == "FORECAST":
        if spec["unit"] != "亿元":
            raise ValueError("CAS_FORECAST_UNIT_MISMATCH")
        expression = (
            rf"经本公司财务部门初步测算，按照中国企业会计准则，预计{year}年全年度"
            rf"实现{LABEL}为人民币({NUMBER})亿元到({NUMBER})亿元"
        )
        match = re.fullmatch(expression, quote)
        if not match or list(match.groups()) != spec["values"]:
            raise ValueError("CAS_FORECAST_LITERAL_MISMATCH")
        period = occurrence(pages, spec["period_page"], f"{year}年1月1日到{year}年12月31日")
        stage = occurrence(pages, spec["page"], "本次业绩预告相关的财务数据未经会计师事务所审计。")
        proof = {"period": period, "audit_status": stage}
    elif spec["kind"] == "REPORTED_RESULT":
        if spec["unit"] != "百万元" or len(spec["table_cells"]) != 2:
            raise ValueError("CAS_TABLE_UNIT_OR_COLUMNS_MISMATCH")
        expression = (
            rf"人民币百万元{LABEL}{year}年度{year - 1}年度"
            rf"按中国企业会计准则\s+({NUMBER})\s+({NUMBER})"
        )
        # 原始行保留列间空白，防止将相邻金额误拼为一个数字；其他表头去空白。
        raw = spec["quote"]
        row = raw[raw.index("按中国企业会计准则") :]
        head = normalize(raw[: raw.index("按中国企业会计准则")])
        match = re.fullmatch(expression, head + row.strip())
        if not match or list(match.groups()) != spec["table_cells"] or spec["values"] != [match[1]]:
            raise ValueError("CAS_TABLE_LITERAL_OR_CURRENT_COLUMN_MISMATCH")
        proof = {"current_year": year, "comparative_year": year - 1, "all_cells": spec["table_cells"]}
    else:
        raise ValueError("CAS_PARENT_STAGE_NOT_SUPPORTED")
    fact = review_claim(spec, pages)
    # 两类来源都把中国准则与“归属于母公司股东”写在同一个指定事实范围内。
    # 证据确认的是该指标的合并归属层级，不能据此声称各期合并实体不变。
    anchor = fact["anchor"]
    return {
        **fact,
        "stage_proof": proof,
        "accounting_standard_verified": True,
        "accounting_basis_anchor": anchor,
        "consolidation_scope_verified": True,
        "consolidation_scope_anchor": anchor,
        "scope_interpretation": "CAS profit attributable to parent shareholders; not parent-only net profit",
        "unchanged_consolidated_entity_population_verified": False,
        "reported_resolution_in_source_unit": [
            str(Decimal(1).scaleb(Decimal(v.replace(",", "")).as_tuple().exponent)) for v in spec["values"]
        ],
        "comparative_is_independent_earlier_disclosure": False,
        "training_ready": False,
    }
