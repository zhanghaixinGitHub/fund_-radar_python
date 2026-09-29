"""核验年度业绩快报，保留初步、未审计及会计准则未明确的事实边界。"""

from datetime import date
from decimal import Decimal

from app.services.fund_earnings_identity_v1 import occurrence
from app.services.fund_earnings_scoped_table_v1 import review_scoped_money_row
from app.services.fund_information_history_v1 import normalize

LABELS = {
    "NET_PROFIT_ATTRIBUTABLE_TO_PARENT": "归属于上市公司股东的净利润",
    "ADJUSTED_NET_PROFIT_ATTRIBUTABLE_TO_PARENT": "归属于上市公司股东的扣除非经常性损益的净利润",
}


def review_preliminary(spec, pages):
    """按原件表格核验当年利润，不把公司内审等同于会计师事务所审计。

    仅支持明确的年度快报、合并表和本报告期列；上年同期保留在原行中，
    不生成更早独立事实。准则未在选定原文明确时保留未知，不借用别份年报。
    金额分辨率根据原单元格小数位和单位计算，用于保留亿元等粗粒度披露。
    """
    start, end = date.fromisoformat(spec["period_start"]), date.fromisoformat(spec["period_end"])
    if (start.month, start.day, end.year, end.month, end.day) != (1, 1, start.year, 12, 31):
        raise ValueError("PRELIMINARY_ANNUAL_PERIOD_REQUIRED")
    if date.fromisoformat(spec["published_date"]) <= end:
        raise ValueError("PRELIMINARY_BEFORE_PERIOD_END")
    title = normalize(spec["source_title"])
    if title not in {f"{start.year}年度业绩快报", f"{start.year}年度业绩快报公告"}:
        raise ValueError("PRELIMINARY_TITLE_OR_YEAR_MISMATCH")
    if spec["kind"] != "PRELIMINARY_RESULT":
        raise ValueError("PRELIMINARY_MUST_NOT_BECOME_REPORTED_RESULT")
    if spec["basis"] != "CONSOLIDATED_STANDARD_UNSPECIFIED":
        raise ValueError("PRELIMINARY_STANDARD_MUST_REMAIN_UNSPECIFIED")
    label = LABELS.get(spec["metric"])
    if label is None or normalize(spec["row_label"]) != label:
        raise ValueError("PRELIMINARY_METRIC_ROW_MISMATCH")
    if spec["selected_column"] != 0 or spec["columns"][0]["label"] != "本报告期":
        raise ValueError("PRELIMINARY_CURRENT_COLUMN_REQUIRED")
    audit = normalize(spec["audit_quote"])
    if "未经会计师事务所审计" not in audit:
        raise ValueError("PRELIMINARY_UNAUDITED_ANCHOR_REQUIRED")
    if "合并报表数据" not in normalize(spec["basis_quote"]):
        raise ValueError("PRELIMINARY_CONSOLIDATION_ANCHOR_REQUIRED")
    anchors = {
        "title": occurrence(pages, spec["title_page"], spec["source_title"]),
        "audit": occurrence(pages, spec["audit_page"], spec["audit_quote"]),
        "basis": occurrence(pages, spec["basis_page"], spec["basis_quote"]),
    }
    fact = review_scoped_money_row(spec, pages)
    value = Decimal(spec["cells"][0].replace(",", ""))
    quantum = Decimal(1).scaleb(value.as_tuple().exponent)
    multiplier = {"元": Decimal(1), "万元": Decimal(10000), "亿元": Decimal(100000000)}[spec["unit"]]
    return {
        **fact,
        "stage_proof": anchors,
        "audit_status": "PRELIMINARY_NOT_AUDITED_BY_ACCOUNTING_FIRM",
        "accounting_standard_verified": False,
        "reported_amount_resolution_cny": str(quantum * multiplier),
        "comparative_is_independent_earlier_disclosure": False,
        "training_ready": False,
    }
