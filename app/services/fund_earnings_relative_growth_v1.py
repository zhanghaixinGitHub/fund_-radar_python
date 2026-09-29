"""核验年度预告中的同比增加额；独立保存，禁止冒充当期利润绝对值。"""

import re
from datetime import date, timedelta
from decimal import Decimal

from app.services.fund_earnings_identity_v1 import occurrence
from app.services.fund_information_history_v1 import explicit_money, normalize

METRICS = {
    "NET_PROFIT_ATTRIBUTABLE_TO_PARENT": "归属于上市公司股东的净利润",
    "ADJUSTED_NET_PROFIT_ATTRIBUTABLE_TO_PARENT": "归属于上市公司股东的扣除非经常性损益的净利润",
}


def review_relative_growth(spec, pages):
    """重放已登记的年度同比增加额和百分比，保留“左右”的近似含义。

    仅支持原文明确写出的全年、上年同期法定披露数据、增加额及增加率。
    不支持减少、跨期、精确值或通过旧值推算绝对利润；遇到其他措辞拒绝。
    issuer 为公司代码，increase_amount 使用原单位，increase_percent 为百分数。
    原件身份和修订日期由调用者先核验；本函数不负责从标题推断这些信息。
    """
    required = (
        "document_id",
        "issuer",
        "metric",
        "period_start",
        "period_end",
        "published_date",
        "page",
        "quote",
        "unit",
        "increase_amount",
        "increase_percent",
        "period_page",
        "period_quote",
    )
    if any(spec.get(key) is None or spec.get(key) == "" for key in required):
        raise ValueError("INCOMPLETE_RELATIVE_GROWTH")
    if spec["metric"] not in METRICS or not re.fullmatch(r"\d{6}", spec["issuer"]):
        raise ValueError("INVALID_RELATIVE_METRIC_OR_ISSUER")
    start, end, published = [date.fromisoformat(spec[k]) for k in ("period_start", "period_end", "published_date")]
    if str(start) != f"{start.year}-01-01" or str(end) != f"{start.year}-12-31" or published < start:
        raise ValueError("RELATIVE_GROWTH_ANNUAL_PERIOD_REQUIRED")
    for page_key, quote_key in (("page", "quote"), ("period_page", "period_quote")):
        page_no = spec[page_key]
        if (
            isinstance(page_no, bool)
            or not isinstance(page_no, int)
            or not 1 <= page_no <= len(pages)
            or normalize(pages[page_no - 1]).count(normalize(spec[quote_key])) != 1
        ):
            raise ValueError("RELATIVE_ANCHOR_NOT_UNIQUE")
    anchor = occurrence(pages, spec["page"], spec["quote"])
    period = occurrence(pages, spec["period_page"], spec["period_quote"])
    expected_period = f"{start.year}年1月1日至{start.year}年12月31日"
    if normalize(spec["period_quote"]).rstrip("。") != expected_period:
        raise ValueError("RELATIVE_PERIOD_ANCHOR_MISMATCH")
    pattern = (
        r"(?:[12]\.)?(?:经本公司财务部门初步测算，)?预计"
        + str(start.year)
        + "年度实现"
        + METRICS[spec["metric"]]
        + r"与上年同期（法定披露数据）相比，将增加(\d+(?:\.\d+)?)(亿元|万元|元)左右，"
        + r"同比增加(\d+(?:\.\d+)?)%左右。"
    )
    match = re.fullmatch(pattern, normalize(spec["quote"]))
    if not match:
        raise ValueError("UNSUPPORTED_RELATIVE_GROWTH_STATEMENT")
    amount, unit, percent = match.groups()
    if (amount, unit, percent) != (str(spec["increase_amount"]), spec["unit"], str(spec["increase_percent"])):
        raise ValueError("RELATIVE_GROWTH_VALUE_MISMATCH")
    if Decimal(amount) <= 0 or Decimal(percent) <= 0:
        raise ValueError("RELATIVE_INCREASE_MUST_BE_POSITIVE")
    # 刻意不返回普通利润事实的 money、values、kind 字段，防止被版本链当成利润水平。
    return {
        "document_id": spec["document_id"],
        "issuer": spec["issuer"],
        "metric": spec["metric"],
        "period_start": str(start),
        "period_end": str(end),
        "published_date": str(published),
        "available_at": str(published + timedelta(days=1)) + "T08:00:00+08:00",
        "measure": "YOY_ABSOLUTE_INCREASE",
        "increase_original": amount,
        "unit": unit,
        "increase_cny": explicit_money(amount, unit)["cny"],
        "increase_percent": percent,
        "approximate": True,
        "approximation_marker": "左右",
        "currency": "CNY",
        "reference_basis": "PREVIOUS_YEAR_STATUTORY_DISCLOSURE",
        "anchor": anchor,
        "period_anchor": period,
        "absolute_profit_not_established": True,
        "independent_prior_disclosure_not_established": True,
        "eligible_for_profit_level_chain": False,
        "training_ready": False,
    }
