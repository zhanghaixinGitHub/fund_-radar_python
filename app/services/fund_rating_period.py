"""滚动月评日期契约：新期必须绑定新期原始观察，不能只改旧包日期。"""

from datetime import date, timedelta


def validate_period(inputs, source, *, legacy_as_of="2026-09-30", legacy_holding="2026-06-30"):
    as_of = date.fromisoformat(inputs["as_of_date"])
    holding = date.fromisoformat(inputs["holding_date"])
    if str(as_of) == legacy_as_of and str(holding) == legacy_holding:
        return
    # 披露期和评分日分开核对；不额外引入没有既有规则依据的固定陈旧天数。
    # 是否已有更新完整报告由本期原件审阅核实，不能用天数代替披露核查。
    if as_of > date.today() or holding > as_of:
        raise ValueError("RATING_MONTHLY_REPORT_PERIOD_INVALID")
    review = inputs.get("monthly_period_review", {})
    if (
        review.get("fund_code") != inputs["fund_code"]
        or review.get("as_of_date") != str(as_of)
        or review.get("holding_date") != str(holding)
        or len(review.get("reason", "").strip()) < 60
        or not review.get("evidence")
    ):
        raise ValueError("RATING_MONTHLY_CURRENT_PERIOD_REVIEW_REQUIRED")
    for proof in review["evidence"]:
        body = source.body(proof["source_ref"])
        quote = proof.get("quote", "")
        if len(quote.strip()) < 20 or quote not in body:
            raise ValueError("RATING_MONTHLY_PERIOD_ORIGINAL_QUOTE_MISMATCH")


def last_completed_month(day):
    return day.replace(day=1) - timedelta(days=1)
