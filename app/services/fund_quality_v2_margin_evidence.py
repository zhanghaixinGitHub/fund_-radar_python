"""按报告明示的保证金扣减工具集合核对期末缺位原句，不以余额横杠代替。

仅用于现金篮子数学边界。未交易、未来不投资、资产负债表衍生科目为零，
均不能证明无需保证金；“全部”范围也不能用三个工具的缺位陈述穷尽。
调用方仍须核验原件身份、实际公开日、报告期及原字节摘要。
"""

import re

from app.services.fund_quality_v2_cash_clauses import INSTRUMENT_LIST, INSTRUMENTS, compact

VERSION = "QUALITY_NAMED_MARGIN_EVIDENCE_20261017_2"
PERIOD = r"本基金(?:在)?(?:本报告期末|本报告期(?:内)?|报告期末)"
ABSENCE = rf"{PERIOD}(?:未投资|未持有)(?P<names>{INSTRUMENT_LIST})(?=[。.;]|$)"
PRESENCE = rf"{PERIOD}(?:投资|持有)(?P<names>{INSTRUMENT_LIST})(?=[。.;]|$)"


def margin_absence_evidence(text: str, *, requirement_scope: str,
                            required_instruments: list[str] | None) -> dict:
    """逐工具保存同源原句及未覆盖集合，只在明示集合被完整覆盖时识别零需求。

    参数中的工具集合来自同报告现金义务原句，不能由调用者默认三个工具。
    返回的零需求结论由实际否定原句推导；接口不接收布尔“已核验”标志。
    原文“全部保证金”未被列举集合穷尽，始终保持金额未知。
    """
    if requirement_scope not in {"EXPLICIT_NAMED_INSTRUMENTS", "ALL_REQUIRED_TRADING_MARGIN",
                                 "UNQUALIFIED_REQUIRED_TRADING_MARGIN"}:
        raise ValueError("QUALITY_CASH_MARGIN_SCOPE_NOT_EXPLICIT")
    if requirement_scope in {"ALL_REQUIRED_TRADING_MARGIN", "UNQUALIFIED_REQUIRED_TRADING_MARGIN"}:
        if required_instruments is not None:
            raise ValueError("QUALITY_CASH_ALL_MARGIN_CANNOT_HAVE_NARROWED_INSTRUMENT_SET")
        required = None
    else:
        if (not isinstance(required_instruments, list) or not required_instruments
                or len(required_instruments) != len(set(required_instruments))
                or not set(required_instruments) <= set(INSTRUMENTS.values())):
            raise ValueError("QUALITY_CASH_REQUIRED_MARGIN_INSTRUMENT_SET_INVALID")
        required = sorted(required_instruments)
    body = compact(text)
    absent, present = {}, set()
    for match in re.finditer(ABSENCE, body):
        for name in re.findall(r"股指期货|国债期货|股票期权", match["names"]):
            absent.setdefault(INSTRUMENTS[name], []).append(match[0])
    for match in re.finditer(PRESENCE, body):
        present.update(INSTRUMENTS[name] for name in re.findall(r"股指期货|国债期货|股票期权", match["names"]))
    if present & absent.keys():
        raise ValueError("QUALITY_CASH_NAMED_MARGIN_ABSENCE_SOURCE_CONFLICT")
    missing = sorted(set(required) - absent.keys()) if required else None
    return {"version": VERSION, "requirement_scope": requirement_scope,
            "required_instruments": required, "absence_originals_by_instrument": absent,
            "missing_absence_instruments": missing,
            "zero_required_margin_identified": missing == [] if required else False,
            "unresolved_reason": "ALL_REQUIRED_MARGIN_NOT_EXHAUSTED_BY_NAMED_ABSENCE_STATEMENTS"
            if required is None else "REQUIRED_INSTRUMENT_ABSENCE_NOT_EXPLICIT" if missing else None,
            "scope": "REPORT_DATE_MARGIN_BOUND_NOT_ACCOUNT_BALANCE_OR_DAILY_COMPLIANCE"}
