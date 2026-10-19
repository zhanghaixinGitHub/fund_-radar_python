"""报告观察日与资料可得日分开：尚未到披露期的期末不能算缺件。

只处理当前考察日最近一期的披露等待。过去的季度点是候选方法所需
历史观察点，不把当前招募书倒用成所有历史时期的法律逾期判断。
"""

import re
from datetime import date

from app.services.fund_rating_public_market import compact


def disclosure_window(expected: list[str], *, as_of: date, prospectus: dict, published_dates: set[str]) -> dict:
    """保留历史观察要求；从有原件依据的最近期剔除尚未到期观察点。

    工作日必须在本产品招募书中明确指向沪深交易日，且季度报告明确
    为季后15工作日；缺任一原句则记待核，不能默认境内日历适用QDII估值。
    """
    text = compact(prospectus.get("text", ""))
    workday = re.search(r"工作日[:：]?指上海证券交易所[、和及,，]*深圳证券交易所的正常交易日", text)
    deadline = re.search(r"(?:每个|每一|各)?季度结束之日起(?:15|十五)个工作日内[^。]{0,90}公告", text)
    evidence = {"source_hashes": [prospectus["ref"]["sha256"]]} if prospectus.get("ref") else {}
    evidence.update(workday_clause=workday[0] if workday else None, quarter_clause=deadline[0] if deadline else None)
    latest = expected[-1] if expected else None
    statuses = []
    required = []
    for observed in expected:
        row = {"observed_on": observed}
        if observed in published_dates:
            row["state"] = "PUBLISHED_ORIGINAL_AVAILABLE"
            required.append(observed)
        elif observed == latest:
            if not workday or not deadline:
                row["state"] = "CURRENT_QUARTER_DISCLOSURE_DEADLINE_UNVERIFIED"
            else:
                from app.services.direction_1d_protocol import calendar

                sessions, calendar_hash = calendar()
                end = date.fromisoformat(observed)
                if not sessions or end < sessions[0] or as_of > sessions[-1]:
                    row["state"] = "DISCLOSURE_WORKDAY_CALENDAR_OUTSIDE_COVERAGE"
                else:
                    elapsed = sum(end < day <= as_of for day in sessions)
                    row.update(elapsed_workdays=elapsed, calendar_hash=calendar_hash)
                    row["state"] = "NOT_YET_DUE" if elapsed < 15 else "CURRENT_QUARTER_REQUIRED"
                    if elapsed >= 15:
                        required.append(observed)
        else:
            row["state"] = "HISTORICAL_METHOD_CHECKPOINT_REQUIRED_NOT_LEGAL_OVERDUE_ASSERTION"
            required.append(observed)
        statuses.append(row)
    return {
        "version": "QUALITY_REPORT_AVAILABILITY_20261017_1",
        "as_of": str(as_of),
        "required_report_dates": required,
        "checkpoints": statuses,
        "deadline_evidence": evidence,
        "scope": "CURRENT_DISCLOSURE_WAIT_ONLY_HISTORICAL_EFFECTIVE_CONTRACTS_STILL_REQUIRED",
    }
