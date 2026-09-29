"""按明确公司和年度选择独立年报摘要，不拿修订版或邻近年份替代。"""

import re

from app.services.fund_information_history_v1 import normalize


def select_annual_summary(rows, stock, year):
    """限定已核验分页结果中的主体、年度和摘要用途；没有唯一原件就停止。

    日期窗口由调用方冻结并由 catalog_rows 核验。这里不按金额、市场表现
    或最新更新时间选文档；重名原件不强行合并，正文和修订摘要均不能替代。
    """
    if not re.fullmatch(r"\d{6}", stock) or not isinstance(year, int) or isinstance(year, bool):
        raise ValueError("INVALID_ANNUAL_REFERENCE_SCOPE")
    matches = [
        row
        for row in rows
        if row["secCode"] == stock
        and re.search(rf"{year}(?:年)?年度报告摘要$", normalize(row["title_plain"]))
        and not any(word in row["title_plain"] for word in ("更正", "更新", "修订", "补充", "取消"))
    ]
    if len(matches) != 1:
        raise ValueError("ANNUAL_SUMMARY_NOT_UNIQUE")
    return matches[0]
