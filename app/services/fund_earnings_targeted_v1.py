"""按已披露引用补三季报原件，限定原始版本和日期，不按预测结果选资料。"""

import re

from app.services.fund_information_history_v1 import normalize


def select_referenced_quarter(rows, stock, year, referenced_date):
    """选择全文/正文各至多一份；更正、重名或公开日期冲突必须停下核对。

    调用方先核完整窗口的分页。引用日期仅是待核条件，不能代替目录公开日；
    全文和正文保留为两个文档，经济事实是否相同留到后续字段核验。
    """
    if not re.fullmatch(r"\d{6}", stock) or isinstance(year, bool) or not isinstance(year, int):
        raise ValueError("INVALID_REFERENCED_SCOPE")
    selected = {}
    for row in rows:
        if row["secCode"] != stock:
            raise ValueError("WRONG_REFERENCED_ISSUER")
        title = normalize(row["title_plain"])
        if str(year) not in title or "第三季度报告" not in title:
            continue
        if any(word in title for word in ("更正", "更新", "修订", "补充", "取消")):
            raise ValueError("REFERENCED_QUARTER_REVISION_REQUIRES_REVIEW")
        match = re.fullmatch(rf"{year}年第三季度报告(全文|正文)", title)
        if not match:
            continue
        if row["published_date"] != referenced_date:
            raise ValueError("REFERENCED_PUBLIC_DATE_CONFLICT")
        kind = match[1]
        if kind in selected:
            raise ValueError("REFERENCED_QUARTER_NOT_UNIQUE")
        selected[kind] = row
    if not selected:
        raise ValueError("REFERENCED_QUARTER_NOT_FOUND")
    return [selected[k] for k in sorted(selected)]
