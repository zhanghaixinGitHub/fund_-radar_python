"""核对财报内嵌预告栏目，不把栏目存在或“不适用”误当成已披露利润。

只生成有页锚点的资料候选；不抽猜金额、不训练、不声明历史完整。
"""

import calendar
import re
from datetime import date

from app.services.fund_information_history_v1 import digest, normalize


def embedded_forecasts(row, pages):
    """查找明确月份区间和勾选状态，保持预告期间独立于承载报告的期间。

    仅识别同页内明示的勾选；跨页、缺勾选、多重勾选均待核验。明确不适用
    表示本栏目未给出预告，既不是利润为零，也不能证明其他公告没有预告。
    返回的金额始终为 None，须另外核对原件字段和可用时间。
    """
    pattern = re.compile(r"对(20\d{2})年(\d{1,2})[-－—~～至](\d{1,2})月经营业绩的预计")
    found = []
    for number, raw in enumerate(pages, 1):
        text = normalize(raw)
        for match in pattern.finditer(text):
            year, first, last = map(int, match.groups())
            if not 1 <= first <= last <= 12:
                raise ValueError("EMBEDDED_FORECAST_PERIOD_INVALID")
            start = date(year, first, 1)
            end = date(year, last, calendar.monthrange(year, last)[1])
            tail = text[match.end() :]
            following = re.search(r"[一二三四五六七八九十]+、", tail)
            stop = match.end() + following.start() if following else len(text)
            section = text[match.start() : stop]
            # 只接受成对勾选顺序；其他栏目里的“√适用”不能替本栏目表态。
            positive = section.count("√适用□不适用")
            negative = section.count("□适用√不适用")
            if positive == 1 and negative == 0:
                status = "APPLICABLE_REQUIRES_FACT_REVIEW"
            elif positive == 0 and negative == 1:
                status = "EXPLICIT_NOT_APPLICABLE_IN_THIS_SECTION"
            else:
                status = "UNRESOLVED_CHECKBOX_OR_PAGE_CONTINUATION"
            source_key = {
                "issuer": row["secCode"],
                "published_date": row["published_date"],
                "period_start": str(start),
                "period_end": str(end),
                "section_text": section,
            }
            found.append(
                {
                    "document_id": row["announcementId"],
                    "issuer": row["secCode"],
                    "published_date": row["published_date"],
                    "carrier_title": row["title_plain"],
                    "period_start": str(start),
                    "period_end": str(end),
                    "status": status,
                    "anchor": {"page": number, "offset": match.start(), "text": section},
                    "content_group_key": digest(source_key),
                    "money": None,
                    "absence_of_other_forecasts_proven": False,
                    "training_ready": False,
                }
            )
    return found
