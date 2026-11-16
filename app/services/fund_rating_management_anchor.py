"""季度持仓使用最近完整管理履职报告时，核同产品、同现任经理和任期。"""

import re
from datetime import date

from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_market_review import normalized_text


def appointment_verified(pages, person, begin):
    """兼容人员表把任职年月日拆到相邻两行的版式；仍核原件实际日期。

    只接受姓名开头的人员行、任职年份后紧接离任/从业栏，以及下一正文行
    的月日。不能在任意履历段落中拼日期，也不能跨人员或跨页寻找月日。
    """
    pattern = rf"{begin.year}(?:年|-0?){begin.month}(?:月|-0?){begin.day}日?"
    for page in pages:
        if re.search(re.escape(person) + r".{0,220}" + pattern, normalized_text(page)):
            return True
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        for line, following in zip(lines, lines[1:], strict=False):
            if not re.match(re.escape(person) + r"\s", line):
                continue
            if not re.search(rf"\b{begin.year}-\s+(?:-|\d+年)(?:\s|$)", line):
                continue
            month_day = rf"(?<!\d)0?{begin.month}-0?{begin.day}(?!\d)"
            if re.search(month_day + r"\s+-", following) or (
                re.search(rf"\b{begin.year}-\s+-\s", line)
                and re.search(month_day + r"(?=\s{2,}|$)", following)
            ):
                return True
    return False


def verify_management_anchor(inputs, root, report_hash):
    anchor = inputs.get("management_primary_report_hash", report_hash)
    if anchor == report_hash:
        return None
    source = Originals(inputs, root, lazy=True)
    primary = normalized_text("\n".join(source.pages(report_hash)))
    complete = normalized_text("\n".join(source.pages(anchor)))
    code = inputs["fund_code"]
    legal = normalized_text(inputs["legal_name"])
    for body in [primary, complete]:
        if legal not in body[:400] or code not in body[:16000]:
            raise ValueError("MANAGEMENT_ANCHOR_SAME_PRODUCT_REQUIRED")
    if not re.search("中期报告|年度报告", complete[:400]):
        raise ValueError("MANAGEMENT_ANCHOR_FULL_REPORT_REQUIRED")

    def report_period(body):
        hit = re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", body[:500])
        if hit is None:
            raise ValueError("MANAGEMENT_ANCHOR_REPORT_PERIOD_REQUIRED")
        return date(*map(int, hit.groups())).isoformat()

    anchor_period = report_period(complete)
    primary_period = report_period(primary)
    if anchor_period > primary_period or primary_period != inputs["holding_date"]:
        raise ValueError("MANAGEMENT_ANCHOR_PERIOD_BINDING")
    # “最近”限于同一已核来源集合；不把别的产品同公司报告混入候选。
    known_full = []
    for sha, ref in inputs["sources"].items():
        if ref["kind"] != "pdf":
            if ref["kind"] != "json":
                continue
            # 已有完整发行人公告也可能以JSON封存正文。净值/目录JSON不能
            # 充当报告，必须实际含完整notice_content后再核正文身份和日期。
            original = source.json(sha)
            data = original.get("data") if isinstance(original, dict) else None
            content = data.get("notice_content") if isinstance(data, dict) else None
            if not isinstance(content, str) or len(content) < 15000:
                continue
        pages = source.pages(sha)
        head = normalized_text("\n".join(pages))
        if legal in head[:400] and code in head[:16000] and re.search("中期报告|年度报告", head[:400]):
            period = report_period(head)
            if period <= primary_period:
                known_full.append(period)
    if not known_full or anchor_period != max(known_full):
        raise ValueError("MANAGEMENT_ANCHOR_LATEST_VERIFIED_FULL_REQUIRED")
    raw = inputs["management_inputs"]
    continuity = raw["continuity"]
    if continuity["period_end"] != inputs["as_of_date"]:
        raise ValueError("MANAGEMENT_ANCHOR_AS_OF_BINDING")
    active = [
        row
        for row in continuity["intervals"]
        if row["start"] <= primary_period and (row.get("end") is None or row["end"] >= primary_period)
    ]
    if not active or report_hash not in raw["experience"]["source_refs"]:
        raise ValueError("MANAGEMENT_ANCHOR_CURRENT_PERSON_ROLE_REQUIRED")
    facts = []
    for row in active:
        person = row["name"]
        begin = date.fromisoformat(row["start"])
        for source_hash in [report_hash, anchor]:
            if not appointment_verified(source.pages(source_hash), person, begin):
                raise ValueError("MANAGEMENT_ANCHOR_PERSON_APPOINTMENT_MISMATCH")
        if not any(
            claim["source_hash"] == report_hash
            and person in normalized_text(claim["quote"])
            and "基金经理" in normalized_text(claim["quote"])
            and normalized_text(claim["quote"]) in primary
            for claim in raw["claims"]
        ):
            raise ValueError("MANAGEMENT_ANCHOR_LATEST_EXPERIENCE_CLAIM_REQUIRED")
        facts.append(
            {
                "name": person,
                "appointment_start": row["start"],
                "end": row.get("end"),
                "latest_report_role_quote_verified": True,
                "full_report_same_person_and_start_verified": True,
            }
        )
    return {
        "anchor_source": anchor,
        "latest_source": report_hash,
        "fund_code": code,
        "legal_name": inputs["legal_name"],
        "anchor_period": anchor_period,
        "latest_period": primary_period,
        "known_latest_full_period": max(known_full),
        "active_managers": facts,
        "scope": "同产品两份原件及最新季度现任经理、实际管理任期绑定，不是仅比较公司名。",
    }
