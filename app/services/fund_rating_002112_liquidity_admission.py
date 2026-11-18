"""从各产品自身原件复核普通日开放条件，不把流动性排除扩大到风险或结构组。"""

import re
from datetime import date

from app.services.fund_rating_thematic_source import RawArchive, iso, plain, require

VERSION = "R5_002112_MONTHLY_LIQUIDITY_ADMISSION_V1"


def _section(text, heading, following):
    start = text.rfind(heading)
    if start < 0:
        return ""
    end = text.find(following, start + len(heading))
    return text[start:end] if end >= 0 else ""


def operating_decision(body, *, legal_name, as_of_date, supplement=None):
    """只读本产品基本情况；管理人其他产品、赎回费持有期限不属于运作限制。"""
    text, name = plain(body), plain(legal_name)
    require(name in text[:20000], "LIQUIDITY_OPERATION_LEGAL_IDENTITY")
    info = _section(text, "2.1基金基本情况", "2.2基金产品说明")
    basis = _section(text, "6.4.1基金基本情况", "6.4.2")
    # 有些真实报告目录沿用旧编号，而正文整体少一节；以有法定名称的正文为准。
    if name not in info:
        info = _section(text, "1.1基金基本情况", "1.2基金产品说明")
    if not basis:
        basis = _section(text, "5.4.1基金基本情况", "5.4.2")
    mode = re.search(r"基金运作方式(.*?)基金合同生效日", info)
    mode = mode[1] if mode else ""
    proof = {"operating_clause": mode, "basic_clause": basis[:1800]}
    restricted = r"定期开放|定开|持有期|封闭式|指数|联接|FOF"
    if (
        re.search(restricted, name)
        or re.search(r"定期开放|定开|持有期|封闭式", mode)
        or re.search(r"本基金(?:为|采用).{0,15}(?:定期开放|滚动持有|最短持有)", basis)
    ):
        return {"status": "EXCLUDED", "reason": "本产品法定名称或自身运作条款为非普通日开放结构", **proof}
    if not info or name not in info:
        return {"status": "PENDING", "reason": "缺本产品完整基本情况原件", **proof}
    # 初始封闭不是永久排除。必须读到已经生效的实际转型日期，不能按年数推算。
    transitions = []
    for match in re.finditer(
        r"(\d{4})年(\d{1,2})月(\d{1,2})日(?:起|日起).{0,90}?(?:转型为|转换为|自动转型为)上市开放式",
        basis,
    ):
        day = date(*map(int, match.group(1, 2, 3))).isoformat()
        if day <= iso(as_of_date):
            transitions.append({"date": day, "quote": match[0]})
    if "封闭期" in mode and not transitions:
        return {"status": "PENDING", "reason": "初始封闭条款尚缺实际已生效的开放转型证明", **proof}
    ordinary = bool(re.search(r"契约(?:型)?[、,]?普通?开放式|契约(?:型)?[、,]?开放式|上市契约型开放式", mode))
    ordinary = ordinary or "契约型,上市开放式" in mode
    # 部分PDF基本表列次序交错，只允许本产品财务附注的明确法律运作说明补证。
    ordinary = ordinary or bool(re.search(r"本基金为契约型[、,]?开放式(?:证券投资)?基金", basis))
    if supplement:
        extra = plain(supplement)
        require(name in extra[:12000], "LIQUIDITY_OPERATION_SUPPLEMENT_IDENTITY")
        explicit = "运作方式普通开放式开放频率每个开放日" in extra
        proof["supplement_clause"] = extra[max(0, extra.find("运作方式") - 20) : extra.find("运作方式") + 100]
        ordinary = ordinary or explicit
    if not ordinary:
        return {"status": "PENDING", "reason": "契约型或LOF标签本身未证明普通日开放，须补明确开放方式", **proof}
    return {
        "status": "ADMITTED",
        "reason": "本产品原件确认普通契约日开放；临时申赎限额另作付款条件观察",
        "transitions": transitions,
        **proof,
    }


def compile_admission(inputs, *, root):
    """每次重放347产品上游准入图及实际原文；不接收手填名单或旧daily_open_identity。"""
    from app.services.fund_quality_v2_disclosures import report_period
    from app.services.fund_rating_active_monthly_admission import replay_monthly_admission

    require(inputs.get("version") == VERSION and inputs.get("fund_code") == "002112", "LIQUIDITY_ADMISSION_SCOPE")
    source = RawArchive(inputs, root)
    ref = inputs["monthly_admission_ref"]
    require(source.sources[ref]["archive_kind"] == "DERIVED", "LIQUIDITY_ADMISSION_INPUT_GRAPH")
    upstream_inputs = source.json(ref)
    require(upstream_inputs["as_of_date"] == inputs["as_of_date"], "LIQUIDITY_ADMISSION_SAME_PERIOD")
    upstream = replay_monthly_admission(upstream_inputs, root=root)
    require(upstream["status"] == "EXPOSURE_ADMISSION_COMPLETE", "LIQUIDITY_ADMISSION_UPSTREAM_PENDING")
    peers = upstream["structure_codes"]
    require(
        len(peers) == len(set(peers)) and bool(peers) and not {"001412", "002112"} & set(peers),
        "LIQUIDITY_ADMISSION_PEER_IDENTITY",
    )
    require(set(peers) <= set(inputs["portfolios"]), "LIQUIDITY_ADMISSION_CANNOT_REMOVE_PEER")
    rows = []
    for code in peers:
        item = inputs["portfolios"][code]
        original = item["source_ref"]
        source.original(original)
        require(
            code in item["share_codes"] and iso(item["period"]) <= iso(item["published_on"]) <= source.asof,
            "LIQUIDITY_OPERATION_REPORT_PERIOD",
        )
        body = source.body(original)
        require(
            str(report_period(body)[1]) == item["period"]
            and source.sources[original].get("published_on") == item["published_on"],
            "LIQUIDITY_OPERATION_ACTUAL_REPORT_PERIOD",
        )
        require(code in plain(body)[:22000], "LIQUIDITY_OPERATION_REPORT_CODE")
        supplement = None
        used = [original]
        if item.get("operation_supplement_ref"):
            extra = item["operation_supplement_ref"]
            source.original(extra)
            # 交易状态证明可事后取得；运作方式则必须在评分截止前已经公开。
            require(
                bool(source.sources[extra].get("published_on"))
                and iso(source.sources[extra]["published_on"]) <= source.asof,
                "LIQUIDITY_OPERATION_SUPPLEMENT_NOT_HISTORICAL_PRICE",
            )
            supplement = source.body(extra)
            require(code in plain(supplement)[:12000], "LIQUIDITY_OPERATION_SUPPLEMENT_CODE")
            used.append(extra)
        decision = operating_decision(
            body, legal_name=item["legal_name"], as_of_date=source.asof, supplement=supplement
        )
        rows.append({"code": code, "legal_name": item["legal_name"], "source_refs": used, **decision})
    return {
        "version": VERSION,
        "fund_code": "002112",
        "as_of_date": source.asof,
        "admitted_codes": [r["code"] for r in rows if r["status"] == "ADMITTED"],
        "excluded": [r for r in rows if r["status"] == "EXCLUDED"],
        "pending": [r for r in rows if r["status"] == "PENDING"],
        "rows": rows,
        "used_source_refs": sorted(source.used),
        "monthly_admission_ref": ref,
        "status": "OPERATION_PENDING" if any(r["status"] == "PENDING" for r in rows) else "COMPLETE",
    }
