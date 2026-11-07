"""新规则管理分的原始证据复算；正面履职原件不能与检索未命中混为一谈。"""

import re
import unicodedata
from datetime import date
from decimal import Decimal, localcontext
from pathlib import Path

from app.services.fund_rating_r5_rules import continuity_score
from app.services.fund_rating_reviewed_quality import STATES
from app.services.material_content import source_text


def _plain(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def _source_text(root: Path, reference: dict) -> str:
    from app.services.fund_rating_reviewed_snapshot import _read

    parsed = _read(reference, root)
    if reference["kind"] == "pdf":
        from app.services.fund_rating_pdf import text_pages

        return _plain("".join(text_pages(root / reference["file"])))
    from bs4 import BeautifulSoup

    if reference["kind"] in {"html", "htm", "text"}:
        # 部分公开转载实际返回GB编码纯文本，文件后缀不代表内容格式。
        # 摘要核验仍先执行；按回执明示编码严格解码，拒绝丢字替换和猜测。
        encoding = reference.get("encoding", "utf-8")
        if encoding not in {"utf-8", "utf-8-sig", "gb18030"}:
            raise ValueError("RATING_MANAGEMENT_SOURCE_ENCODING")
        content = source_text(root / reference["file"], encoding=encoding)
        return _plain(BeautifulSoup(content, "html.parser").get_text() if reference["kind"] != "text" else content)

    def strings(value):
        if isinstance(value, str):
            return [BeautifulSoup(value, "html.parser").get_text() if "<" in value and ">" in value else value]
        if isinstance(value, dict):
            return [part for child in value.values() for part in strings(child)]
        if isinstance(value, list):
            return [part for child in value for part in strings(child)]
        return []

    return _plain("\n".join(strings(parsed)))


def calculate(
    raw: dict,
    *,
    root: Path,
    manager_name: str,
    as_of: str,
    fund_code: str | None = None,
    legal_name: str | None = None,
    primary_report_hash: str | None = None,
) -> dict:
    """履历40%、责任接续20%、治理40%，预算和R5事实档位不变。

    原件逐件验摘要并核实际引用；正常治理须跨期明确的管理人声明和本产品实际
    托管履职，不把托管人无处罚、目录、无检索命中或未知历史变成正常分。
    """
    if raw["manager_name"] != manager_name or raw["as_of_date"] != as_of:
        raise ValueError("RATING_MANAGEMENT_SCOPE")
    sources = raw["sources"]
    text = {key: _source_text(root, ref) for key, ref in sources.items()}
    for claim in raw["claims"]:
        if not claim.get("quote") or _plain(claim["quote"]) not in text[claim["source_hash"]]:
            raise ValueError("RATING_MANAGEMENT_QUOTE_MISMATCH")
    experience = raw["experience"]
    # 本适配器只补已披露的基本职责档；强化经验/跨期优秀档需要另行完整原始观察。
    if experience["state"] != "BASIC_ROLE_EVIDENCED" or not experience.get("source_refs"):
        raise ValueError("RATING_MANAGEMENT_EXPERIENCE_FACTS")
    if not any(
        c["source_hash"] in experience["source_refs"] and "基金经理" in _plain(c["quote"]) for c in raw["claims"]
    ):
        raise ValueError("RATING_MANAGEMENT_ACTUAL_ROLE_QUOTE_REQUIRED")
    if not set(experience["source_refs"]) <= sources.keys():
        raise ValueError("RATING_MANAGEMENT_EXPERIENCE_SOURCE")
    if fund_code is not None or legal_name is not None:
        # 经理职责属于同一法律产品，招募书未必列每个份额代码；份额由本基金
        # 原始定期报告核准，职责原件仍必须出现完整法律产品名，不能借其他基金。
        product_report = text.get(primary_report_hash, "")
        if (
            not fund_code
            or not legal_name
            or fund_code not in product_report
            or not any(_plain(legal_name) in text[ref] for ref in experience["source_refs"])
        ):
            raise ValueError("RATING_MANAGEMENT_PRODUCT_ROLE_IDENTITY")
    continuity = raw["continuity"]
    continuity_value = None
    if continuity is not None:
        if raw.get("fund_inception"):
            inception = raw["fund_inception"]
            quote = _plain(inception["quote"])
            if quote not in text[inception["source_hash"]]:
                raise ValueError("RATING_MANAGEMENT_INCEPTION_SOURCE")
            match = re.search(r"基金合同生效日[:：]?(?:为)?(20\d{2})年(\d{1,2})月(\d{1,2})日", quote)
            if not match or str(date(*map(int, match.groups()))) != inception["date"]:
                raise ValueError("RATING_MANAGEMENT_INCEPTION_DATE")
            expected_start = max(raw["assessment_window_start"], inception["date"])
            if continuity["period_start"] != expected_start or inception["date"] > as_of:
                raise ValueError("RATING_MANAGEMENT_LIFETIME_SCOPE")
        if not set(continuity["source_refs"]) <= sources.keys():
            raise ValueError("RATING_MANAGEMENT_CONTINUITY_SOURCE")
        for span in continuity["intervals"]:
            if not set(span["source_refs"]) <= sources.keys():
                raise ValueError("RATING_MANAGEMENT_PERSON_SOURCE")
        continuity_value, _ = continuity_score(continuity)
    governance = raw["governance"]
    if (
        governance.get("state") not in {"NORMAL_REVIEWED", "REMEDIATION_DISCLOSED"}
        or governance.get("negative_event_review_complete") is not True
    ):
        raise ValueError("RATING_MANAGEMENT_GOVERNANCE_REVIEW")
    events = governance.get("known_applicable_events")
    if governance["state"] == "NORMAL_REVIEWED" and events != []:
        raise ValueError("RATING_MANAGEMENT_GOVERNANCE_EVENT_REQUIRES_OWN_STATE")
    if governance["state"] == "REMEDIATION_DISCLOSED":
        if not events:
            raise ValueError("RATING_MANAGEMENT_REMEDIATION_ORIGINAL_REQUIRED")
        for event in events:
            quote = _plain(event["quote"])
            # 原报告也可能明确披露“整改成果验收通过、措施解除”。这是发行人
            # 的完成声明，不能因未使用“完成整改”四字误判为只有未来整改意向；
            # 仍沿整改观察50档，不把发行人声明当独立监管核验或优秀治理。
            completed = (
                re.search("完成整改|已整改完成|整改报告|完成.*整改工作", quote)
                or re.search(r"整改成果(?:已经|已)(?:(?!计划|争取|申请|尚未|未获).){0,60}验收通过", quote)
                or re.search(
                    # 嘉实原报告使用“公司已从制度建设、流程机制等方面整改完毕”。
                    # 这是已完成的明确声明；未来意向、尚未完成仍不能通过，
                    # 分档仍保留整改观察50，不据发行人声明认定独立验收。
                    r"(?:公司|管理人)已从(?:(?!计划|拟|尚未|未完成|将).){1,80}整改完毕",
                    quote,
                )
                or re.search(
                    # 整改栏标签“如提出整改意见）”跨列，把“完成整改”拆为
                    # “完见）成整改”；只接纳这一明确标签尾部和实际已验收
                    # 声明，原quote仍完整匹配原件，不把未来整改意向认成完成。
                    r"管理人已采取(?:(?!计划|拟|尚未|将|未通过).){1,80}完见[)）]成整改[，,]并通过监管机构验收",
                    quote,
                )
            )
            # 方正富邦年报两列表格将左侧“如提出整改意见）”插入右侧
            # 完成整改句。限定已核管理人、税务机关和已补税缴罚款事实，
            # 只识别原文明确完成；不会将计划整改或未来承诺转成完成。
            completed = completed or (
                manager_name == "方正富邦基金管理有限公司"
                and "受到调查或处罚等措施的主体" + _plain(manager_name) in quote
                and "国家税务总局北京市税务局第二稽查局" in quote
                and "截止本报告期末,公司已主动补缴税款、缴纳罚款," in quote
                and re.search(r"并完成(?:提出整改意见\)整改工作|整出整改意见\)改工作)[。.]", quote)
            )
            if (
                quote not in text[event["source_hash"]]
                or event["date"] > as_of
                or not (
                    re.search("警示函|责令改正|罚款", quote)
                    # 上交所自律警示是原文明确的负面措施，仍沿整改50档。
                    or all(
                        term in quote
                        for term in (
                            "其他(自律监管措施)",
                            "受到的具体措施类型警示",
                            "采取调查或处罚等措施的机构上海证券交易所",
                        )
                    )
                )
                or not completed
            ):
                raise ValueError("RATING_MANAGEMENT_REMEDIATION_NOT_DISCLOSED")
    # 法定管理人更名不等于更换主体。旧名称只能由原件中的明确更名句、
    # 生效日及当前名称共同核准，不能把任意别名或其他管理人的报告并入。
    allowed_manager_names = {manager_name}
    for event in raw.get("manager_identity_events", []):
        old_name, new_name = event["old_name"], event["new_name"]
        quote = _plain(event["quote"])
        effective = date.fromisoformat(event["effective_on"])
        if (
            new_name != manager_name
            or not old_name
            or old_name == new_name
            or event["effective_on"] > as_of
            or event["source_hash"] not in text
            or quote not in text[event["source_hash"]]
            or f"{effective.year}年{effective.month}月{effective.day}日" not in quote
            or _plain(old_name + "变更名称为" + new_name) not in quote
        ):
            raise ValueError("RATING_MANAGEMENT_LEGAL_NAME_CHANGE_NOT_VERIFIED")
        allowed_manager_names.add(old_name)
    declared = set()
    for statement in governance["statements"]:
        if statement["manager_name"] not in allowed_manager_names or statement["scope"] not in {
            "MANAGER",
            "MANAGER_AND_SENIOR_STAFF",
        }:
            raise ValueError("RATING_MANAGEMENT_GOVERNANCE_SUBJECT")
        if _plain(statement["manager_name"]) not in text[statement["source_hash"]][:16000]:
            raise ValueError("RATING_MANAGEMENT_REPORT_MANAGER_IDENTITY")
        quote = _plain(statement["quote"])
        explicit_no_event = re.search(
            r"(?:基金)?管理人.*(?:无受|未出现|未受|没有受|未发生受).*(?:调查|稽查|处罚)", quote
        )
        explicit_no_event = explicit_no_event or ("管理人受调查或处罚等情况" in quote and "管理人无相关情况" in quote)
        # 完整的管理人小节可直接写“无”；标题必须明确对象，裸“无”、
        # 从业人员或托管人小节均不适用，跨期来源和负面事项核验仍保留。
        explicit_no_event = explicit_no_event or re.fullmatch(
            r"(?:\d{1,2}\.){1,3}\d{1,2}管理人受调查或处罚等情况[:：]?无[。.]?", quote
        )
        explicit_no_event = explicit_no_event or re.fullmatch(
            r"\d+\.\d+\.1管理人受调查或处罚等情况本报告期内[，,]本基金管理人不存在受调查或处罚的情况[。.]",
            quote,
        )
        # 旧报告也有先写否定、再写管理人对象的句式；仅接受明示调查/
        # 稽查/处罚的完整对象，不以无搜索结果或一般合规声明替代。
        explicit_no_event = explicit_no_event or re.search(
            r"本报告期内[,，]?(?:无|未发生)(?:基金)?管理人(?:及(?:其)?高级管理人员)?受(?:稽查|调查)或处罚(?:的|等)?(?:情况|情形)[。.]?",
            quote,
        )
        explicit_no_event = explicit_no_event or re.search(
            r"(?:报告期内[，,]?本基金|本基金本报告期)无涉及管理人(?:及其高级管理人员)?受(?:稽查|调查)或处罚等情况[。.]?",
            quote,
        )
        # 新版定报把管理人和从业人员拆为相邻小节，正文可能只写“无。”。
        # 仅接受同一章节连续的管理人小节及下一小节标题，避免把裸“无”、
        # 托管人声明、目录页码或跨章节拼接误当成管理人的无事项声明。
        explicit_no_event = explicit_no_event or re.fullmatch(
            r"(\d+)\.(\d+)\.1管理人受调查或处罚等情况无[。.]\1\.\2\.2管理人相关从业人员受调查或处罚等情况",
            quote,
        )
        disclosed_event = any(e["source_hash"] == statement["source_hash"] for e in events or [])
        if quote not in text[statement["source_hash"]] or not (explicit_no_event or disclosed_event):
            raise ValueError("RATING_MANAGEMENT_NO_EXPLICIT_MANAGER_DECLARATION")
        if statement["published_on"] > as_of:
            raise ValueError("RATING_MANAGEMENT_FUTURE_DECLARATION")
        declared.add(statement["period_end"])
    anchor = date.fromisoformat(max(declared))
    if anchor > date.fromisoformat(as_of) or anchor.month not in {6, 12}:
        raise ValueError("RATING_MANAGEMENT_DECLARATION_PERIOD")
    half_index = anchor.year * 2 + (1 if anchor.month == 12 else 0)
    required = {f"{(half_index - i) // 2}-{'12-31' if (half_index - i) % 2 else '06-30'}" for i in range(6)}
    if not required <= declared:
        raise ValueError("RATING_MANAGEMENT_GOVERNANCE_HISTORY_MISSING")
    if len({s["source_hash"] for s in governance["statements"]}) < len(required):
        raise ValueError("RATING_MANAGEMENT_DISTINCT_PERIOD_ORIGINALS_REQUIRED")
    quote = _plain(governance["trustee_execution"]["quote"])
    source = governance["trustee_execution"]["source_hash"]
    if primary_report_hash is not None and source != primary_report_hash:
        raise ValueError("RATING_MANAGEMENT_PRODUCT_TRUSTEE_IDENTITY")
    actual_review = ("监督" in quote and any(word in quote for word in ["对账", "复核"])) or (
        "投资运作" in quote and "净值计算" in quote and "托管人未发现损害基金份额持有人利益" in quote
    )
    # 招商等原件将“净值的计算”“基金持有人”作为同义表述；仍须
    # 同时核本产品投资运作、申赎计价及明确未发现损害的实际托管正文。
    actual_review = actual_review or (
        "本基金投资运作" in quote
        and re.search(r"基金资产净值(?:的)?计算", quote)
        and re.search(r"基金份额申购赎回价格(?:的)?计算", quote)
        and re.search(r"托管人未发现损害基金(?:份额)?持有人利益的行为", quote)
    )
    if quote not in text[source] or "报告期" not in quote or not actual_review:
        raise ValueError("RATING_MANAGEMENT_ACTUAL_TRUSTEE_EXECUTION_MISSING")
    if re.search(r"\.{4,}", quote):
        raise ValueError("RATING_MANAGEMENT_TABLE_OF_CONTENTS_NOT_EXECUTION")
    with localcontext() as context:
        context.prec = 65
        governance_value = Decimal(STATES[governance["state"]])
        known = Decimal(STATES[experience["state"]]) * Decimal(".4") + governance_value * Decimal(".4")
        lower = known + (continuity_value * Decimal(".2") if continuity_value is not None else 0)
        upper = lower if continuity_value is not None else known + Decimal(20)
    return {
        "score": str(lower) if lower == upper else None,
        "lower": str(lower),
        "upper": str(upper),
        "experience": str(STATES[experience["state"]]),
        "continuity": str(continuity_value) if continuity_value is not None else None,
        "governance": str(governance_value),
        "scope": "PUBLIC_DISCLOSED_MANAGER_AND_ACTUAL_TRUSTEE_EXECUTION_NOT_REGULATORY_AUDIT",
    }
