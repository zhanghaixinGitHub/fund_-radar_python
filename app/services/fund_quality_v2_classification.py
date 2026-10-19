"""按正式产品目标补足无QDII字样和指数研究增强的类别识别。

只解析已完成身份核验的概要/报告正文；网站类型、名称检索结果和基金经理
履历不是投资目标证明。该候选适配仍不能替代历史合同变更的逐期核验。
"""

import re
import unicodedata


def classify_original(title: str, report_text: str = "") -> dict:
    """返回类别及实际原文依据；境外目标不明确时拒绝落入境内默认分组。

    title为原文封面标题，report_text为同产品已核身份报告。只在2.2产品
    说明内读取投资目标和风险特征，不从经理履历、市场展望或持仓地域猜。
    """
    def clean(value):
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))

    header = clean(title).upper()
    body = clean(report_text)
    sections = [match[1] for match in re.finditer(r"2\.2基金产品说明(.{1,6000}?)(?=2\.3|2\.4|§3)", body[:24000])
                if "投资目标" in match[1] and "风险收益特征" in match[1]]
    objective, risk = "", ""
    if len(set(sections)) == 1:
        section = sections[0]
        objective = re.split(r"投资策略|业绩比较基准|风险收益特征", section.split("投资目标", 1)[1], maxsplit=1)[0]
        risk = section.split("风险收益特征", 1)[1]
    foreign_objective = bool(re.search(
        r"全球范围内.{0,18}股票|全球.{0,10}股票|美国.{0,20}股票|境外.{0,10}股票|海外.{0,10}股票", objective
    ))
    foreign_risk = any(phrase in risk for phrase in (
        "本基金为全球证券投资基金", "本基金主要投资于美国", "本基金主要投资于境外", "本基金为境外投资基金",
    ))
    foreign = foreign_objective and (foreign_risk or "境外资产托管人" in body[:16000])
    scope_hint = "QDII" in header or any(word in header for word in ("全球", "美国", "海外", "境外"))
    kind, rule, quote = None, "UNSUPPORTED_OR_UNRESOLVED_OBJECTIVE", title
    if "QDII" in header or foreign:
        if "股票" in header and not any(word in header for word in ("指数", "ETF", "联接", "FOF", "基金中基金")):
            kind = "GLOBAL_ACTIVE_EQUITY"
            rule = "EXPLICIT_QDII_EQUITY_TITLE" if "QDII" in header else "FORMAL_PRODUCT_OBJECTIVE_AND_FOREIGN_SCOPE"
            quote = title if "QDII" in header else "投资目标" + objective
    elif scope_hint or foreign_objective:
        rule = "FOREIGN_OBJECTIVE_SCOPE_REVIEW_REQUIRED"
    elif re.search(r"指数(?:精选|研究|量化)?增强|增强指数", header) or any(
        phrase in objective + risk for phrase in ("本基金为增强型指数基金", "本基金为股票指数增强型基金")
    ):
        kind, rule = "INDEX_ENHANCED", "FORMAL_INDEX_ENHANCEMENT_OBJECTIVE"
    elif any(word in header for word in ("指数", "ETF", "联接")) and not any(
        word in header for word in ("债券", "黄金", "商品", "FOF", "基金中基金")
    ):
        kind, rule = "PASSIVE_INDEX", "FORMAL_DOMESTIC_INDEX_TITLE"
    elif any(word in header for word in ("股票", "混合")) and not any(word in header for word in ("FOF", "基金中基金")):
        kind, rule = "ACTIVE_EQUITY", "FORMAL_DOMESTIC_EQUITY_TITLE"
    return {"kind": kind, "rule": rule, "excerpt": quote, "product_section_parsed": bool(objective),
            "objective_excerpt": objective, "risk_scope_excerpt": risk}
