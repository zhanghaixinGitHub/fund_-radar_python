"""综合结果的独立时间及引用主体核对；不靠模型自审，不用未来实际收益打分。"""

import re


def synthesis_issues(raw: dict, facts: dict, companies: list[dict], timing: dict) -> list[dict]:
    """必要条件检查：改变条件属于目标日、已点名的公司须受本条引用支持。

    主体词在证据中出现仍不能证明因果关系，未命中也不能证明预测有效。公司
    清单只来自冻结持仓，不引入外部行业分类。日期不自行推算，沿用冻结时间契约。
    """
    issues = []
    target = timing["target_date"]
    for index, condition in enumerate(raw.get("change_conditions", [])):
        dates = re.findall(r"20\d{2}-\d{2}-\d{2}", condition)
        if any(day != target for day in dates):
            issues.append(
                {"code": "CHANGE_CONDITION_TARGET_DATE_MISMATCH", "index": index, "target": target, "raw": condition}
            )
    for section in ("reasons", "counterpoints"):
        for index, reason in enumerate(raw.get(section, [])):
            refs = reason.get("refs", [])
            unknown = set(refs) - set(facts)
            if unknown:
                issues.append(
                    {"code": "UNKNOWN_FACT_REFERENCE", "section": section, "index": index, "refs": sorted(unknown)}
                )
                continue
            evidence = "\n".join(facts[ref].get("text", "") + "\n" + facts[ref].get("relation", "") for ref in refs)
            for company in companies:
                name = company["name"]
                if name in reason.get("meaning", "") and name not in evidence:
                    issues.append(
                        {
                            "code": "REASON_COMPANY_NOT_SUPPORTED_BY_REFS",
                            "section": section,
                            "index": index,
                            "company": name,
                            "refs": refs,
                            "raw": reason["meaning"],
                        }
                    )
    return issues
