"""核对非利润披露用途，阻止监管报告或薪酬补充被当作公司利润修订。"""

from app.services.fund_earnings_identity_v1 import occurrence
from app.services.fund_information_history_v1 import normalize


def review_non_profit_sources(specs, documents, verified_ids, money_specs):
    """重放人工登记的用途证据，并排斥同一原件生成利润金额字段。

    参数 documents 为以公告身份编号为键的已存原件；verified_ids 必须来自
    身份和日期核验。每项用途同时约束目录标题和正文锚点，不能仅凭“补充”
    二字排除所有修订。只标记当前研究的利润变化用途，不否定原件的其他价值。
    """
    money_ids = {s["document_id"] for s in money_specs}
    seen, results = set(), []
    for spec in specs:
        key = spec["document_id"]
        if key in seen:
            raise ValueError("DUPLICATE_SOURCE_SCOPE")
        seen.add(key)
        if key not in documents or key not in verified_ids:
            raise ValueError("SOURCE_SCOPE_IDENTITY_NOT_VERIFIED")
        document = documents[key]
        if not document["body_saved"] or document["revision_issues"]:
            raise ValueError("SOURCE_SCOPE_BODY_OR_DATE_INVALID")
        if key in money_ids:
            raise ValueError("NON_PROFIT_SOURCE_USED_FOR_PROFIT_CHANGE")
        title = normalize(document["row"]["title_plain"])
        role = spec["role"]
        if role == "REGULATORY_SOLVENCY_REPORT":
            title_token, body_token = "偿付能力", "保险公司偿付能力报告摘要"
        elif role == "REMUNERATION_SUPPLEMENT":
            title_token, body_token = "年度报告补充公告", "年度最终全部薪酬情况披露如下"
        else:
            raise ValueError("UNKNOWN_NON_PROFIT_SOURCE_ROLE")
        if title_token not in title or not spec["anchors"]:
            raise ValueError("SOURCE_SCOPE_TITLE_OR_ANCHORS_INVALID")
        anchors = [occurrence(document["pages"], p, q) for p, q in spec["anchors"]]
        if not any(body_token in normalize(q) for _, q in spec["anchors"]):
            raise ValueError("SOURCE_SCOPE_BODY_ROLE_NOT_ESTABLISHED")
        results.append({
            **spec,
            "anchors": anchors,
            "source_sha256": document["receipt"]["sha256"],
            "published_date": document["row"]["published_date"],
            "excluded_from_profit_changes": True,
            "training_ready": False,
        })
    return results
