"""按已核对的原件用途修正标题候选范围；不修改冻结的第二版分类器。"""

import re

from app.services.fund_earnings_semantics_v2 import earnings_title_kind
from app.services.fund_information_history_v1 import normalize

# 证据是第四批 1211338408、1212174063 的正文主体，并非根据预测效果挑公司。
# 这里只声明“此标题中的财务报表主体不是目录公司的报表主体”，不推定历年股权关系。
DIFFERENT_REPORT_SUBJECTS = {"601318": ("平安银行",)}


def classify_catalog_row(row):
    """下载前限定公司自身业绩及其修订；保留旧类别、排除原因和原始标题。

    标题判断只是候选范围。正文代码、公开日期、版本和金额仍需独立核验。
    更正通知不当作已经核实的正式财务结果；其追述旧值不可倒填早期事实。
    """
    title = normalize(row["title_plain"])
    previous = earnings_title_kind(title)
    common = {"previous_kind": previous, "training_ready": False}
    if any(term in title for term in ("说明会", "說明會", "企业年度报告书", "企業年度報告書")):
        return {**common, "kind": None, "reason": "NON_EARNINGS_MEETING_OR_CORPORATE_RETURN"}
    if previous and any(name in title for name in DIFFERENT_REPORT_SUBJECTS.get(row["secCode"], ())):
        reason = (
            "DISCLOSURE_FORWARDING_NOTICE_NOT_FINANCIAL_REPORT"
            if "关于披露" in title
            else "VERIFIED_DIFFERENT_FINANCIAL_REPORT_SUBJECT"
        )
        return {**common, "kind": None, "reason": reason}
    if previous and re.search(r"(?:更正|修正|修订|补充).*公告", title):
        return {**common, "kind": "CORRECTION_NOTICE", "reason": "EXPLICIT_CORRECTION_NOTICE"}
    return {**common, "kind": previous, "reason": "LEGACY_SCOPE_RETAINED"}


def identity_matching_text(text):
    """仅用于身份标题等价核对，原目录标题、原文及版本公开日均不变。

    处理已经出现的格式差异：下划线、文件名后缀、更新版后缀及“预增的公告”。
    更新版后缀移除只影响标题匹配，绝不改变正文版本或把公开日提前。
    """
    # 保留原始空白，避免证券代码和相邻年份拼接成一个数字，破坏独立代码核验。
    value = text.replace("_", "")
    value = re.sub(r"[（(](?:更新后|更新版|更正后|修订版|修订稿)[）)]", "", value)
    value = re.sub(r"\.(?:docx|pdf)$", "", value, flags=re.IGNORECASE)
    return re.sub(r"(业绩预(?:增|减|盈|亏))的公告", r"\1公告", value)
