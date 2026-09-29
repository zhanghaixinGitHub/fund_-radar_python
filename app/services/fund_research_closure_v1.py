"""002112 剩余资料的有界收尾工具。

目录补采与正文准入分开：旧原件停止记录继续有效，完成目录不能解除身份、
版本或语义限制。只查询已有研究日期与公司，不读取答案或改变训练样本。
"""

from collections import defaultdict
from datetime import date, timedelta

from app.services.fund_information_history_v1 import covered


def split_windows(gaps, maximum_days=30):
    """把大报告组拆成连续查询片段；片段并集必须等于原缺口，不能丢边界日。

    maximum_days 限制单次目录查询跨度，与模型的 30 日观察窗口无关。
    返回顺序只由证券代码和日期决定，不依据已知预测结果选择资料。
    """
    if isinstance(maximum_days, bool) or not isinstance(maximum_days, int) or maximum_days < 1:
        raise ValueError("INVALID_QUERY_SPAN")
    merged = defaultdict(list)
    for row in gaps:
        lo, hi = date.fromisoformat(row["start"]), date.fromisoformat(row["end"])
        if lo > hi or hi >= date(2025, 1, 1):
            raise ValueError("INVALID_OR_SEALED_QUERY_RANGE")
        merged[row["stock"]].append((lo, hi))
    result = []
    for stock, intervals in sorted(merged.items()):
        union = []
        for lo, hi in sorted(intervals):
            if union and lo <= union[-1][1] + timedelta(days=1):
                union[-1] = (union[-1][0], max(hi, union[-1][1]))
            else:
                union.append((lo, hi))
        for lo, hi in union:
            while lo <= hi:
                end = min(hi, lo + timedelta(days=maximum_days - 1))
                result.append({"stock": stock, "start": str(lo), "end": str(end)})
                lo = end + timedelta(days=1)
    return result


def project_coverage(previous, catalogs):
    """按原记录键累计完整窗口，任何失败片段都不能计入覆盖。

    previous 的 remaining_companies 是上轮真实缺口；每条仍须覆盖目标日前
    完整 30 天。保留所有记录，未知不填零，不删除难补的基金或公司。
    """
    spans = defaultdict(list)
    for item in catalogs:
        if item["catalog_complete"]:
            row = item["window"]
            spans[row["stock"]].append((row["start"], row["end"]))
    result, keys = [], set()
    for row in previous:
        key = (row["fund_code"], row["target"])
        if key in keys:
            raise ValueError("DUPLICATE_RESEARCH_ROW")
        keys.add(key)
        target = date.fromisoformat(row["target"])
        lo, hi = str(target - timedelta(days=30)), str(target - timedelta(days=1))
        missing = [s for s in row["remaining_companies"] if not covered(spans[s], lo, hi)]
        if not row["catalog_complete"] and not row["remaining_companies"]:
            raise ValueError("INCOMPLETE_ROW_WITHOUT_GAP")
        result.append(
            {**row, "remaining_companies": missing, "catalog_complete": row["catalog_complete"] or not missing}
        )
    return result


def admission_status(*, catalog_complete, body_complete, semantics_complete, protocol_frozen, fit_budget):
    """训练入口的必要前置条件；预算和资料检查不得互相替代。"""
    checks = {
        "catalog_complete": catalog_complete is True,
        "body_complete": body_complete is True,
        "semantics_complete": semantics_complete is True,
        "protocol_frozen": protocol_frozen is True,
        "positive_bounded_budget": type(fit_budget) is int and 0 < fit_budget <= 12,
    }
    return {"checks": checks, "fit_execution_allowed": all(checks.values())}
