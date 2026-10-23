"""基金详情的业务说明：只读已核验正式批次，不执行评分或读取研究试算。"""

import re
from decimal import Decimal, InvalidOperation

from app.services.fund_rating_rules import DIMENSIONS


def _number(value: object) -> Decimal | None:
    """将已保存的数值转为展示值；未知、布尔及非有限数均不伪装为零。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() else None


def _point(value: object) -> str | None:
    point = _number(value)
    return f"{point:.2f}" if point is not None and 0 <= point <= 100 else None


def _money(name: str, value: Decimal, period: str) -> dict:
    """公开人民币金额按量级展示，避免十几位数字掩盖基金判断；比例仍用原金额算。"""
    divisor, unit = (Decimal("100000000"), "亿元") if value >= 100000000 else (
        (Decimal("10000"), "万元") if value >= 10000 else (Decimal(1), "元")
    )
    return {"name": name, "value": f"{value / divisor:,.2f}", "unit": unit, "period": period}


def enrich_published_detail(item: dict, result, snapshot: dict) -> dict:
    """补充同批次总分、八维分数及公开事实，不改变持久化结果或既有说明。

    item 的状态已经过批次哈希、来源资格、时效及新增事项检查；只有仍然有效
    的正式评级可展示数字分。撤回、过期和待复核时清空当前数字分，不把旧分
    留在页面。列表摘要和评分计算链路不调用本函数。
    """
    from app.services.fund_rating_reviewed_snapshot import BASIS as SNAPSHOT_BASIS

    if snapshot.get("method", {}).get("basis") == SNAPSHOT_BASIS:
        output = {**item, "score": None}
        if item.get("status") in {"PARTIAL", "NOT_RATED"}:
            return output
        # 失效的区间/部分分与完整分一样隐藏，不把旧结果继续当当前结果。
        output.update(scoreRange=None, partialScore=None, scoreCoverage=None)
        output["dimensions"] = [{**d, "score": None, "scoreRange": None} for d in item.get("dimensions", [])]
        return output
    rated = item.get("status") == "RATED" and result.status == "RATED"
    values = result.dimension_values or {}
    output = {**item, "score": _point(result.score) if rated else None}
    dimensions = []
    for dimension in item.get("dimensions", []):
        key = dimension["key"]
        dimensions.append({
            **dimension,
            "score": _point(values.get(key)) if rated and key in DIMENSIONS else None,
            "metrics": [dict(metric) for metric in dimension.get("metrics", [])],
        })
    output["dimensions"] = dimensions
    if not rated:
        return output

    # 使用已经随正式结果封存的资料；不按基金代码硬编码，也不取隔离研究目录。
    member = next((m for m in snapshot.get("members", []) if m.get("fund_code") == result.fund_code), None)
    bundle = member.get("bundle", {}) if member else {}
    from app.services.fund_rating_reviewed_quality import BASES

    if bundle.get("basis") not in BASES:
        return output
    nodes = bundle.get("quality_metrics", {}).get("observations", {})
    dates = output.get("dataDates", {})
    report_date = dates.get("holdings")
    payment = nodes.get("known_payment", {})
    cash = _number(payment.get("cash_principal"))
    liabilities = _number(payment.get("all_book_liabilities"))
    if payment.get("kind") == "payment" and cash is not None and liabilities is not None and report_date:
        liquidity = next((d for d in dimensions if d["key"] == "liquidity"), None)
        if liquidity and cash >= 0 and liabilities >= 0:
            new_metrics = [
                _money("已核对现金本金", cash, report_date),
                _money("账面负债", liabilities, report_date),
            ]
            if liabilities > 0:
                new_metrics.append({
                    "name": "现金对账面负债覆盖", "value": f"{cash / liabilities:.4f}",
                    "unit": "倍", "period": report_date,
                })
            names = {m["name"] for m in liquidity["metrics"]}
            liquidity["metrics"].extend(m for m in new_metrics if m["name"] not in names)
            note = "这是报告日的静态账面覆盖，不代表未来大额赎回一定安全。"
            # 已发布说明已有同义限制时不重复追加，保持详情文字简洁。
            existing_warning = re.search(
                r"(?:不保证|不代表|不等于|不能|不证明)[^。；]{0,30}未来[^。；]{0,15}赎回",
                liquidity["explanation"],
            )
            if not existing_warning:
                liquidity["explanation"] += " " + note
    return output
