"""业绩变化资料准备 V1：只处理公开事实，不训练、不读取方向标签。

版本链按公司、会计期间和指标口径匹配。区间变化是公司自身披露变化，
不能称为相对市场一致预期的“超预期”，也不输出基金涨跌方向。
"""

import re
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal

from app.services.fund_information_history_v1 import explicit_money, normalize


def earnings_kind(title):
    """目录类型与修订标记分开保存；报告规则、问询回复不冒充财报。"""
    if any(
        word in title
        for word in (
            "工作规程",
            "工作制度",
            "审议",
            "问询",
            "回复",
            "核查",
            "审核意见",
            "法律意见",
            "独立董事",
            "监事会",
            "提示性",
            "取消",
            "延期",
            "预约",
        )
    ):
        return None
    if "业绩预告" in title:
        return "FORECAST"
    if "业绩快报" in title:
        return "PRELIMINARY_RESULT"
    if re.search(r"(?:年度|季度|半年度)报告", title):
        if any(word in title for word in ("更正公告", "补充公告", "修订公告")):
            return "CORRECTION_NOTICE"
        return "REPORTED_RESULT"
    return None


def plan_groups(original_rows, previous_rows, max_groups=4, max_windows=20):
    """按整组记录的补齐效率排序，每基金最多一组；不读取任何标签。

    成本代理为待补公司窗口数，实际请求还受分页和正文数量影响。冻结后
    不根据取得的训练结果调整选择。日期只使用既有训练记录的窗口。
    """
    prior = {(r["fund_code"], r["target"]): r for r in previous_rows}
    if len(prior) != len(previous_rows) or len(prior) != len(original_rows):
        raise ValueError("ROW_KEYS_NOT_UNIQUE_OR_COUNT_CHANGED")
    groups = defaultdict(list)
    for row in original_rows:
        current = prior[(row["fund_code"], row["target"])]
        if not current["catalog_complete"]:
            if not current["remaining_companies"]:
                raise ValueError("INCOMPLETE_ROW_WITHOUT_COMPANY_GAP")
            groups[(row["fund_code"], row["report_sha256"])].append(current)
    ranked = []
    for (fund, report), rows in groups.items():
        stocks = sorted({s for r in rows for s in r["remaining_companies"]})
        ranked.append(
            {
                "fund_code": fund,
                "report_sha256": report,
                "rows": len(rows),
                "windows": len(stocks),
                "score": len(rows) / len(stocks),
                "members": rows,
                "stocks": stocks,
            }
        )
    ranked.sort(key=lambda r: (-r["score"], r["fund_code"], r["report_sha256"]))
    selected, windows, funds = [], [], set()
    for item in ranked:
        if len(selected) >= max_groups or item["fund_code"] in funds or len(windows) + item["windows"] > max_windows:
            continue
        selected.append({k: v for k, v in item.items() if k not in {"members", "stocks"}})
        funds.add(item["fund_code"])
        for stock in item["stocks"]:
            dates = [r["target"] for r in item["members"] if stock in r["remaining_companies"]]
            windows.append(
                {
                    "stock": stock,
                    "start": str(date.fromisoformat(min(dates)) - timedelta(days=30)),
                    "end": str(date.fromisoformat(max(dates)) - timedelta(days=1)),
                    "fund_code": item["fund_code"],
                    "report_sha256": item["report_sha256"],
                    "purpose": "TRAINING_CATALOG_GAP",
                    "target_records": len(dates),
                }
            )
    return {
        "selected": selected,
        "windows": windows,
        "ranking": [{k: v for k, v in r.items() if k not in {"members", "stocks"}} for r in ranked],
    }


def review_claim(spec, pages):
    """重放人工指定的原件字段及口径，保留正负、区间、原单位和页锚点。

    原始金额符号必须出现在引用中，不能由“盈利/亏损”之外的推断补出。
    缺日期、单位或会计口径就拒绝，不把未知编码为零。
    """
    required = (
        "issuer",
        "period_start",
        "period_end",
        "metric",
        "basis",
        "currency",
        "unit",
        "values",
        "kind",
        "published_date",
        "document_id",
        "page",
        "quote",
    )
    if any(spec.get(k) is None or spec.get(k) == "" for k in required):
        raise ValueError("INCOMPLETE_CLAIM")
    start, end = date.fromisoformat(spec["period_start"]), date.fromisoformat(spec["period_end"])
    if start > end or spec["currency"] != "CNY":
        raise ValueError("INVALID_PERIOD_OR_CURRENCY")
    if spec["kind"] not in {"FORECAST", "PRELIMINARY_RESULT", "REPORTED_RESULT"}:
        raise ValueError("INVALID_CLAIM_KIND")
    values = spec["values"]
    if len(values) not in {1, 2} or (spec["kind"] != "FORECAST" and len(values) != 1):
        raise ValueError("INVALID_VALUE_CARDINALITY")
    page_no = spec["page"]
    if not 1 <= page_no <= len(pages):
        raise ValueError("INVALID_ANCHOR_PAGE")
    text, quote = normalize(pages[page_no - 1]), normalize(spec["quote"])
    if not quote or text.count(quote) != 1:
        raise ValueError("CLAIM_ANCHOR_NOT_UNIQUE")
    # 金额边界保留引文中的空白；否则相邻表格单元格会被拼成一个数字。
    cleaned = spec["quote"].replace(",", "").replace("，", "").replace("−", "-")
    # 公告中双连字符和长横线用于区间分隔，不是负号；单个负号保持不变。
    cleaned = cleaned.replace("--", "~").replace("–", "~").replace("—", "~")
    for value in values:
        literal = str(value).replace(",", "").replace("，", "")
        # 数字边界防止把 12 误配进 112，也不允许把负数悄悄丢掉负号。
        if not re.search(r"(?<![\d.\-])" + re.escape(literal) + r"(?![\d.])", cleaned):
            raise ValueError("CLAIM_VALUE_NOT_IN_ANCHOR")
    if spec["unit"] not in quote:
        raise ValueError("CLAIM_UNIT_NOT_IN_ANCHOR")
    money = [explicit_money(v, spec["unit"]) for v in values]
    ordered = [Decimal(m["cny"]) for m in money]
    if ordered != sorted(ordered):
        raise ValueError("REVERSED_FORECAST_RANGE")
    published = date.fromisoformat(spec["published_date"])
    return {
        **spec,
        "money": money,
        "available_at": str(published + timedelta(days=1)) + "T08:00:00+08:00",
        "anchor": {"page": page_no, "offset": text.index(quote), "text": quote},
        "training_ready": False,
    }


def compare_claims(older, newer, as_of):
    """同公司同期间同口径的绝对金额变化；零/负基数不计算增长率。

    as_of 是带时区的预测截止时刻。只使用当时已经可用的两份独立披露；
    修正公告里追述的旧值不能倒灌成更早时点的事实。
    """
    from datetime import datetime

    keys = ("issuer", "period_start", "period_end", "metric", "basis", "currency")
    if any(older[k] != newer[k] for k in keys):
        raise ValueError("INCOMPARABLE_EARNINGS_CLAIMS")
    cutoff = datetime.fromisoformat(as_of)
    times = [datetime.fromisoformat(c["available_at"]) for c in (older, newer)]
    if cutoff.tzinfo is None or any(t.tzinfo is None for t in times):
        raise ValueError("AS_OF_TIMEZONE_REQUIRED")
    if times[0] >= times[1] or times[1] > cutoff:
        raise ValueError("CLAIM_NOT_AVAILABLE_IN_ORDER")
    if older["document_id"] == newer["document_id"]:
        raise ValueError("INDEPENDENT_DISCLOSURES_REQUIRED")
    if any(c.get("revision_issues") or not c.get("source_identity_verified") for c in (older, newer)):
        raise ValueError("SOURCE_REVIEW_NOT_PASSED")
    a = [Decimal(m["cny"]) for m in older["money"]]
    b = [Decimal(m["cny"]) for m in newer["money"]]
    if min(b) > max(a):
        relation = "ABOVE_PREVIOUS_RANGE"
    elif max(b) < min(a):
        relation = "BELOW_PREVIOUS_RANGE"
    else:
        relation = "OVERLAPS_PREVIOUS_RANGE"
    return {
        "older": older["document_id"],
        "newer": newer["document_id"],
        "comparison_kind": older["kind"] + "_TO_" + newer["kind"],
        "relation": relation,
        "currency": "CNY",
        "lower_change_cny": str(min(b) - min(a)),
        "upper_change_cny": str(max(b) - max(a)),
        "midpoint_change_cny": str((min(b) + max(b) - min(a) - max(a)) / 2),
        "percent_change": None,
        "as_of": as_of,
        "is_market_consensus_surprise": False,
        "training_ready": False,
    }
