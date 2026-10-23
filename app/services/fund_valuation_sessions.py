"""核验休市日可用的最近收盘参考，不把参考价伪装成本日成交或基金正式估值。"""

import json
import math
import re
from datetime import date, datetime, time, timedelta
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


def hk_closed_dates(document: dict, year: int) -> set[date]:
    """仅解析港交所本年度完整休市表；缺原文、年份或覆盖时拒绝推算。"""
    body = document.get("body", "")
    if (
        document.get("kind") != "EXCHANGE_CALENDAR"
        or urlparse(document.get("url", "")).hostname != "www.hkex.com.hk"
        or f"Holiday Schedule for Year {year}" not in body
        or "Saturdays, Sundays" not in body
    ):
        return set()
    months = {
        name: i
        for i, name in enumerate(
            (
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ),
            1,
        )
    }
    closed = {
        date(int(y), months[m], int(d))
        for d, m, y in re.findall(r"(\d{1,2})\s+(" + "|".join(months) + r")\s+(\d{4})", body)
        if int(y) == year
    }
    return closed if len(closed) >= 15 else set()


def verified_hk_holiday_reference(material: dict, documents: dict, code: str, quote: dict, window: dict) -> dict | None:
    """绑定本基金条款、官方日历、汇率及真实收盘来源，验证休市日参考资料。

    只支持香港基准日休市、目标日正常开市的现有窗口。报价日期、实际涨跌和取得时间
    全部保留；返回的是资料适用性，既不产生零涨跌，也不推算持仓公允价值。
    开市但停牌、早收市、跨年资料不足及来源不匹配均留在原阻塞路径。
    """
    try:
        e = material.get("calendar_evidence") or {}
        if e.get("policy") not in {"CN_NAV_HK_CONNECT_WINDOW_V1", "QDII_CNY_VERIFIED_SESSION_V1"}:
            return None
        base, target = [date.fromisoformat(window[k]) for k in ("base_nav_date", "target_nav_date")]
        if e.get("base_date") != str(base) or e.get("target_date") != str(target) or base.year != target.year:
            return None
        calendar_ref = e.get("calendar_ref") or e.get("calendar_refs", {}).get("HK")
        closed = hk_closed_dates(documents.get(calendar_ref, {}), base.year)
        if not closed or base not in closed or target in closed or target.weekday() >= 5:
            return None
        previous = base - timedelta(days=1)
        while previous.year == base.year and (previous.weekday() >= 5 or previous in closed):
            previous -= timedelta(days=1)
        if previous.year != base.year or quote.get("date") != str(previous) or not code.endswith(".HK"):
            return None
        own = material["fund_code"]
        report = documents.get(e.get("report_ref"), {})
        prospectus = documents.get(e.get("prospectus_ref"), {})
        if (
            report.get("kind") != "REPORT"
            or own not in report.get("codes", [])
            or prospectus.get("kind") != "PROSPECTUS"
            or own not in prospectus.get("codes", [])
        ):
            return None
        clauses = e.get("holiday_valuation_quotes", {})
        if set(clauses) != {"close", "fx"} or any(
            not isinstance(q, str) or not q or q not in prospectus.get("body", "") for q in clauses.values()
        ):
            return None
        compact = re.sub(r"\s+", "", clauses["close"])
        compact_fx = re.sub(r"\s+", "", clauses["fx"])
        if not (
            re.search(r"估值日(?:无交易|没有交易)", compact)
            and "最近交易日" in compact
            and "收盘价" in compact
            and "汇率" in compact_fx
            and "中间价" in compact_fx
        ):
            return None
        fx = documents.get(e.get("fx_ref"), {})
        if fx.get("kind") != "FX_REFERENCE_DATA" or urlparse(fx.get("url", "")).hostname not in {
            "www.safe.gov.cn",
            "www.chinamoney.com.cn",
        }:
            return None
        if not any(
            r.get("date") == str(base)
            and r.get("currency") == "HKD"
            and math.isfinite(float(r["cny_per_unit"]))
            and float(r["cny_per_unit"]) > 0
            for r in json.loads(fx["body"])["rates"]
        ):
            return None
        hours = documents.get(e.get("market_hours_ref"), {})
        hours_body = hours.get("body", "")
        if (
            hours.get("kind") != "EXCHANGE_TRADING_HOURS"
            or urlparse(hours.get("url", "")).hostname != "www.hkex.com.hk"
            or not all(s in hours_body for s in ("16:08", "16:10"))
        ):
            return None
        # 收市竞价在16:08—16:10随机结束；16:10是整段交易完成的保守上界，
        # 不是本证券最后一笔成交的时间。原报价时间仍保留在source_reported_time中。
        close = datetime.combine(previous, time(16, 10), ZoneInfo("Asia/Hong_Kong"))
        source = documents.get(quote.get("source_ref"), {})
        if (
            quote.get("currency") != "HKD"
            or quote.get("valuation_market") != "HK"
            or quote.get("session") != "REGULAR_CLOSE"
            or quote.get("change_pct") is None
            or not math.isfinite(float(quote.get("close", 0)))
            or not math.isfinite(float(quote["change_pct"]))
            or float(quote.get("close", 0)) <= 0
            or datetime.fromisoformat(quote["closed_at"]) != close
            or source.get("kind") != "MARKET_DATA"
            or datetime.fromisoformat(source["available_at"]) < close
        ):
            return None
        if material.get("as_of") and datetime.fromisoformat(source["available_at"]) > datetime.fromisoformat(
            material["as_of"]
        ):
            return None
        return {
            "reference_date": str(previous),
            "valuation_date": str(base),
            "calendar_ref": calendar_ref,
            "prospectus_ref": e["prospectus_ref"],
            "fx_ref": e["fx_ref"],
            "reason": "HK_EXCHANGE_HOLIDAY_LAST_SESSION_REFERENCE",
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
