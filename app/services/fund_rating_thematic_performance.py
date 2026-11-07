"""原净值和分红编译六项表现观察；不读取派生财富路径、候选分或研究目录代码。"""

import importlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from decimal import localcontext
from statistics import median

from app.services.fund_rating_direct_index_math import performance
from app.services.material_content import content_digest

_source = importlib.import_module((__package__ + "." if __package__ else "") + "fund_rating_thematic_source")
iso, plain, require = _source.iso, _source.plain, _source.require
RETURN_PEERS = ["002450", "005585", "001714", "001628", "003053", "164205", "001223"]
STRICT_PEERS = ["002450", "001714", "001628", "003053", "001223"]
CALENDAR_SHA256 = "6d42ecc9d5a7a44a953f93d484bd6882d36397d3e7a6a59bb2bbb75b28fb00e4"


def cumulative_crosscheck(licensed, published):
    """仅接受原授权四位披露数值的一个IEEE序列化尾差，不做四舍五入。"""
    a, b = D(str(licensed)), D(str(published))
    require(a.is_finite() and b.is_finite() and a > 0 and b > 0, "ACCUM_VALUE")
    if a == b:
        return None
    require(a.as_tuple().exponent >= -4, "ACCUM_LICENSED_DISCLOSURE_PRECISION")
    delta = abs(a - b)
    bound = D.from_float(math.ulp(float(a)))
    require(delta <= bound, "NAV_SOURCE_CONFLICT")
    return {
        "licensed_accum": str(a),
        "public_accum": str(b),
        "delta": str(delta),
        "single_ieee_ulp": str(bound),
        "economic_values_rounded": False,
    }


def curve(source, sha, code):
    text = source.body(sha)
    require(re.findall(r'var\s+fS_code\s*=\s*"(\d{6})"\s*;', text) == [code], "PUBLIC_NAV_SHARE")
    arrays = {}
    for name in ["Data_netWorthTrend", "Data_ACWorthTrend"]:
        found = re.findall(r"\bvar\s+" + name + r"\s*=\s*(\[.*?\])\s*;", text, re.S)
        require(len(found) == 1, "PUBLIC_NAV_SCHEMA:" + name)
        arrays[name] = json.loads(found[0], parse_float=D)
    units, accum, notes = {}, {}, {}
    for row in arrays["Data_netWorthTrend"]:
        day = datetime.fromtimestamp(int(row["x"]) / 1000, timezone(timedelta(hours=8))).date().isoformat()
        require(day not in units, "PUBLIC_NAV_DUPLICATE")
        units[day], notes[day] = D(str(row["y"])), row.get("unitMoney")
    for stamp, value in arrays["Data_ACWorthTrend"]:
        day = datetime.fromtimestamp(int(stamp) / 1000, timezone(timedelta(hours=8))).date().isoformat()
        require(day not in accum, "PUBLIC_ACCUM_DUPLICATE")
        accum[day] = D(str(value))
    return units, accum, notes


def fund_wealth(source, spec, code, days):
    has_public = bool(spec.get("public_curve_ref"))
    if has_public:
        public, accum, notes = curve(source, spec["public_curve_ref"], code)
        require(set(days) <= public.keys() & accum.keys(), "PUBLIC_NAV_WINDOW")
    else:
        # 原005585只有完整授权API及原无分红响应；不新增“每同行必须双源”门槛。
        require(code == "005585", "PUBLIC_NAV_SOURCE_REQUIRED_FOR_PROFILE")
        public, accum, notes = {}, {}, {day: None for day in days}
    selected, announcements, serialization_audit = {}, {}, []
    for sha in spec.get("nav_refs", []):
        ref = source.original(sha)
        params = ref.get("params_without_credentials") or ref.get("params") or {}
        require(params.get("ts_code") == code + ".OF", "NAV_REQUEST_IDENTITY")
        for row in source.table(sha, symbol=code + ".OF"):
            day, ann = iso(row.get("nav_date")), iso(row.get("ann_date"))
            if day not in days:
                continue
            require(day <= source.asof and ann <= source.research, "NAV_TIME_BOUNDARY")
            point = D(str(row["unit_nav"]))
            require(day not in selected or selected[day] == point, "NAV_CONFLICTING_DUPLICATE")
            if has_public:
                require(point == public[day], "NAV_UNIT_SOURCE_CONFLICT")
                tail = cumulative_crosscheck(row["accum_nav"], accum[day])
                if tail:
                    serialization_audit.append(
                        {
                            **tail,
                            "code": code,
                            "date": day,
                            "licensed_source": sha,
                            "public_source": spec["public_curve_ref"],
                        }
                    )
            else:
                require(point == D(str(row["accum_nav"])) == D(str(row["adj_nav"])), "UNREVIEWED_CORPORATE_ACTION")
            selected[day], announcements[day] = point, ann
    if not spec.get("nav_refs"):
        require(code == "018590" and spec.get("internal_history_crosscheck_ref"), "TARGET_NAV_CROSSCHECK_REQUIRED")
        snapshot = source.json(spec["internal_history_crosscheck_ref"])
        points = {iso(row["nav_date"]): row for row in snapshot["rows"]}
        require(set(days) <= points.keys(), "TARGET_INTERNAL_WINDOW")
        for day in days:
            row = points[day]
            require(row["source_code"] == "TUSHARE_PRO_FUND" and row.get("content_hash"), "TARGET_PROVIDER_PROVENANCE")
            require(
                D(row["unit_nav"]) == public[day] == D(row["accumulated_nav"]) == D(row["adjusted_nav"]) == accum[day],
                "TARGET_NAV_SOURCE_OR_ACTION_CONFLICT",
            )
            require(not notes[day], "TARGET_UNREVIEWED_CORPORATE_ACTION")
            selected[day] = public[day]
        # 公共曲线未提供原公告日，保留未知；实际取得日只证明研究时已可观测。
        acquired = source.sources[spec["public_curve_ref"]].get("acquired_at")
        require(
            acquired and datetime.fromisoformat(acquired).date().isoformat() <= source.research,
            "TARGET_PRICE_KNOWN_BY_RESEARCH",
        )
    require(set(selected) == set(days), "NAV_COMPLETE_CALENDAR")
    events, records = {}, {}
    for sha in spec.get("dividend_refs", []):
        ref = source.original(sha)
        params = ref.get("params_without_credentials") or ref.get("params") or {}
        require(params.get("ts_code") == code + ".OF", "DIVIDEND_REQUEST_IDENTITY")
        for row in source.table(sha, symbol=code + ".OF"):
            require(iso(row.get("ann_date")) <= source.asof, "DIVIDEND_PUBLICATION_CUTOFF")
            status = row.get("div_proc")
            require(status in {"实施", "预案", "股东大会通过", "取消"}, "DIVIDEND_UNKNOWN_STATUS")
            if status != "实施":
                continue
            day = iso(row.get("ex_date"))
            if not days[0] < day <= days[-1]:
                continue
            require(day in selected, "DIVIDEND_NOT_ON_TRUE_CALENDAR")
            cash = D(str(row["div_cash"]))
            require(cash.is_finite() and cash > 0, "DIVIDEND_CASH")
            record = {
                key: iso(row[key]) if row.get(key) else None for key in ["record_date", "pay_date", "imp_anndate"]
            }
            signature = (cash, tuple(sorted(record.items())))
            require(day not in records or records[day] == signature, "DIVIDEND_SAME_DATE_CONFLICT")
            records[day], events[day] = signature, cash
    require(code == "018590" or spec.get("dividend_refs"), "DIVIDEND_ABSENCE_ORIGINAL_REQUIRED")
    if "2023-12-27" in events and code == "002450":
        sha = spec["exact_A_dividend_report_ref"]
        source.original(sha)
        text = plain(source.body(sha))
        start = text.rfind("7.4.11利润分配情况")
        stop = text.find("7.4.12", start)
        section = text[start:stop]
        a = section.find("平安睿享文娱混合A")
        b = section.find("平安睿享文娱混合C", a + 1)
        asection = section[a:b] if b > a else section[a:]
        require(
            a >= 0
            and "3.9662" in asection
            and "月27日" in asection
            and abs(events["2023-12-27"] - D(".39662")) <= D(".00005"),
            "EXACT_A_DIVIDEND_ORIGINAL",
        )
        events["2023-12-27"] = D("3.9662") / 10
    require({day for day in days if notes[day]} == set(events), "PUBLIC_ACTION_NOTE_CONFLICT")
    wealth, factor = [], D(1)
    for i, day in enumerate(days):
        unit = selected[day]
        require(unit.is_finite() and unit > 0, "NAV_VALUE")
        if day in events:
            require(i > 0 and selected[days[i - 1]] > events[day], "STANDARD_GROWTH_PREVIOUS_NAV")
            factor *= selected[days[i - 1]] / (selected[days[i - 1]] - events[day])
        wealth.append(unit / selected[days[0]] * factor)
    return wealth, {
        "events": events,
        "announcements": announcements,
        "serialization_tail_audit": serialization_audit,
        "public_ann_dates_not_fabricated": not bool(announcements),
    }


def compile_performance(inputs, source):
    spec = inputs["performance"]
    days, months = spec["dates"], spec["month_ends"]
    calendar = []
    for digest in inputs["calendar_refs"]:
        for row in source.provider_rows(digest, ["exchange", "cal_date", "is_open"]):
            require(row["exchange"] == "SSE" and row["is_open"] == 1, "CALENDAR_ACTUAL_OPEN_EXCHANGE")
            calendar.append(iso(row["cal_date"]))
    require(len(calendar) == len(set(calendar)) and sorted(calendar) == days, "CALENDAR_ORIGINAL_COMPLETE_WINDOW")
    require(spec["return_peers"] == RETURN_PEERS and spec["strict_peers"] == STRICT_PEERS, "FROZEN_PEER_SET")
    require(
        len(days) == 727 and len(months) == 37 and days == sorted(set(days)) and months == sorted(set(months)),
        "FROZEN_CALENDAR",
    )
    require(days[0] == months[0] == "2023-09-28" and days[-1] == months[-1] == "2026-09-30", "FROZEN_WINDOW")
    require(
        content_digest(json.dumps(days, separators=(",", ":")).encode()) == spec["calendar_sha256"] == CALENDAR_SHA256,
        "CALENDAR_HASH",
    )
    ids = [int(day[:4]) * 12 + int(day[5:7]) for day in months]
    require(ids == list(range(ids[0], ids[0] + 37)), "TRUE_MONTH_SEQUENCE")
    require(all(day == max(d for d in days if d[:7] == day[:7]) for day in months), "TRUE_MONTH_ENDS")
    wealth, audits, metrics, returns = {}, {}, {}, {}
    with localcontext() as ctx:
        ctx.prec = 70
        for code in ["018590"] + RETURN_PEERS:
            wealth[code], audits[code] = fund_wealth(source, spec["funds"][code], code, days)
            metrics[code] = performance(days, months, wealth[code], [D(1)] * len(days))
            returns[code] = [right / left - 1 for left, right in zip(wealth[code], wealth[code][1:], strict=False)]
        risks, benchmark_audit = {}, {}
        for code in ["018590"] + STRICT_PEERS:
            peers = [p for p in STRICT_PEERS if p != code]
            require(len(peers) >= 2 and "018590" not in peers and "003397" not in peers, "RISK_REFERENCE_EXCLUSION")
            benchmark = [D(1)]
            for i in range(len(days) - 1):
                benchmark.append(benchmark[-1] * (1 + sum(returns[p][i] for p in peers) / len(peers)))
            risks[code] = performance(days, months, wealth[code], benchmark)
            benchmark_audit[code] = peers
        nodes = {}
        for name, metric in [
            ("return_1y", "net_return_1y"),
            ("return_3y", "net_annual_return_3y"),
            ("extra_drawdown", "extra_drawdown"),
            ("extra_downside", "extra_downside"),
        ]:
            peers = RETURN_PEERS if name.startswith("return_") else STRICT_PEERS
            rows = metrics if name.startswith("return_") else risks
            nodes[name] = {
                "kind": "rank",
                "direction": "higher" if name.startswith("return_") else "lower",
                "target": rows["018590"][metric],
                "expected_products": ["CN_PUBLIC_FUND:" + p for p in peers],
                "peers": [{"product_id": "CN_PUBLIC_FUND:" + p, "value": rows[p][metric]} for p in peers],
            }
        for name, step in [("monthly_success", 1), ("annual_success", 12)]:
            rows = []
            for i in range(step, 37):
                start, end = months[i - step], months[i]
                refs = [wealth[p][days.index(end)] / wealth[p][days.index(start)] - 1 for p in STRICT_PEERS]
                rows.append(
                    {
                        "start": start,
                        "end": end,
                        "target": str(wealth["018590"][days.index(end)] / wealth["018590"][days.index(start)] - 1),
                        "reference": str(median(refs)),
                    }
                )
            nodes[name] = {"kind": "windows", "expected_windows": len(rows), "rows": rows}
    return nodes, {
        "wealth": wealth,
        "funds": audits,
        "risk_reference_products": benchmark_audit,
        "window_days": len(days),
    }
