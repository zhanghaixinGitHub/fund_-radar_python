"""160323收益、风险原件编译候选：不联网、不写库、不导入研究目录。

只读显式root/sources并逐次校字节；固定实审摘要约束准入和事件三态，
不接收预计算分值或日因子。完整基金分与等级始终留空。
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, OrderedDict
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path
from threading import RLock

from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS
from app.services.material_content import content_digest, source_bytes, source_exists, source_size
from app.services.material_content import document_reader as PdfReader

VERSION = "R5_MIXED_160323_PERFORMANCE_ORIGINALS_20261030_2"
ADMISSION_REVIEW_SHA = "bba2065a4d68b238c6cd8d904bc82829fc1aaa4d7f36bd03caf25045a40f5096"
MATERIAL_REVIEW_SHA = "216a501be072374c6b693e707ed72365f81c7b98a208efa7d58d353bf917d0f4"
COHORT_AUDIT_SHA = "8c2a16581984f7407434e6b500570e6e75cd7440a77ad4415891a965f6d7de78"
PRIMARY_SHA = "bae39540e2b86c2034a7678aaa783b679ca6e0a88885984c174244b6e5952e9e"
TIME_CONTRACT = {
    "kind": "RETROSPECTIVE_HISTORICAL_QUALITY_RESEARCH",
    "valuation_window_end": "as_of_date",
    "nav_and_market_publication_boundary": "research_on",
    "fundamental_disclosure_boundary": "as_of_date",
    "not_historical_asof_tradable_signal": True,
}
NAV_FIELDS = ["ts_code", "ann_date", "nav_date", "unit_nav", "accum_nav", "adj_nav"]
DIV_FIELDS = [
    "ts_code",
    "ann_date",
    "imp_anndate",
    "ex_date",
    "net_ex_date",
    "div_proc",
    "div_cash",
    "base_unit",
    "record_date",
    "pay_date",
]


def _require(value, code):
    if not value:
        raise ValueError(code)


def _iso(value):
    _require(isinstance(value, str) and re.fullmatch(r"(?:\d{8}|\d{4}-\d{2}-\d{2})", value), "MIXED_DATE_SCHEMA")
    return date.fromisoformat(value).isoformat()


def _json(source, digest):
    _require(digest in source.raw, "MIXED_UNARCHIVED_SOURCE")
    return json.loads(source.raw[digest], parse_float=D)


def _originals(inputs, root):
    """旧绝对路径只允许作为嵌套追溯字符串；运行时文件路径必须相对sources。"""
    _require(
        inputs.get("version") == VERSION and inputs.get("time_contract") == TIME_CONTRACT,
        "MIXED_EXPLICIT_VERSION_TIME_CONTRACT",
    )
    _require(
        inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("holding_date") == "2026-06-30"
        and _iso(inputs.get("research_on")) == "2026-10-30",
        "MIXED_FIXED_PERIOD",
    )
    for digest, ref in inputs["sources"].items():
        file = Path(ref["file"])
        _require(
            not file.is_absolute() and file.parts and file.parts[0] == "sources" and ".." not in file.parts,
            "MIXED_SOURCE_RELATIVE_SCOPE",
        )
        if ref.get("path"):
            _require(
                not Path(ref["path"]).is_absolute() and ref["path"] == ref["file"],
                "MIXED_NO_ARCHIVED_ABSOLUTE_RUNTIME_PATH",
            )
        _require(digest == ref.get("sha256"), "MIXED_SOURCE_KEY_HASH")
    checked = deepcopy(inputs)
    checked["as_of_date"] = inputs["research_on"]
    for ref in checked["sources"].values():
        if ref.get("published_on"):
            ref["published_on"] = _iso(ref["published_on"])
    return Originals(checked, Path(root))


def _admission(inputs, source):
    """固定人工实审manifest绑定产品、源和两类准入，不能由调用方改写。"""
    _require(
        ADMISSION_REVIEW_SHA and inputs.get("admission_review_ref") == ADMISSION_REVIEW_SHA,
        "MIXED_FROZEN_ADMISSION_REVIEW_REQUIRED",
    )
    review = _json(source, ADMISSION_REVIEW_SHA)
    _require(
        review.get("version") == "160323-MIXED-ADMISSION-ORIGINAL-BINDING-V2"
        and review.get("fund_code") == "160323"
        and review.get("as_of_date") == inputs["as_of_date"]
        and review.get("review_closed") is True,
        "MIXED_ADMISSION_REVIEW_SCOPE",
    )
    _require(
        inputs["target"] == review["target"]
        and inputs["peers"] == review["peers"]
        and inputs["calendar_sources"] == review["calendar_sources"]
        and inputs["trade_calendar"] == review["trade_calendar"],
        "MIXED_FIXED_PRODUCT_SOURCE_BINDING",
    )
    codes = [p["code"] for p in review["peers"]]
    masters = [p["master_product_id"] for p in review["peers"]]
    expected_return = review.get("expected_return_product_count")
    expected_risk = review.get("expected_risk_product_count")
    _require(
        type(expected_return) is int
        and expected_return > 0
        and type(expected_risk) is int
        and expected_risk > 1
        and len(codes) == len(set(codes)) == len(set(masters)) == expected_return
        and not set(codes) & {"160323", "013360"}
        and len(review["risk_codes"]) == expected_risk
        and set(review["risk_codes"]) <= set(codes),
        "MIXED_FIXED_REFERENCE_PRODUCTS",
    )
    for digest, binding in review["source_bindings"].items():
        _require(digest in source.raw, "MIXED_ADMISSION_ORIGINAL_MISSING")
        ref = inputs["sources"][digest]
        _require(ref.get("binding") == binding, "MIXED_SOURCE_PROVENANCE_BINDING")
        if binding.get("published_on"):
            boundary = inputs["research_on"] if binding.get("price_response") else inputs["as_of_date"]
            _require(_iso(binding["published_on"]) <= boundary, "MIXED_DOCUMENT_PUBLICATION_BOUNDARY")
            _require(ref.get("published_on") == binding["published_on"], "MIXED_SOURCE_PUBLICATION_MUTATION")
    extension_sha = review.get("research_date_extension_source")
    _require(extension_sha == inputs.get("research_date_extension_ref"), "MIXED_RESEARCH_EXTENSION_BINDING")
    extension = _json(source, extension_sha)
    _require(
        extension.get("research_on") == inputs["research_on"] == "2026-10-30"
        and extension.get("as_of_date") == inputs["as_of_date"]
        and extension.get("actual_receipts_remain_unchanged") is True
        and extension.get("prior_time_decision_source_sha") == review["nav_time_decision_source"]
        and extension.get("empty_dividend_verification_source_sha") in source.raw,
        "MIXED_RESEARCH_EXTENSION_SCOPE",
    )
    return review


def _calendar(inputs, source):
    """由原官方休市定义重新生成日历，不从净值缺行反推交易日。"""
    sessions = []
    for digest in inputs["calendar_sources"]:
        definition = _json(source, digest)
        _require(
            definition["construction"] == "WEEKDAYS_MINUS_OFFICIAL_HOLIDAYS"
            and definition["market"] == "CN_A_SHARE_SSE_SZSE",
            "MIXED_REAL_CALENDAR_DEFINITION",
        )
        closed = set()
        for notice in definition["years"]:
            _require(
                _iso(notice["announced_on"]) < str(notice["year"]) + "-01-01" and len(notice["sources"]) == 2,
                "MIXED_CALENDAR_NOTICE_SOURCE",
            )
            for first, last in notice["closed_ranges"]:
                a, b = date.fromisoformat(first), date.fromisoformat(last)
                closed.update(a + timedelta(days=i) for i in range((b - a).days + 1))
        a, b = date.fromisoformat(definition["coverage_start"]), date.fromisoformat(definition["coverage_end"])
        sessions.extend(
            str(a + timedelta(days=i))
            for i in range((b - a).days + 1)
            if (a + timedelta(days=i)).weekday() < 5 and a + timedelta(days=i) not in closed
        )
    days = [d for d in sessions if "2023-09-28" <= d <= inputs["as_of_date"]]
    _require(
        days == sorted(set(days)) and len(days) == 727 and days[0] == "2023-09-28" and days[-1] == "2026-09-30",
        "MIXED_REAL_727_CALENDAR",
    )
    independent = inputs["trade_calendar"]
    proof = _json(source, independent["receipt_source_sha"])
    receipts = [r for r in proof["receipts"] if r.get("sha256") == independent["raw_source_sha"]]
    _require(len(receipts) == 1, "MIXED_INDEPENDENT_CALENDAR_RECEIPT")
    request = receipts[0]
    _require(
        request.get("api") == "trade_cal"
        and request.get("params_without_credentials")
        == {"exchange": "SSE", "start_date": "20230928", "end_date": "20260930", "is_open": "1"},
        "MIXED_INDEPENDENT_CALENDAR_REQUEST",
    )
    rows = _table(source, independent["raw_source_sha"], ["exchange", "cal_date", "is_open", "pretrade_date"])
    _require(
        all(r["exchange"] == "SSE" and type(r["is_open"]) in {int, str} and str(r["is_open"]) == "1" for r in rows),
        "MIXED_INDEPENDENT_CALENDAR_MARKET",
    )
    actual = sorted(_iso(r["cal_date"]) for r in rows)
    _require(actual == days, "MIXED_INDEPENDENT_CALENDAR_DAY_MISMATCH")
    by_day = {_iso(r["cal_date"]): _iso(r["pretrade_date"]) for r in rows}
    _require(
        all(by_day[current] == previous for previous, current in zip(days[:-1], days[1:], strict=True)),
        "MIXED_INDEPENDENT_CALENDAR_PREVIOUS_DAY",
    )
    return days


def _table(source, digest, fields):
    raw = _json(source, digest)
    _require(
        type(raw.get("code")) is int and raw["code"] == 0 and isinstance(raw.get("data"), dict),
        "MIXED_PROVIDER_SUCCESS_RESPONSE_REQUIRED",
    )
    data = raw["data"]
    _require(
        data.get("has_more") is False and data.get("fields") == fields and isinstance(data.get("items"), list),
        "MIXED_PROVIDER_COMPLETE_FIELDS",
    )
    count = data.get("count")
    _require(
        type(count) is int and count >= 0 and (count == 0 or count == len(data["items"])), "MIXED_PROVIDER_PAGE_COUNT"
    )
    rows = []
    for row in data["items"]:
        _require(isinstance(row, list) and len(row) == len(fields), "MIXED_PROVIDER_ROW_SCHEMA")
        rows.append(dict(zip(fields, row, strict=True)))
    return rows


def _js(text, key):
    found = list(re.finditer(r"\bvar\s+" + re.escape(key) + r"\s*=\s*(.*?);", text, re.S))
    _require(len(found) == 1, "MIXED_PUBLIC_NAV_SCHEMA:" + key)
    return json.loads(found[0][1], parse_float=D)


def _curve(member, source, days):
    text = source.raw[member["public_nav_source"]].decode("utf-8-sig")
    _require(_js(text, "fS_code") == member["code"], "MIXED_PUBLIC_NAV_WRONG_SHARE")
    curve = {}
    for row in _js(text, "Data_netWorthTrend"):
        stamp = datetime.fromtimestamp(int(row["x"]) / 1000, timezone(timedelta(hours=8))).date().isoformat()
        _require(stamp not in curve, "MIXED_PUBLIC_NAV_DUPLICATE")
        curve[stamp] = row
    _require(set(days) <= set(curve), "MIXED_PUBLIC_NAV_MISSING_DAY")
    _require(not any("拆分" in (curve[d].get("unitMoney") or "") for d in days), "MIXED_SPLIT_REVIEW_REQUIRED")
    return curve


def _nav(member, source, inputs, days, curve):
    rows, units, late, serializations = [], {}, [], []
    for digest in member["licensed_nav_sources"]:
        rows += _table(source, digest, NAV_FIELDS)
    _require(rows, "MIXED_LICENSED_NAV_REQUIRED")
    for row in rows:
        _require(row["ts_code"] == member["ticker"], "MIXED_LICENSED_NAV_WRONG_SHARE")
        stamp, announced = _iso(row["nav_date"]), _iso(row["ann_date"])
        _require(stamp <= inputs["as_of_date"], "MIXED_NAV_FUTURE_VALUATION")
        _require(announced <= inputs["research_on"], "MIXED_NAV_ANNOUNCED_AFTER_RESEARCH")
        unit = D(str(row["unit_nav"]))
        _require(unit.is_finite() and unit > 0 and stamp not in units, "MIXED_NAV_INVALID_OR_DUPLICATE")
        units[stamp] = unit
        if stamp in days and announced > inputs["as_of_date"]:
            late.append({"nav_date": stamp, "ann_date": announced, "known_at_nominal_cutoff": False})
    _require(set(days) <= set(units), "MIXED_LICENSED_NAV_MISSING_DAY")
    for day in days:
        licensed, public = units[day], D(str(curve[day]["y"]))
        _require(public.is_finite() and public > 0, "MIXED_PUBLIC_NAV_INVALID")
        if licensed == public:
            continue
        # 只容许真实4位披露值的一个二进制浮点ULP尾差，不舍入或替换任何净值。
        with localcontext() as precision:
            precision.prec = 80
            delta = abs(licensed - public)
        _require(
            licensed == licensed.quantize(D(".0001")) and math.isfinite(float(licensed)), "MIXED_DUAL_NAV_CONFLICT"
        )
        ulp = D.from_float(math.ulp(float(licensed)))
        _require(delta <= ulp, "MIXED_DUAL_NAV_CONFLICT")
        serializations.append(
            {
                "nav_date": day,
                "licensed_original": str(licensed),
                "public_original": str(public),
                "absolute_difference": str(delta),
                "single_float_ulp": str(ulp),
                "values_rounded_or_replaced": False,
            }
        )
    return units, late, serializations


def _public_dividends(member, source):
    if not member.get("public_dividend_source"):
        return []
    soup = BeautifulSoup(source.raw[member["public_dividend_source"]].decode("utf-8-sig"), "html.parser")
    tables = [
        t
        for t in soup.find_all("table")
        if [h.get_text(strip=True) for h in t.find_all("th")]
        == ["年份", "权益登记日", "除息日", "每10份分红", "分红发放日"]
    ]
    _require(len(tables) == 1, "MIXED_PUBLIC_DIVIDEND_TABLE_SCHEMA")
    rows = []
    for tr in tables[0].find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
        if cells:
            _require(len(cells) == 5, "MIXED_PUBLIC_DIVIDEND_ROW_SCHEMA")
            match = re.fullmatch(r"每10份派现金([0-9.]+)元", cells[3])
            _require(match, "MIXED_PUBLIC_DIVIDEND_AMOUNT")
            rows.append(
                {"record": _iso(cells[1]), "effective": _iso(cells[2]), "cash": D(match[1]) / 10, "pay": _iso(cells[4])}
            )
    return rows


def _dividends(member, source, inputs, days, curve, review):
    """成功完整空集与失败、错误份额标识严格分开；同经济事件按Decimal去重。"""
    _require(member["fund_div_sources"], "MIXED_COMPLETE_FUND_DIV_REQUIRED")
    rows = []
    for digest in member["fund_div_sources"]:
        part = _table(source, digest, DIV_FIELDS)
        if not part:
            proof_sha = member.get("fund_div_request_proofs", {}).get(digest)
            _require(proof_sha in source.raw, "MIXED_EMPTY_DIVIDEND_ACTUAL_REQUEST_PROOF_REQUIRED")
            proof = _json(source, proof_sha)
            _require(
                proof.get("proof_origin") == "ACTUAL_HTTP_REQUEST_BODY_WITH_TOKEN_REMOVED"
                and proof.get("credential_archived") is False
                and proof.get("api_name") == "fund_div"
                and proof.get("params") == {"ts_code": member["ticker"]}
                and proof.get("fields") == ",".join(DIV_FIELDS)
                and proof.get("response_sha256") == digest
                and proof.get("method") == "POST"
                and proof.get("http_status") == 200,
                "MIXED_EMPTY_DIVIDEND_REQUEST_SHARE_BINDING",
            )
        rows += part
    events = {}
    for row in rows:
        _require(row["ts_code"] == member["ticker"], "MIXED_DIVIDEND_WRONG_SHARE")
        _require(row.get("div_proc") in {"实施", "预案", "股东大会通过", "终止"}, "MIXED_DIVIDEND_UNKNOWN_STATUS")
        dates = {_iso(row[k]) for k in ["ex_date", "net_ex_date"] if row.get(k)}
        if not dates:
            _require(row.get("div_proc") != "实施", "MIXED_IMPLEMENTED_DIVIDEND_DATE_MISSING")
            continue
        if max(dates) <= days[0] or min(dates) > days[-1]:
            continue
        _require(row.get("div_proc") == "实施", "MIXED_DIVIDEND_STATUS_OR_ANNOUNCEMENT")
        _require(row.get("ann_date") and row.get("imp_anndate"), "MIXED_IMPLEMENTATION_ANNOUNCEMENT_DATES_REQUIRED")
        _require(
            all(_iso(row[key]) <= inputs["as_of_date"] for key in ["ann_date", "imp_anndate"]),
            "MIXED_DIVIDEND_STATUS_OR_ANNOUNCEMENT",
        )
        _require(
            row.get("div_cash") is not None and not isinstance(row["div_cash"], bool), "MIXED_DIVIDEND_CASH_MISSING"
        )
        cash = D(str(row["div_cash"]))
        if cash == 0 and member["code"] in review.get("dividend_zero_exceptions", {}):
            exception = review["dividend_zero_exceptions"][member["code"]]
            _require(
                dates == {exception["date"]}
                and set(member["fund_div_sources"]) == set(exception["fund_div_sources"])
                and exception["primary_report_source"] in source.raw,
                "MIXED_ZERO_DIVIDEND_EXCEPTION_SCOPE",
            )
            source.claim({"source_hash": exception["primary_report_source"], "quote": exception["quote"]})
            continue
        _require(cash.is_finite() and cash > 0, "MIXED_DIVIDEND_CASH_VALUE")
        key = (_iso(row.get("record_date")), _iso(row.get("pay_date")))
        if key in events:
            _require(events[key]["cash"] == cash, "MIXED_DIVIDEND_CONFLICT")
            events[key]["dates"].update(dates)
        else:
            events[key] = {"cash": cash, "dates": dates}
    public, applied = _public_dividends(member, source), {}
    for (record, pay), event in events.items():
        matches = [r for r in public if r["record"] == record and r["pay"] == pay and r["cash"] == event["cash"]]
        _require(len(matches) == 1, "MIXED_DIVIDEND_PUBLIC_CROSSCHECK_REQUIRED")
        day = matches[0]["effective"]
        _require(day in days and day in event["dates"] and day not in applied, "MIXED_DIVIDEND_EFFECTIVE_DATE_CONFLICT")
        applied[day] = event["cash"]
    notes = {d for d in days if days[0] < d and curve[d].get("unitMoney")}
    _require(
        notes == set(applied) and all("分红" in curve[d]["unitMoney"] for d in notes), "MIXED_DIVIDEND_CURVE_COVERAGE"
    )
    return applied


def _risk(factors, reference):
    wealth = base = peak = D(1)
    drawdown = negative = D(0)
    _require(len(factors) == len(reference) == 726, "MIXED_RISK_DAILY_CALENDAR")
    for f, b in zip(factors, reference, strict=True):
        _require(f > 0 and b > 0, "MIXED_RISK_GROWTH_POSITIVE")
        wealth *= f
        base *= b
        relative = wealth / base
        peak = max(peak, relative)
        drawdown = max(drawdown, 1 - relative / peak)
        negative += min(f - b, D(0)) ** 2
    return {"extra_drawdown": str(drawdown), "extra_downside": str((D(252) / len(factors) * negative).sqrt())}


def _growth(days, units, cash, start, annual=False):
    """单位净值增长含法定现金调整；保持调用处75位Decimal上下文，不做现金再投实验。"""
    result = units[days[-1]] / units[start]
    for ex, amount in cash.items():
        if start < ex <= days[-1]:
            previous = days[days.index(ex) - 1]
            result *= units[previous] / (units[previous] - amount)
    if annual:
        result = result ** (D("365.25") / D((date.fromisoformat(days[-1]) - date.fromisoformat(start)).days))
    return str(result - 1)


def _compile(inputs, source, review):
    days = _calendar(inputs, source)
    daily, returns, audit = {}, {}, []
    with localcontext() as ctx:
        ctx.prec = 75
        for member in [inputs["target"]] + inputs["peers"]:
            curve = _curve(member, source, days)
            units, late, serializations = _nav(member, source, inputs, days, curve)
            cash = _dividends(member, source, inputs, days, curve, review)
            _require(
                all(units[days[days.index(ex) - 1]] > amount for ex, amount in cash.items()),
                "MIXED_DIVIDEND_EXCEEDS_PREVIOUS_NAV",
            )
            factors = [units[d] / (units[p] - cash.get(d, D(0))) for p, d in zip(days[:-1], days[1:], strict=True)]
            _require(all(f > 0 and f.is_finite() for f in factors), "MIXED_CASH_EXCEEDS_PREVIOUS_NAV")
            daily[member["code"]] = factors

            returns[member["code"]] = {
                "return_1y": _growth(days, units, cash, "2025-09-30"),
                "return_3y": _growth(days, units, cash, days[0], True),
            }
            audit.append(
                {
                    "code": member["code"],
                    "days": len(days),
                    "nav_announced_after_valuation_cutoff": late,
                    "dual_source_serialization_only": serializations,
                    "cash_events": {d: str(c) for d, c in cash.items()},
                }
            )
        codes = review["risk_codes"]
        sums = [sum(daily[c][i] for c in codes) for i in range(726)]
        target = _risk(daily["160323"], [s / len(codes) for s in sums])
        peers_risk = {
            c: _risk(daily[c], [(sums[i] - daily[c][i]) / (len(codes) - 1) for i in range(726)]) for c in codes
        }
    nodes = {}
    for name in ["return_1y", "return_3y", "extra_drawdown", "extra_downside"]:
        is_return = name.startswith("return")
        peers = inputs["peers"] if is_return else [m for m in inputs["peers"] if m["code"] in codes]
        nodes[name] = {
            "kind": "rank",
            "direction": "higher" if is_return else "lower",
            "target": returns["160323"][name] if is_return else target[name],
            "peers": [
                {
                    "product_id": m["master_product_id"],
                    "value": returns[m["code"]][name] if is_return else peers_risk[m["code"]][name],
                }
                for m in peers
            ],
            "expected_products": [m["master_product_id"] for m in peers],
        }
    return {
        "observations": nodes,
        "audit": {
            "nav_and_events": audit,
            "calendar_days": 727,
            "daily_steps": 726,
            "return_peers": len(inputs["peers"]),
            "risk_peers": len(codes),
            "annualization": 252,
            "growth": "PREVIOUS_TRADING_NAV_MINUS_LEGAL_CASH",
            "peer_own_product_excluded": True,
        },
    }


def _compact(value):
    """只规范Unicode/空白以比对原文，不改数额、日期或标点含义。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)))


def _decode(raw):
    from app.services.material_content import exact_source_text

    return exact_source_text(raw)


def _at_json(value, path):
    catalog = re.fullmatch(r"Data\[ID=([^\]]+)\]\.PUBLISHDATE", path)
    if catalog:
        rows = [r for r in value["Data"] if r["ID"] == catalog[1]]
        _require(len(rows) == 1, "MIXED_EVENT_PUBLICATION_CATALOG_ID")
        return rows[0]["PUBLISHDATE"]
    for key in path.split("."):
        value = value[key]
    return value


def _publication(claim, source, cutoff):
    """发布日期逐条核实际目录/正文/原官网片段；签发日不替代公开日。"""
    published = _iso(claim.get("published_on"))
    _require(published <= cutoff, "MIXED_EVENT_FUTURE_PUBLICATION")
    proof = claim["publication_date_evidence"]
    digest = proof.get("source_hash")
    _require(digest in source.raw, "MIXED_EVENT_PUBLICATION_ORIGINAL_REQUIRED")
    if proof.get("raw_json_path"):
        actual = _at_json(_json(source, digest), proof["raw_json_path"])
        _require(actual == proof.get("actual_value"), "MIXED_EVENT_PUBLICATION_VALUE_MISMATCH")
        if proof["raw_json_path"] == "url":
            # 原获取回执只证明规范URL的日期，不冒充HTTP正文或正文签发日期。
            match = re.search(r"/t(\d{8})_", actual)
            _require(
                match and _iso(match[1]) == published and proof.get("date_basis"), "MIXED_EVENT_PUBLICATION_URL_DATE"
            )
        else:
            _require(isinstance(actual, str) and actual[:10] == published, "MIXED_EVENT_PUBLICATION_DATE_MISMATCH")
    else:
        raw_html = _decode(source.raw[digest])
        first, last = proof["raw_html_chars"]
        fragment = proof.get("actual_date_fragment", proof.get("actual_date_and_article_fragment"))
        _require(type(first) is int and type(last) is int and 0 <= first < last, "MIXED_EVENT_PUBLICATION_HTML_SLICE")
        # 部分旧定位来自read_text通用换行，另一些直接decode原字节；两者逐片段精确核。
        candidates = [raw_html, raw_html.replace("\r\n", "\n")]
        matched = [h for h in candidates if last <= len(h) and h[first:last] == fragment]
        _require(matched, "MIXED_EVENT_PUBLICATION_HTML_SLICE")
        html = matched[0]
        year, month, day = published.split("-")
        date_pattern = rf"{year}(?:-|年|/|\.)(?:{month}|{int(month)})(?:-|月|/|\.)(?:{day}|{int(day)})(?:日)?"
        _require(re.search(date_pattern, fragment), "MIXED_EVENT_PUBLICATION_HTML_DATE")
        article = proof.get("attachment_or_article_id")
        if article:
            _require(str(article) in fragment, "MIXED_EVENT_PUBLICATION_ATTACHMENT")
        article = proof.get("expected_article_id")
        if article:
            # 新浪目录按“日期+链接”排列；下一条日期不能给上一条链接作证。
            matches = list(re.finditer(r"(?:[?&]id=|/id/)" + re.escape(str(article)) + r"(?!\d)", html))
            _require(
                matches and any(re.search(date_pattern, html[max(0, m.start() - 250) : m.start()]) for m in matches),
                "MIXED_EVENT_PUBLICATION_ARTICLE_DATE_BINDING",
            )


def _claim_text(claim, source, pdf_cache):
    """按定位核真实原字节；扫描/原公告JSON不会因扩展名而被当作无资料。"""
    digest = claim.get("source_hash")
    _require(digest in source.raw and len(claim.get("quote", "")) >= 12, "MIXED_EVENT_CLAIM_ORIGINAL_REQUIRED")
    location = claim["source_location"]
    raw = source.raw[digest]
    if "pdf_page" in location:
        page_number = location["pdf_page"]
        key = (digest, page_number)
        _require(
            type(page_number) is int and page_number > 0 and raw.startswith(b"%PDF"), "MIXED_EVENT_PDF_PAGE_SCHEMA"
        )
        if key not in pdf_cache:
            # 固定C审阅的引句按pypdf文本序排列；换抽取器可改变表格阅读顺序。
            # 每份PDF只建reader，按声明物理页核原字节，绝不读取派生pages副本。
            if digest not in pdf_cache:
                pdf_cache[digest] = PdfReader(raw)
            reader = pdf_cache[digest]
            _require(page_number <= len(reader.pages), "MIXED_EVENT_PDF_PAGE_RANGE")
            pdf_cache[key] = reader.pages[page_number - 1].extract_text() or ""
        text = pdf_cache[key]
    elif "json_path" in location:
        text = _at_json(_json(source, digest), location["json_path"])
        _require(isinstance(text, str), "MIXED_EVENT_NOTICE_BODY_REQUIRED")
        text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    else:
        _require(
            location.get("html_text_extractor") == 'BeautifulSoup.get_text(separator=" ",strip=True)',
            "MIXED_EVENT_HTML_EXTRACTOR",
        )
        text = BeautifulSoup(_decode(raw), "html.parser").get_text(" ", strip=True)
        first, last = location["html_text_chars"]
        _require(0 <= first < last <= len(text), "MIXED_EVENT_HTML_TEXT_RANGE")
        text = text[first:last]
    _require(_compact(claim["quote"]) in _compact(text), "MIXED_EVENT_QUOTE_NOT_IN_ORIGINAL")


def _confirmed_material(events):
    """只派生冻结R5的三条件；金额/分类未知不自行提升为已确认重大。"""
    return [
        e
        for e in events
        if e.get("applicability") == "FUND_CONFIRMED"
        and e.get("status") == "UNRESOLVED"
        and e.get("severity") in {"MATERIAL", "SEVERE"}
    ]


def _material(inputs, source):
    """原文审阅闭合、重大性未知、R5三条件布尔分别保留，不相互替代。"""
    _require(
        MATERIAL_REVIEW_SHA and inputs.get("material_event_review_ref") == MATERIAL_REVIEW_SHA,
        "MIXED_EXPLICIT_SOURCED_MATERIAL_REVIEW_REQUIRED",
    )
    review = _json(source, MATERIAL_REVIEW_SHA)
    _require(
        review.get("fund_code") == "160323"
        and review.get("share_class") == "A"
        and review.get("as_of") == inputs["as_of_date"]
        and review.get("portfolio_date") == inputs["holding_date"]
        and review.get("review_complete") is True
        and review.get("event_catalog_and_originals_review_complete") is True
        and set(review.get("selected_dimensions", [])) == {"return", "risk"},
        "MIXED_MATERIAL_ORIGINAL_REVIEW_NOT_CLOSED",
    )
    identity, registry = review["identity"], review["sources"]
    _require(
        identity.get("A_code") == "160323"
        and identity.get("C_nav_borrowed") is False
        and set(identity.get("excluded_same_product_aliases", [])) == {"160323", "013360"}
        and any(registry[e["source_id"]]["sha256"] == inputs["primary_ref"] for e in identity["evidence"]),
        "MIXED_MATERIAL_SCOPE_OR_ORIGINALS",
    )
    basic_refs = [r for r in registry.values() if r["kind"] == "LICENSED_FUND_BASIC_RAW_RESPONSE"]
    _require(len(basic_refs) == 1, "MIXED_TARGET_LICENSED_IDENTIFIER_SOURCE")
    basic = _json(source, basic_refs[0]["sha256"])
    fields = basic.get("data", {}).get("fields", [])
    _require(
        basic.get("code") == 0 and {"ts_code", "name", "management", "market"} <= set(fields),
        "MIXED_TARGET_LICENSED_IDENTIFIER_SCHEMA",
    )
    selected = [
        dict(zip(fields, r, strict=True))
        for r in basic["data"]["items"]
        if r[fields.index("ts_code")] == inputs["target"]["ticker"]
    ]
    _require(
        len(selected) == 1
        and selected[0]["ts_code"] == "160323.SZ"
        and selected[0]["name"] == "华夏磐泰混合(LOF)-A"
        and selected[0]["management"] == inputs["manager_name"]
        and selected[0]["market"] == "E",
        "MIXED_TARGET_LICENSED_IDENTIFIER_BINDING",
    )
    for role, key in [
        ("LICENSED_TARGET_NAV_RAW_RESPONSE", "licensed_nav_sources"),
        ("LICENSED_TARGET_DIVIDEND_RAW_RESPONSE", "fund_div_sources"),
    ]:
        _require(
            {r["sha256"] for r in registry.values() if r["kind"] == role} == set(inputs["target"][key]),
            "MIXED_TARGET_CURRENT_ORIGINAL_BINDING",
        )
    originals, excluded = set(), []
    for source_id, ref in registry.items():
        if ref["kind"] in {"RESEARCH_SNAPSHOT", "CURRENT_PEER_ADMISSION_LEDGER"} or ref["kind"].startswith("PARSED_"):
            excluded.append(source_id)
            continue
        _require(ref["sha256"] in source.raw, "MIXED_MATERIAL_ORIGINAL_NOT_ARCHIVED")
        originals.add(ref["sha256"])
    required = set(review["required_material_gate_original_source_refs"])
    _require(
        required
        and required <= originals
        and all(source.sources[h].get("archive_kind", "ORIGINAL") == "ORIGINAL" for h in required),
        "MIXED_EVENT_GATE_REAL_ORIGINALS",
    )
    claims = review.get("claims", [])
    _require(claims and len({c["claim_id"] for c in claims}) == len(claims), "MIXED_EVENT_CLAIMS_REQUIRED")
    by_claim, cache = {}, {}
    for claim in claims:
        _require(claim["source_hash"] in required, "MIXED_EVENT_CLAIM_GATE_SOURCE")
        _claim_text(claim, source, cache)
        _publication(claim, source, inputs["as_of_date"])
        by_claim[claim["claim_id"]] = claim
    events = review.get("fund_gate_events")
    _require(
        isinstance(events, list)
        and events
        and len({e["event_id"] for e in events}) == len(events)
        and review.get("anchors", {}).get("material_events", {}).get("160323") == events,
        "MIXED_EVENT_ANCHORS_REQUIRED",
    )
    for event in events:
        refs, claim_ids = set(event.get("source_refs", [])), event.get("claim_ids", [])
        _require(
            refs
            and refs <= required
            and claim_ids
            and all(
                c in by_claim and by_claim[c]["event_id"] == event["event_id"] and by_claim[c]["source_hash"] in refs
                for c in claim_ids
            ),
            "MIXED_EVENT_CLAIM_EVENT_BINDING",
        )
    pending = bool(_confirmed_material(events))
    _require(
        review.get("material_event_pending_meaning") == "CONFIRMED_FUND_SPECIFIC_MATERIAL_UNRESOLVED_EVENTS_ONLY"
        and type(review.get("material_event_pending")) is bool
        and review["material_event_pending"] is pending
        and type(inputs.get("material_event_pending")) is bool
        and inputs["material_event_pending"] is pending,
        "MIXED_EVENT_GATE_DERIVATION_MISMATCH",
    )
    return {
        "source_sha": MATERIAL_REVIEW_SHA,
        "events_preserved": len(events),
        "claims_verified": len(claims),
        "actual_originals_verified": len(originals),
        "required_gate_originals": len(required),
        "research_snapshots_not_compiled": excluded,
        "material_event_pending": pending,
        "materiality_review": review.get("cash_reconciliation_materiality_review"),
        "preserved_blockers": review.get("preserved_blockers", []),
        "fund_wide_safety_inferred": False,
        "anchors": deepcopy(review["anchors"]),
    }


def _allocations(raw):
    soup = BeautifulSoup(raw, "html.parser")
    tables = [t for t in soup.find_all("table") if "股票占净比" in t.get_text() and "债券占净比" in t.get_text()]
    _require(len(tables) == 1, "MIXED_COHORT_ASSET_TABLE")
    rows = {}
    for tr in tables[0].find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if cells and re.fullmatch(r"\d{4}-\d{2}-\d{2}", cells[0]):
            _require(len(cells) == 5, "MIXED_COHORT_ASSET_ROW")
            number = cells[1].replace("%", "").replace(",", "")
            value = D(number) if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", number) else None
            _require(value is None or value.is_finite() and value >= 0, "MIXED_COHORT_INVALID_STOCK_PERCENT")
            _require(cells[0] not in rows or rows[cells[0]] == value, "MIXED_COHORT_ASSET_DUPLICATE")
            rows[cells[0]] = value
    return rows


def _region_bounds(lines, code, day, stock):
    """行业净值比例为非负有限数；防负数抵销及反向区间进入准入。"""
    _require(all(x["BZDM"] == code and x["FSRQ"][:10] == day for x in lines), "MIXED_COHORT_REGION_DATE")
    _require(
        all(
            not isinstance(x["ZJZBL"], bool) and D(str(x["ZJZBL"])).is_finite() and D(str(x["ZJZBL"])) >= 0
            for x in lines
        ),
        "MIXED_COHORT_NEGATIVE_OR_NONFINITE_REGION",
    )
    domestic = [x for x in lines if re.fullmatch("[A-S]", str(x["HYDM"]))]
    foreign = [x for x in lines if re.fullmatch(r"\d+", str(x["HYDM"]))]
    _require(len(domestic) + len(foreign) == len(lines), "MIXED_COHORT_REGION_CATEGORIES")
    amount = sum((D(str(x["ZJZBL"])) for x in domestic), D(0))
    total = sum((D(str(x["ZJZBL"])) for x in lines), D(0))
    _require(
        stock.is_finite() and stock > D(".005") and abs(total - stock) <= D(".005") * (len(lines) + 1),
        "MIXED_COHORT_REGION_STOCK_MATCH",
    )
    low = max(D(0), amount - D(".005") * len(domestic)) / (stock + D(".005")) * 100
    high = min(D(100), (amount + D(".005") * len(domestic)) / (stock - D(".005")) * 100)
    _require(low.is_finite() and high.is_finite() and 0 <= low <= high <= 100, "MIXED_COHORT_INVALID_REGION_INTERVAL")
    return low, high


def _cohort(inputs, source, review):
    """发现总体/去向实算证据固定原件摘要，预算及地区从原HTML/JSON再次计算。

    法律/策略人工结论保留其固定审阅SHA；不把来源索引自述当成数值证明，
    不读报告内旧绝对路径，也不接纳预计算收益/风险分值。
    """
    _require(
        review.get("cohort_provenance_ref") == inputs.get("cohort_provenance_ref") == COHORT_AUDIT_SHA,
        "MIXED_COHORT_PROVENANCE_REQUIRED",
    )
    audit = _json(source, COHORT_AUDIT_SHA)
    _require(
        audit.get("status") == "PASS"
        and audit.get("fund_code") == "160323"
        and audit["discovery_groups"] == 1812
        and audit["representative_codes"] == 1811
        and audit["discovery_original_replay_passed"] is True
        and not audit["catalog_unknown"]
        and not audit["target_replay_errors"]
        and not audit["admission_mismatches"],
        "MIXED_COHORT_DISCOVERY_AUDIT",
    )
    _require(
        all(r["sha256"] in source.raw for r in audit["source_manifest"] + audit["manual_decision_sources"]),
        "MIXED_COHORT_ALL_ACTUAL_SOURCES_REQUIRED",
    )
    catalog = audit["source_catalog"]
    _require(catalog["sha256"] in source.raw, "MIXED_COHORT_DISCOVERY_RECORD")
    records = {}
    for page in catalog["pages"]:
        raw = _json(source, page["source"]["sha256"])
        data = raw.get("data", {})
        _require(
            raw.get("code") == 0 and len(data["items"]) == page["row_count"] and data["has_more"] is page["has_more"],
            "MIXED_COHORT_DISCOVERY_RAW_PAGE",
        )
        for row in data["items"]:
            record = dict(zip(data["fields"], row, strict=True))
            records[record["ts_code"]] = record
    _require(
        len(records) == catalog["unique_catalog_records"] == 32493
        and any(p["params"].get("offset") == 15000 and p["has_more"] is False for p in catalog["pages"]),
        "MIXED_COHORT_DISCOVERY_COMPLETE_PAGES",
    )
    rows = audit["return_rows"]
    _require(
        len(rows) == len({r["code"] for r in rows}) == 1811
        and dict(Counter(r["final_return_decision"] for r in rows)) == audit["return_destination_counts"],
        "MIXED_COHORT_ALL_DESTINATIONS",
    )
    target = {}
    for proof in audit["target_raw_pdf_replay"]:
        facts = proof["facts"]
        with localcontext() as context:
            context.prec = 28
            ratio = D(facts["stock_cny"]) / D(facts["net_assets_cny"]) * 100
        _require(
            ratio == D(proof["ratio"])
            and set(facts["share_codes_in_actual_report"]) == {"160323", "013360"}
            and proof["source"]["sha256"] in source.raw,
            "MIXED_COHORT_TARGET_RAW_AMOUNTS",
        )
        target[proof["date"]] = ratio
    _require(len(target) == 12, "MIXED_COHORT_TARGET_TWELVE_QUARTERS")
    admitted = {r["code"] for r in rows if r["final_return_decision"] == "ADMITTED"}
    _require(
        admitted == {m["code"] for m in inputs["peers"]}
        and len(admitted) == 79
        and "009515" in admitted
        and review["expected_return_product_count"] == len(admitted),
        "MIXED_COHORT_RETURN_MEMBERSHIP",
    )
    numeric, exact = {}, []
    members = {m["code"]: m for m in inputs["peers"]}
    with localcontext() as context:
        context.prec = 28
        for row in rows:
            if row.get("budget_replay_applicable") is False:
                _require(
                    row["final_return_decision"] == "OTHER_CROSSBORDER_MECHANISM"
                    and "QDII" in row["actual_profile_type"],
                    "MIXED_COHORT_CROSSBORDER",
                )
                continue
            actual = _allocations(source.raw[row["source"]["sha256"]])
            values = {d: actual.get(d) for d in target}
            _require(
                {d: str(v) if v is not None else None for d, v in values.items()} == row["stock_nav_by_quarter"],
                "MIXED_COHORT_STOCK_BUDGET_ORIGINAL",
            )
            lower = [max(D(0), abs(v - target[d]) - D(".005")) for d, v in values.items() if v is not None]
            upper = [abs(v - target[d]) + D(".005") for d, v in values.items() if v is not None]
            passed = len(lower) == 12 and sum(upper) / 12 <= 10 and max(upper) <= 20
            outside = bool(lower) and (sum(lower) / 12 > 10 or max(lower) > 20)
            _require(
                passed is row["return_budget_pass"] and outside is row["return_budget_outside"],
                "MIXED_COHORT_BUDGET_RECALCULATION",
            )
            numeric[row["code"]] = (
                values,
                sum(lower) / 12 if lower else None,
                sum(upper) / 12 if len(upper) == 12 else None,
                max(lower) if lower else None,
                max(upper) if len(upper) == 12 else None,
            )
            if row["code"] in admitted:
                histories = members[row["code"]]["strategy_history_reviews"]
                _require(
                    passed
                    and {h["period"] for h in histories}
                    == {"2023-12-31", "2024-06-30", "2024-12-31", "2025-06-30", "2025-12-31", "2026-06-30"},
                    "MIXED_COHORT_CURRENT_AND_SIX_HISTORIES",
                )
                _require(
                    all(
                        h["source_sha"] in source.raw
                        and _iso(h["published_on"]) <= inputs["as_of_date"]
                        and h["publication_basis"]["published"] == h["published_on"]
                        for h in histories
                    ),
                    "MIXED_COHORT_HISTORY_ORIGINAL_DATE",
                )
            if "precise_exclusion_replayed" in row:
                proof = row["precise_exclusion_replayed"]
                raw = _json(source, proof["actual_source"]["sha256"])
                _require(
                    raw.get("success") == 1
                    and raw["data"].get("is_ai_summary", 0) == 0
                    and raw["data"]["notice_date"][:10] <= inputs["as_of_date"],
                    "MIXED_COHORT_EXACT_ISSUER_ORIGINAL",
                )
                facts = proof["facts"]
                ratio = D(facts["stock_cny"]) / D(facts["net_assets_cny"]) * 100
                _require(
                    ratio == D(proof["ratio"]) and abs(ratio - target[proof["period"]]) > 20,
                    "MIXED_COHORT_EXACT_EXCLUSION",
                )
                body = _compact(raw["data"]["notice_content"])
                _require(
                    all(_compact(facts[k]) in body for k in ["stock_quote", "net_assets_quote"]),
                    "MIXED_COHORT_EXACT_AMOUNT_QUOTES",
                )
                exact.append(row["code"])
        risk_members = []
        for row in audit["risk_rows"]:
            code, quarters = row["code"], {}
            values, mean_low, mean_high, max_low, max_high = numeric[code]
            for ref in row["actual_region_sources"]:
                raw = _json(source, ref["sha256"])
                _require(raw.get("ErrCode") == 0 and raw["Data"]["FundCode"] == code, "MIXED_COHORT_REGION_IDENTITY")
                for quarter in raw["Data"]["QuarterInfos"]:
                    day = quarter["JZRQ"][:10]
                    if day not in target:
                        continue
                    lines = quarter["HYPZInfo"]
                    low, high = _region_bounds(lines, code, day, values[day])
                    _require(day not in quarters or quarters[day] == (low, high), "MIXED_COHORT_REGION_DUPLICATE")
                    quarters[day] = (low, high)
            _require(
                set(quarters) == set(target) and all(low >= 80 for low, high in quarters.values()),
                "MIXED_COHORT_RETURN_REGION",
            )
            decision = (
                "BUDGET_OUTSIDE"
                if mean_low > 5 or max_low > 10
                else "REGION_OUTSIDE"
                if any(high < 90 for low, high in quarters.values())
                else "ADMITTED"
                if mean_high <= 5 and max_high <= 10 and all(low >= 90 for low, high in quarters.values())
                else "UNRESOLVED"
            )
            _require(decision == row["decision"], "MIXED_COHORT_RISK_RECALCULATION")
            if decision == "ADMITTED":
                risk_members.append(code)
    _require(
        len(audit["risk_rows"]) == 79
        and sorted(risk_members) == sorted(review["risk_codes"]) == sorted(audit["risk_members"])
        and len(risk_members) == 17
        and len(exact) == 15,
        "MIXED_COHORT_RISK_AND_BOUNDARIES",
    )
    return {
        "source_sha": COHORT_AUDIT_SHA,
        "actual_source_count": len(audit["source_manifest"]),
        "discovery_groups": 1812,
        "representative_codes": 1811,
        "licensed_catalog_records": len(records),
        "return_destination_counts": audit["return_destination_counts"],
        "return_products": len(admitted),
        "risk_products": len(risk_members),
        "exact_amount_exclusion_codes": exact,
        "boundary_cases": audit["closest_risk_budget_cases"],
        "old_absolute_paths_read": False,
        "manual_legal_strategy_source_shas": [r["sha256"] for r in audit["manual_decision_sources"]],
    }


# 最多保留两个纯编译结果；不保存Originals、raw字节、来源核验成功状态或DB资格。
# 锁从真实原件校验前覆盖至结果入缓存，避免同进程并发miss各持有约500MB原件再等待。
_CALCULATION_CACHE = OrderedDict()
_CALCULATION_LOCK = RLock()


def _calculation_cache_key(inputs, root, protocol):
    """键绑定安全根、完整输入元数据及冻结规则；不以mtime或路径名替代SHA核验。"""

    def fingerprint(value):
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
        return content_digest(encoded)

    frozen_context = {
        "protocol": protocol,
        "compiler_version": VERSION,
        "time_contract": TIME_CONTRACT,
        "admission_review_sha": ADMISSION_REVIEW_SHA,
        "material_review_sha": MATERIAL_REVIEW_SHA,
        "cohort_audit_sha": COHORT_AUDIT_SHA,
        "primary_sha": PRIMARY_SHA,
    }
    return str(Path(root).resolve()), fingerprint(inputs), fingerprint(frozen_context)


def calculate(inputs: dict, *, root: Path) -> dict:
    """逐次真实核原件/准入/身份，再复用完全相同来源及规则下的确定性编译。

    本锁只在同进程singleflight；每个调用均实际执行_originals、_admission与
    原身份guard后才查缓存。prepare/check_binding的fresh current资格仍逐次核，
    此缓存不能替代DB当前指针、撤销、有效期或最新事项复核。
    """
    with _CALCULATION_LOCK:
        # 调用期间固定完整输入，调用方随后修改字典不能污染键或已核字节的绑定。
        inputs = deepcopy(inputs)
        source = _originals(inputs, root)
        review = _admission(inputs, source)
        _require(
            inputs.get("fund_code") == "160323"
            and inputs.get("share_class") == "A"
            and inputs.get("family") == "MIXED"
            and inputs.get("primary_ref") == PRIMARY_SHA
            and inputs.get("legal_name") == "华夏磐泰混合型证券投资基金(LOF)"
            and inputs.get("manager_name") == "华夏基金管理有限公司",
            "MIXED_TARGET_IDENTITY",
        )
        protocol = deepcopy(PROTOCOL)
        key = _calculation_cache_key(inputs, root, protocol)
        if key in _CALCULATION_CACHE:
            _CALCULATION_CACHE.move_to_end(key)
            return deepcopy(_CALCULATION_CACHE[key])
        # 全部原计算体原样迁至下方；失败自动释放锁，异常/未完成结果不入缓存。
        result = _calculate_verified(inputs, source, review, PROTOCOL=protocol)
        _CALCULATION_CACHE[key] = deepcopy(result)
        _CALCULATION_CACHE.move_to_end(key)
        while len(_CALCULATION_CACHE) > 2:
            _CALCULATION_CACHE.popitem(last=False)
        return deepcopy(result)


def _calculate_verified(inputs, source, review, *, PROTOCOL):
    """只接收本次已核源/准入与键绑定的规则副本；原数学/入组/事件AST保持不变。"""
    cohort = _cohort(inputs, source, review)
    event = _material(inputs, source)
    compiled = _compile(inputs, source, review)
    packet = {
        "fund_code": "160323",
        "as_of_date": inputs["as_of_date"],
        "basis": ACTIVE_BASIS,
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_metrics": {
            "target_product_aliases": ["160323", "013360"],
            "material_event_pending": event["material_event_pending"],
            "observations": compiled["observations"],
        },
        "r4_evidence_registry": {},
        "raw_archive": [{"kind": source.sources[h].get("archive_kind", "ORIGINAL"), "sha256": h} for h in source.raw],
        "quality_public": {"dataDates": {"nav": inputs["as_of_date"], "holdings": inputs["holding_date"]}},
    }
    result = _calculate_frozen(packet, PROTOCOL, event["anchors"], selected_dimensions={"return", "risk"})
    _require(all(result.get(k) is None for k in ["score", "raw_score", "grade"]), "MIXED_COMPONENT_TOTAL_LEAK")
    return {
        "dimensions": result["dimensions"],
        "observations": compiled["observations"],
        "complete_score": None,
        "grade": None,
        "admission_ready": not event["material_event_pending"],
        "audit": {**compiled["audit"], "cohort_provenance": cohort, "material_review": event},
        "standard_replay": result,
    }


def check_binding(bundle: dict, inputs: dict) -> None:
    """本A份额、法律产品、管理人及主中报绑定最新current。"""
    _require(
        bundle.get("family") == inputs.get("family") == "MIXED"
        and bundle.get("fund_code") == inputs.get("fund_code") == "160323"
        and bundle.get("share_class") == inputs.get("share_class") == "A"
        and str(bundle.get("as_of_date")) == inputs.get("as_of_date") == "2026-09-30"
        and str(bundle.get("holding_date")) == inputs.get("holding_date") == "2026-06-30"
        and bundle["identity"]["legal_name"] == inputs.get("legal_name") == "华夏磐泰混合型证券投资基金(LOF)"
        and bundle["identity"]["manager_name"] == inputs.get("manager_name") == "华夏基金管理有限公司"
        and bundle["original_report_archive"]["sha256"] == inputs.get("primary_ref") == PRIMARY_SHA,
        "MIXED_CURRENT_BINDING",
    )


def verify_registered_original(bundle: dict, ref: dict, *, root: Path) -> None:
    """只核本组件同一来源ref的真实PDF，沿Originals的50MB硬上限。

    供快照extra_archives定向使用；不改变普通归档25MB界，不读取派生文档，
    不缓存校验结果。路径、文件类型、长度和SHA每次重新核验。
    """
    inputs = bundle.get("mixed_performance_calculation", {}).get("inputs", {})
    check_binding(bundle, inputs)
    digest = ref.get("sha256")
    _require(
        inputs.get("version") == VERSION
        and inputs.get("admission_review_ref") == ADMISSION_REVIEW_SHA
        and inputs.get("material_event_review_ref") == MATERIAL_REVIEW_SHA
        and inputs.get("cohort_provenance_ref") == COHORT_AUDIT_SHA
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest)
        and inputs.get("sources", {}).get(digest) == ref,
        "MIXED_BOUND_ORIGINAL_REFERENCE",
    )
    _require(
        ref.get("kind") == "pdf" and ref.get("archive_kind", "ORIGINAL") == "ORIGINAL", "MIXED_BOUND_ORIGINAL_PDF_ROLE"
    )
    relative = Path(ref["file"])
    _require(
        not relative.is_absolute()
        and relative.parts
        and relative.parts[0] == "sources"
        and ".." not in relative.parts
        and ref["file"] in {"sources/" + digest + ".pdf", "sources/pipeline/" + digest + ".pdf"}
        and (not ref.get("path") or ref["path"] == ref["file"]),
        "MIXED_BOUND_ORIGINAL_RELATIVE_PATH",
    )
    root = Path(root).resolve()
    path = (root / relative).resolve()
    _require(path.is_relative_to(root / "sources") and source_exists(path), "MIXED_BOUND_ORIGINAL_PATH")
    _require(0 < source_size(path) <= 50000000, "MIXED_BOUND_ORIGINAL_SIZE")
    raw = source_bytes(path)
    _require(len(raw) <= 50_000_000, "MIXED_BOUND_ORIGINAL_SIZE")
    _require(content_digest(raw) == digest, "MIXED_BOUND_ORIGINAL_HASH")
    _require(raw.startswith(b"%PDF"), "MIXED_BOUND_ORIGINAL_PDF_BYTES")


def prepare(session: Session, request: dict, *, root: Path) -> dict:
    """只组织本期追加请求；重读最新current，保留其他六维，不写业务库或发布。"""
    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    root = Path(root).resolve()
    code, raw = request["fund_code"], deepcopy(request["inputs"])
    _require(code == "160323" and request["as_of_date"] == "2026-09-30", "MIXED_PREPARE_PERIOD")
    previous = current_inputs(session, [code]).get(code)
    _require(previous and str(previous["as_of_date"]) == request["as_of_date"], "MIXED_LATEST_CURRENT_REQUIRED")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    check_binding(bundle, raw)
    _require(
        bundle.get("basis") == BASIS
        and not any(
            bundle.get(k)
            for k in [
                "r5_calculation",
                "passive_components_calculation",
                "passive_performance_calculation",
                "enhanced_performance_calculation",
                "bosera_performance_calculation",
                "gold_components_calculation",
            ]
        ),
        "MIXED_PARTIAL_SNAPSHOT_REQUIRED",
    )
    result = calculate(raw, root=root)
    _require(result["admission_ready"], "MIXED_CONFIRMED_MATERIAL_EVENT_NOT_ADMISSIBLE")
    # 相同输入或分数不能代替fresh current绑定、全部原件重算和正式归档。
    # 重复发布由正式fingerprint负责；这里始终组织本期可审阅的追加请求。
    document = bundle["reviewed_document"]
    for key in ["return", "risk"]:
        explanation = request["explanations"][key].strip()
        _require(len(explanation) >= 40, "MIXED_BUSINESS_EXPLANATION_REQUIRED")
        point = result["dimensions"][key]
        document["dimensions"][key] = {"lower": point, "upper": point, "candidate": point, "reason": explanation}
        bundle.setdefault("public_explanations", {})[key] = explanation
        bundle.get("public_metrics", {}).pop(key, None)
    # 旧61只/35%及“仍缺17只参照”不再代表这次实际来源；保留旧证据作历史追溯。
    input_archive = archive_json(root, raw)
    previous_evidence = document.get("return_evidence")
    same_evidence_package = isinstance(previous_evidence, dict) and (
        previous_evidence.get("mixed_input_archive") == input_archive
        and previous_evidence.get("admission_review_sha256") == raw["admission_review_ref"]
        and previous_evidence.get("cohort_original_proof_sha256") == raw["cohort_provenance_ref"]
    )
    if same_evidence_package:
        # 同一79标准包已经保存了历史链；再包裹自身会改变研究档案摘要并制造重复批次。
        # 此处仅保持元数据稳定，之前的fresh current/原件重算及之后的归档/快照照常执行。
        document["return_evidence"] = deepcopy(previous_evidence)
    else:
        document["return_evidence"] = {
            "mixed_input_archive": input_archive,
            "admission_review_sha256": raw["admission_review_ref"],
            "cohort_original_proof_sha256": raw["cohort_provenance_ref"],
            "previous_trace": deepcopy(previous_evidence),
        }
    bundle["comparison_product_count"] = len(raw["peers"])
    # 旧底稿曾把有分类假设的现金上界写成总体结论；本版原件仍保留分类未知，
    # 两项历史表现补齐后也不能继续透出过强的执行判断。仅修无点分项的说明。
    execution = document["dimensions"].get("execution")
    if execution is None:
        # 原空节点的数学域即0..100；显式写出相同未知域仅用于承载真实说明。
        execution = {"lower": "0", "upper": "100", "candidate": None}
        document["dimensions"]["execution"] = execution
    if isinstance(execution, dict) and execution.get("candidate") is None:
        explanation = (
            "18期公开配置及适用合同已核；同日合格现金及一年内政府债范围、分类和适用口径仍待核实，"
            "暂不能判断是否满足合同要求，目标执行暂不评分。"
        )
        execution["reason"] = explanation
        bundle.setdefault("public_explanations", {})["execution"] = explanation
    limits = []
    for value in bundle.get("public_limitations", []):
        if "3个维度得分确定" in value and "35%" in value:
            value = "部分分不代表完整总分，其余缺项继续保留。"
        if "合同现金条件差异" in value:
            value = "现金比例、地方政府债归类及相关处置仍待核实，暂不能判断是否满足合同要求。"
        if "17只参照仍缺必要历史条文" in value:
            value = value.replace("；17只参照仍缺必要历史条文", "")
        if "仍缺完整债券明细及信用、期限、成交和资产质量证据" in value:
            value = value.replace("信用、期限、成交和资产质量证据", "信用、期限和资产质量证据")
        if value and value not in limits:
            limits.append(value)
    if not any("现金" in v for v in limits):
        limits.append("现金比例、分类及相关处置仍待核实，不据此认定合规或违规。")
    bundle["public_limitations"] = limits
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, document),
        mixed_performance_calculation={"inputs": raw, "input_archive": input_archive},
    )
    known = {r["sha256"] for r in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(ref for h, ref in raw["sources"].items() if h not in known)
    validate_sources(bundle, datetime.now(UTC), root=root)
    checked = snapshot_calculate(bundle, root=root)
    _require(
        checked["status"] == "PARTIAL" and checked["score"] is None and checked.get("grade") is None,
        "MIXED_CANNOT_CLAIM_COMPLETE",
    )
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}
