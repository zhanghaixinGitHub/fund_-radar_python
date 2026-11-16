"""越南人民币股票QDII的估值契约；日期支持与实际行情、汇率就绪分开核验。"""

import hashlib
import json
import re
from copy import deepcopy
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

from app.services.direction_1d_protocol import digest

POLICY = "QDII_CNY_VN_CONTRACT_SESSION_V1"
VN_ZONE = ZoneInfo("Asia/Ho_Chi_Minh")
LONDON = ZoneInfo("Europe/London")


def _stamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("TIMEZONE_REQUIRED")
    return result


def _positive(value):
    number = Decimal(str(value))
    if not number.is_finite() or number <= 0:
        raise ValueError("POSITIVE_FINITE_VALUE_REQUIRED")
    return number


def _host(document):
    uri = urlparse(document.get("url", ""))
    if uri.scheme != "https" or uri.username or uri.password:
        raise ValueError("HTTPS_SOURCE_REQUIRED")
    return uri.hostname or ""


def _under(host, domain):
    return host == domain or host.endswith("." + domain)


def _document(documents, ref, kinds, at):
    document = documents[ref]
    if document.get("kind") not in kinds or digest(document["body"]) != document.get("body_hash"):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    if _stamp(document["available_at"]) > at:
        raise ValueError("SOURCE_AFTER_CUTOFF")
    _host(document)
    return document


def vietnam_calendar(document, review, year):
    """消费已逐表审阅并绑定摘要的官方年度表，不从普通工作日生成假期。"""
    if document.get("kind") != "EXCHANGE_CALENDAR" or not _under(_host(document), "hsx.vn"):
        raise ValueError("VN_OFFICIAL_CALENDAR_REQUIRED")
    parsed = json.loads(document["body"])
    days = parsed["closed_dates"]
    if (
        parsed["year"] != year
        or days != sorted(set(days))
        or len(days) < 10
        or any(date.fromisoformat(day).year != year for day in days)
        or review.get("body_hash") != digest(document["body"])
        or review.get("closed_dates_hash") != digest(days)
        or review.get("coverage_start") != f"{year}-01-01"
        or review.get("coverage_end") != f"{year}-12-31"
    ):
        raise ValueError("VN_CALENDAR_REVIEW_OR_COVERAGE_INVALID")
    if any(date.fromisoformat(day).weekday() != 5 for day in parsed.get("makeup_saturdays_not_trading", [])):
        raise ValueError("VN_MAKEUP_DAY_INVALID")
    return {date.fromisoformat(day) for day in days}, date(year, 1, 1), date(year, 12, 31)


def _contract(profile, material, documents, window, at):
    evidence = material["calendar_evidence"]
    code = material["fund_code"]
    name = profile.get("fund_name", "")
    if (
        evidence.get("policy") != POLICY
        or "QDII" not in name.upper()
        or "越南" not in name
        or re.search("美元|港元|FOF|联接", name, re.I)
        or evidence.get("base_date") != window["base_nav_date"]
        or evidence.get("target_date") != window["target_nav_date"]
    ):
        raise ValueError("VN_PRODUCT_OR_WINDOW_INVALID")
    own = _document(documents, evidence["prospectus_ref"], {"PROSPECTUS"}, at)
    report = _document(documents, evidence["report_ref"], {"REPORT"}, at)
    if code not in own.get("codes", []) or code not in report.get("codes", []):
        raise ValueError("VN_OWN_PRODUCT_PROOF_REQUIRED")
    clauses = evidence["valuation_quotes"]
    if set(clauses) != {"day", "close", "fx"} or any(not q or q not in own["body"] for q in clauses.values()):
        raise ValueError("VN_CONTRACT_QUOTES_INVALID")
    compact = {k: re.sub(r"\s+", "", v).replace("：", ":") for k, v in clauses.items()}
    if (
        "估值日为本基金相关的证券交易场所的交易日" not in compact["day"]
        or not all(s in compact["close"] for s in ("股票", "估值日", "收盘价"))
        or not all(s in compact["fx"] for s in ("中间价", "彭博", "伦敦时间16:00", "美元", "折算"))
    ):
        raise ValueError("VN_ACTUAL_VALUATION_CLAUSES_REQUIRED")
    return evidence, own


def normalize_dnse_history(original, raw: bytes, symbol: str, *, at):
    """把实际DNSE日线响应转成单证券逐行原文；保留原始字节摘要，不推测报价单位。"""
    uri = urlparse(original["url"])
    if (
        uri.hostname != "api.dnse.com.vn"
        or not uri.path.startswith("/chart-api/v2/ohlcs/")
        or parse_qs(uri.query).get("symbol") != [symbol]
        or parse_qs(uri.query).get("resolution") != ["1D"]
        or hashlib.sha256(raw).hexdigest() != original["raw_sha256"]
        or at.tzinfo is None
        or at < _stamp(original["available_at"])
        or _stamp(original.get("original_available_at", original["available_at"])) > _stamp(original["available_at"])
    ):
        raise ValueError("VN_DNSE_RAW_IDENTITY_INVALID")
    payload = json.loads(raw)
    if len(payload["t"]) != len(payload["c"]) or len(payload["t"]) < 2:
        raise ValueError("VN_DNSE_DAILY_ROWS_REQUIRED")
    rows = []
    for stamp, value in zip(payload["t"], payload["c"], strict=True):
        instant = datetime.fromtimestamp(stamp, VN_ZONE)
        rows.append(
            {
                "symbol": symbol,
                "date": instant.date().isoformat(),
                "bar_timestamp": instant.isoformat(),
                "close": str(_positive(value)),
            }
        )
    days = [r["date"] for r in rows]
    if days != sorted(set(days)) or any(date.fromisoformat(day) > at.astimezone(VN_ZONE).date() for day in days):
        raise ValueError("VN_DNSE_DUPLICATE_OR_FUTURE_DAY")
    body = json.dumps({"symbol": symbol, "rows": rows}, ensure_ascii=False, sort_keys=True)
    return {
        **deepcopy(original),
        "id": original["id"] + ":verified-daily",
        "body": body,
        "body_hash": digest(body),
        "body_scope": "VERIFIED_SOURCE_TRANSFORM",
        "available_at": at.isoformat(),
        "original_available_at": original.get("original_available_at", original["available_at"]),
        "origin_document_ref": original["id"],
        "normalization": "单证券日线日期取越南时区，保留柱标记；价格单位未从数值猜测。",
    }


def _daily_pair(source, proof, symbol, base, documents, evidence):
    """证券、日期、价格和涨跌幅必须来自同一条已核记录及前一真实交易日。"""
    payload = json.loads(source["body"])
    rows = payload["rows"]
    days = [r["date"] for r in rows]
    query_symbol = parse_qs(urlparse(source["url"]).query).get("symbol")
    resolution = parse_qs(urlparse(source["url"]).query).get("resolution")
    if (
        payload.get("symbol") != symbol
        or (query_symbol is not None and query_symbol != [symbol])
        or (resolution is not None and resolution != ["1D"])
        or any(r.get("symbol", symbol) != symbol for r in rows)
        or days != sorted(set(days))
        or any(day > base for day in days)
        or proof not in source["body"]
    ):
        raise ValueError("VN_SINGLE_SECURITY_RECORD_REQUIRED")
    calendar = documents[evidence["calendar_refs"]["VN"]]
    closed, start, _ = vietnam_calendar(calendar, evidence["calendar_review"], date.fromisoformat(base).year)
    previous = date.fromisoformat(base) - timedelta(days=1)
    while previous.weekday() >= 5 or previous in closed:
        previous -= timedelta(days=1)
    if previous < start:
        raise ValueError("VN_PREVIOUS_SESSION_CALENDAR_REQUIRED")
    by_day = {r["date"]: r for r in rows}
    current, prior = by_day[base], by_day[str(previous)]
    if json.loads(proof) not in (payload, current):
        raise ValueError("VN_CURRENT_RECORD_PROOF_REQUIRED")
    return _positive(current["close"]), _positive(prior["close"])


def _fx(material, documents, evidence, window, at):
    base = window["base_nav_date"]
    official = _document(documents, evidence["fx_ref"], {"FX_REFERENCE_DATA"}, at)
    if _host(official) not in {"www.safe.gov.cn", "www.chinamoney.com.cn"}:
        raise ValueError("VN_OFFICIAL_USD_CNY_REQUIRED")
    rows = [r for r in json.loads(official["body"])["rates"] if r["date"] == base and r["currency"] == "USD"]
    if len(rows) != 1:
        raise ValueError("VN_USD_CNY_DATE_OR_DUPLICATE_INVALID")
    usd_cny = _positive(rows[0]["cny_per_unit"])
    fixing = _document(documents, evidence["vnd_fx_ref"], {"FX_FIXING_DATA", "CUSTODIAN_FX_REFERENCE"}, at)
    row = json.loads(fixing["body"])
    value = _positive(row["rate"])
    pair = (row["base_currency"], row["quote_currency"])
    if row["date"] != base or pair not in {("USD", "VND"), ("VND", "USD")}:
        raise ValueError("VN_FIXING_DATE_OR_PAIR_INVALID")
    if _stamp(fixing["available_at"]).astimezone(VN_ZONE).date() < date.fromisoformat(base):
        raise ValueError("VN_FIXING_OBSERVATION_BEFORE_BASE_DATE")
    if fixing["kind"] == "FX_FIXING_DATA":
        expected = datetime.combine(date.fromisoformat(base), time(16), LONDON)
        if (
            not _under(_host(fixing), "bloomberg.com")
            or row.get("rate_type") != "BLOOMBERG_LONDON_1600"
            or _stamp(row["fixing_at"]) != expected
            or _stamp(fixing["available_at"]) < expected
            or at < expected
        ):
            raise ValueError("VN_CONTRACT_LONDON_FIXING_REQUIRED")
    else:
        # 无法联网不代表托管人已启用替代。必须是本基金当日真实托管估值证据。
        quote = evidence.get("custodian_application_quote", "")
        custodian = "香港上海汇丰银行有限公司"
        if (
            not (_under(_host(fixing), "hsbc.com.hk") or _under(_host(fixing), "thfund.com.cn"))
            or material["fund_code"] not in fixing.get("codes", [])
            or not quote
            or quote not in fixing["body"]
            or not all(s in quote for s in (material["fund_code"], base, custodian, "估值"))
            or row.get("rate_type") != "CUSTODIAN_APPLIED_FALLBACK"
            or row.get("reason") != "CONTRACT_RATE_UNAVAILABLE"
            or row.get("applied_for_fund_valuation") is not True
            or "基金托管人或境外托管人" not in evidence["valuation_quotes"]["fx"]
        ):
            raise ValueError("VN_ACTUAL_CUSTODIAN_FALLBACK_REQUIRED")
    # 不猜报价方向。USD/VND表示每美元多少越南盾；VND/USD表示每越南盾多少美元。
    vnd_cny = usd_cny / value if pair == ("USD", "VND") else usd_cny * value
    return {"usd_cny": str(usd_cny), "vnd_cny": str(vnd_cny), "method": row["rate_type"], "source_ref": fixing["id"]}


def _quote(quote, documents, evidence, base, at, *, index, symbol):
    if quote.get("date") != base:
        raise ValueError("VN_QUOTE_BASE_DATE_MISSING_OR_STALE")
    if quote.get("valuation_market") != "VN" or quote.get("currency") != "VND":
        raise ValueError("VN_QUOTE_MARKET_OR_CURRENCY_INVALID")
    close = datetime.combine(date.fromisoformat(base), time(15), VN_ZONE)
    hours = _document(documents, evidence["market_hours_ref"], {"EXCHANGE_TRADING_HOURS"}, at)
    if not any(_under(_host(hours), domain) for domain in ("hsx.vn", "ssi.com.vn", "dnse.com.vn")):
        raise ValueError("VN_TRADING_HOURS_SOURCE_REQUIRED")
    if not re.search(r"15(?:h|:)00", hours["body"]) or not re.search(r"14(?:h|:)45", hours["body"]):
        raise ValueError("VN_TRADING_COMPLETION_PROOF_REQUIRED")
    if (
        quote.get("session") not in {"REGULAR_CLOSE", "COMPLETED_DAILY_HISTORY"}
        or quote.get("close_timestamp_kind") != "EXCHANGE_COMPLETION_BOUND"
        or _stamp(quote["closed_at"]) != close
        or at < close
    ):
        raise ValueError("VN_COMPLETED_SESSION_REQUIRED")
    source = _document(documents, quote["source_ref"], {"MARKET_DATA"}, at)
    observed = _stamp(source.get("original_available_at", source["available_at"]))
    if observed < close or observed > _stamp(source["available_at"]):
        raise ValueError("VN_INTRADAY_SOURCE_REJECTED")
    price = _positive(quote["close"])
    change = Decimal(str(quote["change_pct"]))
    if not change.is_finite():
        raise ValueError("VN_CHANGE_VALUE_INVALID")
    proof = quote.get("price_quote", "")
    if not proof:
        raise ValueError("VN_PRICE_DATE_SOURCE_PROOF_REQUIRED")
    current, previous = _daily_pair(source, proof, symbol, base, documents, evidence)
    expected_change = (current / previous - 1) * 100
    if abs(change - expected_change) > Decimal("0.0001"):
        raise ValueError("VN_CHANGE_NOT_BOUND_TO_DAILY_SOURCE")
    if index:
        if (
            quote.get("price_unit") != "INDEX_POINTS"
            or price != current
            or _positive(quote["previous_close"]) != previous
        ):
            raise ValueError("VN_INDEX_POINTS_REQUIRED")
    else:
        raw = _positive(quote["provider_close"])
        scale = _positive(quote["provider_to_vnd_multiplier"])
        units = _document(documents, quote["unit_evidence_ref"], {"PROVIDER_UNIT_EXCERPT"}, at)
        unit_proof = quote.get("unit_quote", "")
        # 单位原文必须属于同一提供方，VNDIRECT说明不能替DNSE背书。
        domains = ("vndirect.com.vn", "dnse.com.vn")
        same = any(_under(_host(source), d) and _under(_host(units), d) for d in domains)
        if (
            not same
            or quote.get("price_unit") != "VND_PER_SHARE"
            or scale not in {Decimal(1), Decimal(1000)}
            or price != raw * scale
            or raw != current
            or _positive(quote["previous_close"]) != previous * scale
            or not unit_proof
            or unit_proof not in units["body"]
            or not re.search("VND|VNĐ|đồng", unit_proof, re.I)
            or (scale == 1000 and not re.search(r"(?:x|×)\s*1[.,]?000", unit_proof, re.I))
            or (scale == 1 and re.search(r"1[.,]?000", unit_proof))
        ):
            raise ValueError("VN_PROVIDER_PRICE_UNIT_NOT_VERIFIED")


def assess_vietnam_valuation(profile, material, documents, window, *, as_of):
    """返回可执行的缺项；supported只表示合同/年度日历适配，不代表能预测。"""
    result = {
        "policy": POLICY,
        "supported": False,
        "ready": False,
        "fx_ready": False,
        "market_ready": False,
        "gaps": [],
        "scope": "CONTRACT_CALENDAR_MARKET_FX_ONLY_NAV_SEPARATE",
    }
    try:
        if as_of is None or as_of.tzinfo is None:
            raise ValueError("VN_CUTOFF_TIMEZONE_REQUIRED")
        evidence, own = _contract(profile, material, documents, window, as_of)
        base, target = (date.fromisoformat(window[k]) for k in ("base_nav_date", "target_nav_date"))
        calendar = _document(documents, evidence["calendar_refs"]["VN"], {"EXCHANGE_CALENDAR"}, as_of)
        closed, start, stop = vietnam_calendar(calendar, evidence["calendar_review"], base.year)
        if not all(start <= d <= stop and d.weekday() < 5 and d not in closed for d in (base, target)):
            raise ValueError("VN_BASE_OR_TARGET_NOT_VERIFIED_TRADING_DAY")
        result.update(
            supported=True,
            coverage_start=str(start),
            coverage_end=str(stop),
            contract_ref=own["id"],
            calendar_ref=calendar["id"],
        )
    except (KeyError, TypeError, ValueError, ArithmeticError) as error:
        result["gaps"].append(
            {
                "kind": "CONTRACT_CALENDAR",
                "reason": "本基金的估值条款或官方年度日历证据尚未通过核验。",
                "code": str(error) if isinstance(error, ValueError) else "VN_CONTRACT_FIELDS_MISSING",
            }
        )
        return result
    try:
        result["fx"] = _fx(material, documents, evidence, window, as_of)
        result["fx_ready"] = True
    except (KeyError, TypeError, ValueError, ArithmeticError) as error:
        result["gaps"].append(
            {
                "kind": "FX",
                "code": str(error) if isinstance(error, ValueError) else "VN_CONTRACT_FIXING_MISSING",
                "reason": "需核验同日官方美元兑人民币及合同约定的越南盾兑美元估值汇率，不能用代理牌价替代。",
            }
        )
    rows = [
        (c.get("code", ""), c.get("quote") or {}, False, c.get("ticker", c.get("code", "")))
        for c in material.get("companies", [])
    ]
    rows += [(code, q, True, code) for code, q in material.get("market", {}).items()]
    if not material.get("companies") or not material.get("market"):
        result["gaps"].append(
            {"kind": "MARKET", "code": "VN_EXPOSURE_OR_INDEX_MISSING", "reason": "主要持仓或适用指数资料尚未取得。"}
        )
    for code, quote, index, symbol in rows:
        try:
            if not index and not re.fullmatch(r"VN[A-Z0-9]{10}|[A-Z0-9]+\.VN", code):
                raise ValueError("VN_UNSUPPORTED_SECURITY_IDENTITY")
            if index and code not in {"VN30", "VNINDEX"}:
                raise ValueError("VN_UNSUPPORTED_INDEX")
            _quote(quote, documents, evidence, window["base_nav_date"], as_of, index=index, symbol=symbol)
        except (KeyError, TypeError, ValueError, ArithmeticError) as error:
            result["gaps"].append(
                {
                    "kind": "MARKET",
                    "reason": "该证券的基准日收盘、来源或报价单位尚未完整核验。",
                    "security": code,
                    "code": str(error) if isinstance(error, ValueError) else "VN_QUOTE_FIELDS_MISSING",
                }
            )
    result["market_ready"] = not any(g["kind"] == "MARKET" for g in result["gaps"])
    result["ready"] = result["supported"] and result["fx_ready"] and result["market_ready"]
    return result


def future_sessions(data, documents, sessions, coverage_end, calendar_hash, *, at):
    """仅延续中越官方日历一致区间；不把相交日期说成永久估值条款。"""
    evidence = data["calendar_evidence"]
    profile = {"fund_name": data["fund_name"]}
    status = assess_vietnam_valuation(profile, data, documents, data["window"], as_of=at)
    if not status["supported"]:
        raise ValueError("VN_CONTRACT_CALENDAR_UNVERIFIED")
    calendar = documents[evidence["calendar_refs"]["VN"]]
    target = date.fromisoformat(data["window"]["target_nav_date"])
    closed, start, stop = vietnam_calendar(calendar, evidence["calendar_review"], target.year)
    end = min(coverage_end, stop)
    domestic = set(sessions)
    day = target
    while day <= end:
        if (day in domestic) != (day.weekday() < 5 and day not in closed):
            end = day - timedelta(days=1)
            break
        day += timedelta(days=1)
    own = documents[evidence["prospectus_ref"]]
    source_hash = digest([POLICY, calendar_hash, own["body_hash"], calendar["body_hash"], evidence["calendar_review"]])
    return tuple(d for d in sessions if start <= d <= end), end, source_hash
