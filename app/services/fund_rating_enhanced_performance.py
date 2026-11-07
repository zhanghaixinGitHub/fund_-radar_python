"""同指数增强基金三项表现原件编译与标准分项复算；不发布、不导入研究目录代码。

输入显式提供冻结日历、固定参照、原净值、分红、公开曲线及原合同审阅。
净值增长沿2026-10-19冻结口径，用真实除息前一交易日净值扣单位分红。
"""

import json
import math
import re
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D
from decimal import localcontext
from functools import lru_cache
from pathlib import Path
from statistics import median

from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_direct_index_math import performance
from app.services.fund_rating_inputs import canonical
from app.services.fund_rating_market_review import normalized_text
from app.services.fund_rating_pdf import text_pages
from app.services.fund_rating_public_market import provider_history
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS
from app.services.material_content import document_reader as PdfReader

VERSION = "R5_ENHANCED_500_PERFORMANCE_ORIGINALS_20261029_1"
# 本次独立审阅的不可变事项底稿。新底稿须重新审阅并提升组件版本，
# 不允许调用方只把pending改成False绕过原事件适用性判断。
MATERIAL_REVIEW_HASH = "dace6f54941d848656036d51f20ae1a5a25a6e40bd64175dad2b4ed745c33ff7"
TIME_CONTRACT = {
    "kind": "RETROSPECTIVE_HISTORICAL_QUALITY_RESEARCH",
    "valuation_window_end": "as_of_date",
    "nav_and_market_publication_boundary": "research_on",
    "fundamental_disclosure_boundary": "as_of_date",
    "not_historical_asof_tradable_signal": True,
}
DIVIDEND_FIELDS = [
    "ts_code",
    "ann_date",
    "imp_anndate",
    "ex_date",
    "div_proc",
    "div_cash",
    "base_unit",
    "record_date",
    "pay_date",
]


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _read(ref, root):
    """读取已由Originals限目录、校字节的原件；嵌套旧路径仅作来源追溯。"""
    digest = ref["sha256"]
    _require(isinstance(root, Originals) and digest in root.raw, "ENHANCED_SOURCE_NOT_ARCHIVED")
    return root.raw[digest]


def _json(ref, root):
    return json.loads(_read(ref, root), parse_float=D)


def _iso(value):
    _require(isinstance(value, str) and re.fullmatch(r"(?:\d{8}|\d{4}-\d{2}-\d{2})", value), "ENHANCED_DATE_SCHEMA")
    return date.fromisoformat(value).isoformat()


def _originals(inputs, root):
    """区分历史价格在研究日已知与合同/持仓在月末已公开，保留原日期。"""
    _require(inputs.get("time_contract") == TIME_CONTRACT, "ENHANCED_EXPLICIT_TIME_CONTRACT_REQUIRED")
    research = _iso(inputs.get("research_on"))
    _require(research >= inputs["as_of_date"], "ENHANCED_RESEARCH_BEFORE_VALUATION_WINDOW")
    price_refs = {inputs["target"][key]["sha256"] for key in ["issuer_curve_source", "licensed_history_source"]}
    price_refs.add(inputs["index_source"]["sha256"])
    price_refs.update(
        m[key]["sha256"] for m in inputs["peers"] for key in ["licensed_nav_source", "public_curve_source"]
    )
    for digest, ref in inputs["sources"].items():
        if ref.get("published_on"):
            boundary = research if digest in price_refs else inputs["as_of_date"]
            _require(_iso(ref["published_on"]) <= boundary, "ENHANCED_DOCUMENT_OUTSIDE_KNOWLEDGE_BOUNDARY")
    # Originals沿用旧入口的ISO字符串比较；本入口先严格归一化临时元数据，
    # 原输入与原始公告日期仍原样归档，不让compact/ISO格式改变日期含义。
    checked = {
        **inputs,
        "as_of_date": research,
        "sources": {
            sha: {**ref, **({"published_on": _iso(ref["published_on"])} if ref.get("published_on") else {})}
            for sha, ref in inputs["sources"].items()
        },
    }
    return Originals(checked, Path(root))


def _nav_time_audit(rows, inputs, code):
    """历史净值可晚公告，但必须在研究日已公告；不将它说成月末时已知。"""
    _require(isinstance(rows, list) and rows, "ENHANCED_NAV_DATE_ROWS_REQUIRED")
    research = _iso(inputs.get("research_on"))
    required = set(inputs["calendar"]["dates"])
    late = []
    for row in rows:
        day, announced = _iso(row.get("nav_date")), _iso(row.get("ann_date"))
        _require(day <= inputs["as_of_date"], "ENHANCED_NAV_FUTURE_VALUATION")
        _require(announced <= research, "ENHANCED_NAV_ANNOUNCED_AFTER_RESEARCH")
        if day in required and announced > inputs["as_of_date"]:
            late.append(
                {
                    "fund_code": code,
                    "nav_date": day,
                    "original_ann_date": announced,
                    "research_on": research,
                    "was_known_at_nominal_cutoff": False,
                }
            )
    return late


def _js_value(text, key):
    matches = list(re.finditer(r"var\s+" + re.escape(key) + r"\s*=\s*(.*?);", text, re.S))
    _require(len(matches) == 1, "ENHANCED_PUBLIC_CURVE_SCHEMA:" + key)
    return json.loads(matches[0][1], parse_float=D)


def standard_growth(dates, units, events):
    """构造标准净值增长财富路径；事件缺前一真实交易日或本金不正即拒绝。

    events必须已完成同份额、实施状态、公告截止、日期及现金额规范化。
    仅s<ex<=t的事件影响区间增长，首日事件不重复计入起点。
    """
    _require(len(dates) == len(units) and dates == sorted(set(dates)), "ENHANCED_NAV_CALENDAR")
    _require(all(D(v).is_finite() and D(v) > 0 for v in units), "ENHANCED_NAV_VALUE")
    _require(set(events) <= set(dates), "ENHANCED_DIVIDEND_NOT_ON_REAL_CALENDAR")
    out, proof = [], []
    with localcontext() as ctx:
        ctx.prec = 75
        factor = D(1)
        for i, (day, unit) in enumerate(zip(dates, units, strict=True)):
            unit = D(unit)
            cash = events.get(day, D(0))
            _require(cash.is_finite() and cash >= 0, "ENHANCED_DIVIDEND_CASH_VALUE")
            if i and cash:
                previous = D(units[i - 1])
                _require(previous > cash, "ENHANCED_DIVIDEND_EXCEEDS_PREVIOUS_NAV")
                adjustment = previous / (previous - cash)
                factor *= adjustment
                proof.append(
                    {
                        "ex_date": day,
                        "previous_trading_day": dates[i - 1],
                        "previous_original_unit_nav": str(previous),
                        "cash_per_share": str(cash),
                        "factor": str(adjustment),
                    }
                )
            out.append(unit * factor)
    return out, proof


def _dividends(member, root, days, curve, contract_reviews):
    ref = member["dividend_source"]
    raw = _json(ref, root)
    _require(
        type(raw.get("code")) is int and raw["code"] == 0 and isinstance(raw.get("data"), dict),
        "ENHANCED_DIVIDEND_RESPONSE",
    )
    data = raw["data"]
    _require(
        data.get("fields") == DIVIDEND_FIELDS and isinstance(data.get("items"), list), "ENHANCED_DIVIDEND_FIELD_SCHEMA"
    )
    count = data.get("count")
    _require(
        data.get("has_more") is False
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        and (data["items"] or count == 0)
        and (count == 0 or count == len(data["items"])),
        "ENHANCED_DIVIDEND_PAGE_INCOMPLETE",
    )
    ticker = member["ticker"]
    _require(ref["params"] == {"ts_code": ticker}, "ENHANCED_DIVIDEND_REQUEST_SHARE")
    events, keys, by_ex, originals = {}, set(), {}, []
    for values in data["items"]:
        _require(isinstance(values, list) and len(values) == len(DIVIDEND_FIELDS), "ENHANCED_DIVIDEND_ROW_SCHEMA")
        row = dict(zip(DIVIDEND_FIELDS, values, strict=True))
        _require(row["ts_code"] == ticker, "ENHANCED_DIVIDEND_WRONG_SHARE")
        _require(row["div_proc"] in {"实施", "预案", "股东大会通过", "终止"}, "ENHANCED_DIVIDEND_UNKNOWN_STATUS")
        ex_raw = row["ex_date"]
        if not ex_raw:
            # 未实施或没有除息日期的方案不构成已知现金事件；不能把状态未知的
            # 窗口内正额方案静默当0。若已标实施却缺日期，直接拒绝。
            _require(row["div_proc"] != "实施", "ENHANCED_IMPLEMENTED_DIVIDEND_MISSING_DATE")
            continue
        ex = _iso(ex_raw)
        if not days[0] <= ex <= days[-1]:
            continue
        _require(row["div_proc"] == "实施", "ENHANCED_WINDOW_DIVIDEND_NOT_IMPLEMENTED")
        for key in ["ann_date", "imp_anndate"]:
            stamp = _iso(row[key])
            _require(stamp <= days[-1] and stamp <= ex, "ENHANCED_DIVIDEND_ANNOUNCEMENT_CUTOFF")
        _require(
            row["div_cash"] is not None and not isinstance(row["div_cash"], bool), "ENHANCED_DIVIDEND_CASH_MISSING"
        )
        cash = D(str(row["div_cash"]))
        _require(cash.is_finite() and cash >= 0, "ENHANCED_DIVIDEND_CASH_VALUE")
        record = _iso(row["record_date"])
        pay = _iso(row["pay_date"])
        _require(record <= pay, "ENHANCED_DIVIDEND_RECORD_PAY_ORDER")
        if member["code"] == "502000" and ex == "2025-09-29":
            proof = next(
                (r for r in contract_reviews if r["code"] == "502000" and r["period_end"] == "2025-12-31"), None
            )
            _require(
                proof is not None
                and any(
                    "场内场外12025年9月26日2025年9月29日2025年9月26日0.8000" in s
                    for s in proof.get("dividend_excerpts", [])
                )
                and cash == D(".08")
                and record == "2025-09-26",
                "ENHANCED_LOF_NAV_EX_DATE_REQUIRES_SAME_A_ORIGINAL",
            )
            ex = "2025-09-26"
        # Decimal相等性规范去重：0.08、0.080、8E-2是同一现金金额。
        # 以完整经济事件键判重复，不因原JSON格式或供应商重复行重复计息。
        key = (ticker, ex, cash, record, pay)
        if key in keys:
            continue
        keys.add(key)
        _require(ex not in by_ex or by_ex[ex] == key, "ENHANCED_DIVIDEND_SAME_DATE_CONFLICT")
        by_ex[ex] = key
        events[ex] = cash
        originals.append({"effective_nav_ex_date": ex, "cash_per_share": str(cash), "original_row": row})
    markers = {
        day for day, row in curve.items() if days[0] <= day <= days[-1] and "分红" in (row.get("unitMoney") or "")
    }
    positive = {day for day, cash in events.items() if cash > 0}
    _require(markers == positive, "ENHANCED_DIVIDEND_MARKER_COVERAGE")
    return events, originals


def _compile(inputs, root):
    """返回原始观察节点和复核证据，由核心R5计算器负责换分及部分维度状态。"""
    _require(inputs["version"] == VERSION and inputs["as_of_date"] == "2026-09-30", "ENHANCED_INPUT_VERSION_CUTOFF")
    protocol = _json(inputs["standard_growth_protocol"], root)
    _require(
        protocol["direct_formula"]
        == (
            "G(s,t)=NAV(t)/NAV(s)*PRODUCT[NAV(previous_trading_day)/"
            "(NAV(previous_trading_day)-cash_per_share)]；s<ex_date<=t"
        )
        and protocol["no_imputation"] is True
        and protocol["no_complete_case_deletion"] is True,
        "ENHANCED_STANDARD_GROWTH_PROTOCOL",
    )
    days, months = inputs["calendar"]["dates"], inputs["calendar"]["month_ends"]
    _require(
        len(days) == 727 and len(months) == 37 and days[0] == "2023-09-28" and days[-1] == inputs["as_of_date"],
        "ENHANCED_FIXED_WINDOW",
    )
    _require(days == sorted(set(days)) and months == sorted(set(months)), "ENHANCED_CALENDAR_DUPLICATE")
    expected_months = [max(d for d in days if d[:7] == month) for month in sorted({d[:7] for d in days})]
    _require(months == expected_months, "ENHANCED_MONTH_ENDS_NOT_FROM_REAL_CALENDAR")
    members = inputs["peers"]
    codes = [m["code"] for m in members]
    _require(
        codes == inputs["cohort"]["expected_codes"] and len(codes) == len(set(codes)) == 34, "ENHANCED_FIXED_COHORT"
    )
    _require(not set(codes) & set(inputs["cohort"]["target_aliases"]), "ENHANCED_SELF_PRODUCT_REFERENCE")
    selected = _json(inputs["cohort"]["selection_source"], root)["selection"]
    selected = [
        r["ts_code"][:6]
        for r in selected
        if r["selection_group"] == "ENHANCED_500" and r["ts_code"][:6] not in inputs["cohort"]["target_aliases"]
    ]
    _require(codes == selected, "ENHANCED_SELECTION_SOURCE_MISMATCH")
    reviews = _verify_contracts(inputs, root)
    index = _json(inputs["index_source"], root)
    _require(index["meta"][0]["ts_code"] == "000905.SH", "ENHANCED_COMMON_INDEX_IDENTITY")
    benchmark = {}
    for row in index["rows"]:
        _require(row["ts_code"] == "000905.SH", "ENHANCED_COMMON_INDEX_WRONG_ROW")
        stamp = _iso(row["trade_date"])
        _require(stamp not in benchmark, "ENHANCED_COMMON_INDEX_DUPLICATE")
        benchmark[stamp] = D(str(row["close"]))
    _require([d for d in sorted(benchmark) if days[0] <= d <= days[-1]] == days, "ENHANCED_COMMON_INDEX_CALENDAR")
    original_index = {}
    for ref in index["receipts"]:
        raw = _json(ref, root)
        _require(raw["code"] == 0 and isinstance(raw.get("data"), dict), "ENHANCED_INDEX_ORIGINAL_RESPONSE")
        fields = raw["data"]["fields"]
        if fields == ["ts_code", "trade_date", "close"]:
            for row in raw["data"]["items"]:
                _require(len(row) == 3 and row[0] == "000905.SH", "ENHANCED_INDEX_ORIGINAL_SCHEMA")
                stamp = _iso(row[1])
                _require(stamp not in original_index, "ENHANCED_INDEX_ORIGINAL_DUPLICATE")
                original_index[stamp] = D(str(row[2]))
    _require(all(original_index.get(day) == benchmark[day] for day in days), "ENHANCED_INDEX_DERIVATION_CONFLICT")
    own = inputs["target"]
    _require(own["code"] == inputs["fund_code"] == "014156", "ENHANCED_TARGET_SHARE")
    source = own["issuer_curve_source"]
    _require(json.loads(source["body"]) == {"fundCode": own["code"], "timeType": "now"}, "ENHANCED_ISSUER_QUERY_SHARE")
    raw = _json(source, root)
    _require(
        raw["success"] is True and raw["code"] == 0 and isinstance(raw["result"], list),
        "ENHANCED_ISSUER_CURVE_RESPONSE",
    )
    own_by = {}
    for row in raw["result"]:
        stamp = date.fromisoformat(row["releaseDate"]).isoformat()
        _require(stamp not in own_by, "ENHANCED_ISSUER_CURVE_DUPLICATE")
        own_by[stamp] = row
    target_provider = _json(own["licensed_history_source"], root)
    time_audit = {own["code"]: _nav_time_audit(target_provider["history"]["rows"], inputs, own["code"])}
    _require(target_provider["fund_code"] == own["code"], "ENHANCED_TARGET_PROVIDER_SHARE")
    _require(
        target_provider["source"]["enabled"] is True
        and target_provider["source"]["authorization_verified_at"]
        and target_provider["event_refresh"]["status"] == "SUCCEEDED"
        and target_provider["event_refresh"]["dividends_verified_at"]
        and target_provider["dividend_records"] == [],
        "ENHANCED_TARGET_COMPLETE_EVENT_REFRESH_REQUIRED",
    )
    _, history = provider_history(target_provider, date.fromisoformat(inputs["as_of_date"]))
    _require(history["dates"] == days, "ENHANCED_TARGET_PROVIDER_CALENDAR")
    licensed = dict(zip(history["dates"], history["unit_nav"], strict=True))
    for day in days:
        _require(
            day in own_by and D(own_by[day]["netValue"]) == D(str(licensed[day])),
            "ENHANCED_TARGET_UNIT_SOURCE_CONFLICT",
        )
        _require(
            own_by[day]["netValue"] == own_by[day]["totalNetValue"], "ENHANCED_TARGET_CORPORATE_ACTION_REVIEW_REQUIRED"
        )
    observations = {
        own["code"]: performance(days, months, [D(own_by[d]["netValue"]) for d in days], [benchmark[d] for d in days])
    }
    audits = []
    for member in members:
        code = member["code"]
        text = _read(member["public_curve_source"], root).decode("utf-8")
        _require(_js_value(text, "fS_code") == code, "ENHANCED_PUBLIC_CURVE_WRONG_SHARE")
        curve = {}
        for row in _js_value(text, "Data_netWorthTrend"):
            stamp = datetime.fromtimestamp(int(row["x"]) / 1000, timezone(timedelta(hours=8))).date().isoformat()
            _require(stamp not in curve, "ENHANCED_PUBLIC_CURVE_DUPLICATE")
            curve[stamp] = row
        _require(set(days) <= curve.keys(), "ENHANCED_PUBLIC_CURVE_MISSING_DAY")
        _require(
            not any("拆分" in (curve[d].get("unitMoney") or "") for d in days),
            "ENHANCED_SPLIT_ORIGINAL_FACTOR_REQUIRED",
        )
        raw = _json(member["licensed_nav_source"], root)
        _require(raw["code"] == 0 and isinstance(raw["data"]["fields"], list), "ENHANCED_LICENSED_NAV_RESPONSE")
        _require(raw["data"].get("has_more") is False, "ENHANCED_LICENSED_NAV_INCOMPLETE_PAGE")
        fields = raw["data"]["fields"]
        _require(
            len(fields) == len(set(fields)) and {"ts_code", "ann_date", "nav_date", "unit_nav"} <= set(fields),
            "ENHANCED_LICENSED_NAV_FIELDS",
        )
        units = {}
        original_rows = []
        for row in raw["data"]["items"]:
            _require(len(row) == len(fields), "ENHANCED_LICENSED_NAV_ROW_SCHEMA")
            values = dict(zip(fields, row, strict=True))
            original_rows.append(values)
            _require(values["ts_code"] == member["ticker"], "ENHANCED_LICENSED_NAV_WRONG_SHARE")
            stamp = _iso(values["nav_date"])
            announced = _iso(values["ann_date"])
            _require(announced <= _iso(inputs["research_on"]), "ENHANCED_NAV_ANNOUNCED_AFTER_RESEARCH")
            _require(stamp not in units, "ENHANCED_LICENSED_NAV_DUPLICATE")
            units[stamp] = D(str(values["unit_nav"]))
        time_audit[code] = _nav_time_audit(original_rows, inputs, code)
        _require(set(days) <= units.keys(), "ENHANCED_LICENSED_NAV_MISSING_DAY")
        serializations = []
        for day in days:
            public, unit = D(str(curve[day]["y"])), units[day]
            if public != unit:
                _require(
                    unit == unit.quantize(D(".0001")) and abs(unit - public) <= D(str(math.ulp(float(unit)))),
                    "ENHANCED_UNIT_NAV_CONFLICT",
                )
                serializations.append({"day": day, "public": str(public), "licensed": str(unit)})
        events, originals = _dividends(member, root, days, curve, reviews)
        wealth, proof = standard_growth(days, [units[d] for d in days], events)
        observations[code] = performance(days, months, wealth, [benchmark[d] for d in days])
        audits.append(
            {
                "code": code,
                "dividend_events": originals,
                "growth_adjustments": proof,
                "public_js_serialization_only": serializations,
                "complete_nav_dates": len(days),
            }
        )
    ids = ["CN_PUBLIC_FUND:" + code for code in codes]
    nodes = {}
    refs = sorted(
        {
            own["issuer_curve_source"]["sha256"],
            own["licensed_history_source"]["sha256"],
            inputs["index_source"]["sha256"],
            inputs["cohort"]["selection_source"]["sha256"],
            inputs["contract_review_source"]["sha256"],
            *[
                m[key]["sha256"]
                for m in members
                for key in ["licensed_nav_source", "public_curve_source", "dividend_source"]
            ],
        }
    )
    for name in ["return_1y", "return_3y", "extra_drawdown", "extra_downside"]:
        nodes[name] = {
            "kind": "rank",
            "target": observations[own["code"]][name],
            "direction": "higher" if name.startswith("return") else "lower",
            "expected_products": ids,
            "peers": [{"product_id": "CN_PUBLIC_FUND:" + code, "value": observations[code][name]} for code in codes],
            "source_refs": refs,
        }
    for name, count in [("monthly_success", 36), ("annual_success", 25)]:
        nodes[name] = {
            "kind": "windows",
            "expected_windows": count,
            "source_refs": refs,
            "rows": [
                {
                    "start": row["start"],
                    "end": row["end"],
                    "target": row["target"],
                    "reference": str(median(D(observations[c][name][i]["target"]) for c in codes)),
                }
                for i, row in enumerate(observations[own["code"]][name])
            ],
        }
    return {
        "observations": nodes,
        "audit": {
            "version": VERSION,
            "growth_metric": "STANDARD_NAV_GROWTH_PRE_EX_TRADING_NAV",
            "peer_count": len(codes),
            "sources": refs,
            "values": observations,
            "contract_scope": "FOUR_ACTUAL_PERIODS_REPARSED_FROM_ORIGINALS",
            "contracts": reviews,
            "time_contract": inputs["time_contract"],
            "research_on": inputs["research_on"],
            "nav_late_publication_audit": time_audit,
            "corporate_actions": audits,
        },
        "anchor": None,
    }


PERIODS = {"2023-12-31", "2024-12-31", "2025-12-31", "2026-06-30"}
TARGET_ALIASES = ["014155", "014156", "025007"]
SELECTION_HASH = "1f8728dadf879cc5900bee7c7dd1e3053a9b75de727c1bd1946ddd5b0abb65c0"
GROWTH_PROTOCOL_HASH = "771606ecb96ebca237b1b58ccc09f0b1aba6d5a7763558ba0c4de7ae379a563c"


def _contract_prefix(source, digest, *, page_limit=10):
    """仅重读经济期、身份与合同所在报告前部，不声称再审全部持仓和附注。"""
    cache = getattr(source, "_enhanced_contract_cache", None)
    if cache is None:
        cache = source._enhanced_contract_cache = {}
    key = (digest, page_limit)
    if key not in cache:
        raw = source.raw[digest]
        if raw.startswith(b"%PDF"):
            # 沿项目已有PDFium串行锁，避免在140期报告重复走慢速纯Python提取。
            # 文件已由Originals核摘要且属于不可变sources目录；不缓存来源核验。
            path = source.root / source.sources[digest]["file"]
            pages = text_pages(path, limit=page_limit)
            cache[key] = ("\n".join(pages), len(pages), None)
        else:
            cache[key] = ("\n".join(source.pages(digest)), None, None)
    return cache[key]


def _verify_contracts(inputs, source):
    """每产品四期核实际期末、稳定主代码及同500增强95/5，不能只信审阅JSON。"""
    result, names, masters = [], {}, {}
    members = [inputs["target"], *inputs["peers"]]
    for member in members:
        code = member["code"]
        contracts = member.get("contracts", {})
        _require(
            set(contracts) == PERIODS and len({r["sha256"] for r in contracts.values()}) == 4,
            "ENHANCED_FOUR_ACTUAL_CONTRACTS_REQUIRED:" + code,
        )
        codes = set()
        for period, ref in contracts.items():
            digest = ref["sha256"]
            limit = 10
            text, read_pages, total_pages = _contract_prefix(source, digest, page_limit=limit)
            start, end = report_period(text)
            needed = max(period[:4] + "-01-01", "2023-09-28")
            _require(
                str(end) == period and start.year == int(period[:4]) and str(start) <= needed,
                "ENHANCED_ACTUAL_CONTRACT_PERIOD:" + code + ":" + period,
            )
            plain = normalized_text(text)
            master = re.search(r"基金主代码(?:基金代码)?(\d{6})", plain[:25000])
            name = re.search(r"基金名称(.{1,180}?)基金简称", plain[:25000])
            _require(
                master is not None and name is not None and code in plain[:25000],
                "ENHANCED_CONTRACT_LEGAL_IDENTITY:" + code,
            )
            _require(master[1] == ("014155" if code == "014156" else code), "ENHANCED_LEGAL_MASTER:" + code)
            codes.add(master[1])
            clauses = []
            for match in re.finditer("业绩比较基准", plain[:35000]):
                clause = plain[match.start() : match.start() + 700].split("风险收益特征")[0]
                if (
                    ("中证500指数" in clause or "中证小盘500指数" in clause)
                    and re.search(r"(?<!\d)95%", clause)
                    and re.search(r"(?<!\d)5%", clause)
                    and "存款" in clause
                ):
                    clauses.append(clause)
            _require(
                # 原群按同标的、95/5预算及增强产品准入，基本面选股同样可以
                # 增强，不因缺少“量化”字样擅自删除已冻结的独立成员。
                clauses and "增强" in name[1],
                "ENHANCED_ACTUAL_500_ENHANCED_95_5_ROUTE:" + code,
            )
            published = source.sources[digest].get("published_on")
            _require(published and _iso(published) <= inputs["as_of_date"], "ENHANCED_CONTRACT_PUBLICATION_REQUIRED")
            row = {
                "code": code,
                "period_end": period,
                "source": source.sources[digest],
                "actual_start": str(start),
                "master_code": master[1],
                "legal_name": name[1],
                "benchmark_quote": min(clauses, key=len),
                "contract_pages_read": read_pages,
                "report_pages_total": total_pages,
                "not_entire_report_reparsed": True,
            }
            if code == "502000" and period == "2025-12-31":
                # 该原版年报利润分配详表在物理第47页；前部年度合计没有
                # 场内/场外除息日期，不能由每10份现金合计推断实际净值除息日。
                _require(
                    digest == "9a83c5226e99584f2104bad4bcca3f7ab9754f370d85bc17a569748fc0fc7ce8",
                    "ENHANCED_LOF_DIVIDEND_LAYOUT_UNREVIEWED",
                )
                reader = PdfReader(source.raw[digest])
                detail = normalized_text(reader.pages[46].extract_text() or "")
                a = detail.find("西部利得中证500指数增强A")
                b = detail.find("西部利得中证500指数增强C", a + 1)
                _require(a >= 0 and b > a, "ENHANCED_LOF_A_DIVIDEND_SECTION")
                row["dividend_excerpts"] = [detail[a:b]]
                row["dividend_note_physical_page"] = 47
            result.append(row)
            if period == "2026-06-30":
                names[code], masters[code] = name[1], master[1]
        _require(len(codes) == 1, "ENHANCED_LEGAL_MASTER_CHANGED:" + code)
    _require(
        len(set(names.values())) == len(members) and len(set(masters.values())) == len(members),
        "ENHANCED_DUPLICATE_LEGAL_PRODUCT",
    )
    return result


def compile_observations(inputs, *, root):
    """逐次限目录核字节，再编译六子项；不在此推断重大事项状态或产生分值。"""
    _require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "014156"
        and inputs.get("share_class") == "C"
        and inputs.get("basis") == ACTIVE_BASIS
        and inputs.get("holding_date") == "2026-06-30"
        and inputs.get("as_of_date") == "2026-09-30",
        "ENHANCED_COMPONENT_SCOPE",
    )
    _require(
        inputs["cohort"]["target_aliases"] == TARGET_ALIASES
        and inputs["cohort"]["selection_source"]["sha256"] == SELECTION_HASH
        and inputs["standard_growth_protocol"]["sha256"] == GROWTH_PROTOCOL_HASH,
        "ENHANCED_ORIGINAL_FROZEN_AUTHORITY",
    )
    _require(
        not any(k in inputs for k in ["score", "dimensions", "observations", "quality_metrics", "packet"]),
        "ENHANCED_ORIGINAL_INPUTS_REQUIRED",
    )
    root = Path(root).resolve()
    _originals(inputs, root)
    # 观察中的精确小数已用字符串表达；审计原行的Decimal也保留十进制文本，
    # 使结果可归档为JSON，不把原件的十进制数字转成二进制浮点近似值。
    return json.loads(json.dumps(_compile_cached(canonical(inputs), str(root)), ensure_ascii=False, default=str))


@lru_cache(maxsize=4)
def _compile_cached(serialized, root_name):
    inputs = json.loads(serialized)
    return _compile(inputs, _originals(inputs, Path(root_name)))


def _material_review(inputs, source):
    """核真实014156底稿及其原节；管理人受罚事实与本基金未决事项分别保留。

    SAFE网页是明确标注的工具提取件，只做身份交叉核对。处罚、没收和
    已整改事实另核上市公司2026H1原PDF第67页，不冒充监管独立验收。
    本次只认可已独立审阅的固定底稿；没有该证据或当前原节即拒绝。
    """
    _require(inputs.get("material_event_pending") is False, "ENHANCED_EXPLICIT_MATERIAL_REVIEW_REQUIRED")
    digest = inputs.get("material_event_review_ref")
    _require(
        digest == MATERIAL_REVIEW_HASH and digest in source.sources,
        "ENHANCED_MATERIAL_REVIEW_SOURCE_REQUIRED",
    )
    review = source.json(digest)
    _require(
        review.get("version") == "MATERIAL_EVENT_APPLICABILITY_REVIEW_014156_V1"
        and review.get("fund_code") == "014156"
        and review.get("share_class") == "C"
        and review.get("legal_product_master_code") == "014155"
        and review.get("as_of") == inputs["as_of_date"]
        and review.get("review_closed") is True
        and review.get("material_event_pending") is False,
        "ENHANCED_MATERIAL_REVIEW_SCOPE",
    )
    identity = review["anchors"]["fund_identity"]
    _require(
        identity["legal_name"] == inputs["legal_name"]
        and identity["manager_name"] == inputs["manager_name"]
        and identity["primary_half_sha256"] == inputs["primary_ref"],
        "ENHANCED_MATERIAL_IDENTITY_BINDING",
    )
    registry, evidence = review["source_registry"], review["evidence"]
    _require(
        isinstance(registry, dict) and isinstance(evidence, list) and len(evidence) == 26,
        "ENHANCED_MATERIAL_REVIEW_EVIDENCE_SCHEMA",
    )
    _require(
        isinstance(review.get("events"), list) and len(review["events"]) == 23,
        "ENHANCED_MATERIAL_REVIEW_EVENTS_REQUIRED",
    )
    for sha, ref in registry.items():
        _require(sha == ref["sha256"] and sha in source.raw, "ENHANCED_MATERIAL_ORIGINAL_NOT_ARCHIVED")
    keys, editorial_notes = set(), []
    for item in evidence:
        sha = item["source_hash"]
        _require(item["key"] not in keys and sha in registry, "ENHANCED_MATERIAL_EVIDENCE_SCOPE")
        keys.add(item["key"])
        ref = registry[sha]
        if "observed_facts" in item:
            # 工具事实必须与已归档的实际工具提取JSON一致；不是HTML全文。
            receipt = source.json(sha)
            _require(
                item.get("role") == "OFFICIAL_PAGE_TOOL_EXTRACTION_CROSS_CHECKED_WITH_LISTED_ISSUER_ORIGINAL_PDF"
                and receipt["pages"][item["key"]] == item["observed_facts"],
                "ENHANCED_OFFICIAL_TOOL_RECEIPT_CONFLICT",
            )
            continue
        _require(
            item.get("quote")
            and item.get("published_on")
            and _iso(item["published_on"]) <= inputs["as_of_date"]
            and ref.get("published_on") == item["published_on"],
            "ENHANCED_MATERIAL_EVIDENCE_SCOPE",
        )
        if item.get("pdf_page"):
            reader = PdfReader(source.raw[sha])
            page = int(item["pdf_page"])
            _require(1 <= page <= len(reader.pages), "ENHANCED_MATERIAL_PDF_PAGE")
            body = normalized_text(reader.pages[page - 1].extract_text() or "")
            _require(normalized_text(item["quote"]) in body, "ENHANCED_MATERIAL_ORIGINAL_QUOTE")
        else:
            # 原底稿对五份长法律文件保留原文前节，并把“完整原文见...”
            # 作为换行后的研究注附在quote末尾。只分离这个固定、已审阅的
            # 注记；前节必须仍逐字命中完整原件，不能把研究注当发行人原话。
            note = "\n（完整原文见同哈希原件；未将省略部分改写为结论）"
            quote = item["quote"]
            if quote.endswith(note):
                quote = quote.removesuffix(note)
                editorial_notes.append(item["key"])
            source.claim({"source_hash": sha, "quote": quote})
    for event in review["events"]:
        _require(
            event.get("source_hashes")
            and set(event["source_hashes"]) <= registry.keys()
            and event.get("unresolved") is False
            and event.get("affects_selected_performance_release") in {False, None}
            and _iso(event.get("verified_disclosed_on") or event.get("announced_on")) <= inputs["as_of_date"],
            "ENHANCED_MATERIAL_EVENT_PENDING_OR_UNSOURCED",
        )
    # 当前管理/托管履职与10.6.1原节绑定真实管理输入；H1无处罚陈述
    # 只对应其报告期间，不能覆盖8月新增处罚。管理状态必须保留整改观察。
    management_sha = inputs.get("material_management_review_ref")
    _require(management_sha in source.raw, "ENHANCED_CURRENT_MATERIAL_SECTIONS_REQUIRED")
    management = source.json(management_sha)
    governance = management["governance"]
    current = [s for s in governance["statements"] if s["period_end"] == inputs["holding_date"]]
    _require(
        management["manager_name"] == inputs["manager_name"]
        and management["as_of_date"] == inputs["as_of_date"]
        and governance["state"] == "REMEDIATION_DISCLOSED"
        and governance["negative_event_review_complete"] is True
        and len(current) == 1
        and current[0]["source_hash"] == inputs["primary_ref"]
        and governance["trustee_execution"]["source_hash"] == inputs["primary_ref"],
        "ENHANCED_CURRENT_MATERIAL_SECTIONS_REQUIRED",
    )
    for item in [current[0], governance["trustee_execution"]]:
        source.claim(item)
    actual_events = governance.get("known_applicable_events", [])
    _require(len(actual_events) == 1, "ENHANCED_MANAGER_PENALTY_MUST_BE_PRESERVED")
    _require(
        actual_events[0]["source_hash"] == review["anchors"]["remediation_disclosure"]["source_hash"]
        and actual_events[0]["date"] == "2026-08-17",
        "ENHANCED_MANAGER_PENALTY_MUST_BE_PRESERVED",
    )
    return {
        "source_hash": digest,
        "known_events_preserved": len(review["events"]),
        "governance_not_reclassified": True,
        "manager_governance_state": "REMEDIATION_DISCLOSED",
        "material_management_review_ref": management_sha,
        "official_page_receipt_is_downloaded_html": False,
        "research_note_separated_from_original_quote": editorial_notes,
        "independent_regulatory_acceptance": False,
    }


def calculate(inputs, *, root):
    """标准R5仅重放收益、风险、稳定性；重大事项未知拒绝，完整总分恒空。"""
    source = _originals(inputs, Path(root))
    event_review = _material_review(inputs, source)
    compiled = compile_observations(inputs, root=root)
    registry = {h: {**ref, "path": str(Path(root) / ref["file"])} for h, ref in source.sources.items()}
    packet = {
        "fund_code": "014156",
        "basis": ACTIVE_BASIS,
        "as_of_date": inputs["as_of_date"],
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_metrics": {
            "target_product_aliases": ["CN_PUBLIC_FUND:" + c for c in TARGET_ALIASES],
            "material_event_pending": inputs["material_event_pending"],
            "observations": compiled["observations"],
        },
        "r4_evidence_registry": registry,
        "raw_archive": [{"kind": r.get("archive_kind", "ORIGINAL"), "sha256": h} for h, r in registry.items()],
        "quality_public": {"dataDates": {"nav": "2026-09-30", "holdings": "2026-06-30"}},
    }
    result = _calculate_frozen(packet, PROTOCOL, {}, selected_dimensions={"return", "risk", "stability"})
    _require(all(result.get(k) is None for k in ["score", "raw_score", "grade"]), "ENHANCED_COMPONENT_TOTAL_LEAK")
    return {
        "dimensions": result["dimensions"],
        "observations": compiled["observations"],
        "complete_score": None,
        "grade": None,
        "audit": {**compiled["audit"], "material_event_review": event_review},
        "standard_replay": result,
    }


def check_binding(bundle, inputs):
    """prepare与快照共用本期本C份额、管理人及真实主报告绑定。"""
    _require(
        bundle.get("family") == "ACTIVE_EQUITY"
        and bundle.get("fund_code") == inputs.get("fund_code")
        and bundle.get("share_class") == inputs.get("share_class")
        and str(bundle.get("as_of_date")) == inputs.get("as_of_date")
        and str(bundle.get("holding_date")) == inputs.get("holding_date")
        and bundle["identity"]["legal_name"] == inputs.get("legal_name")
        and bundle["identity"]["manager_name"] == inputs.get("manager_name")
        and bundle["original_report_archive"]["sha256"] == inputs.get("primary_ref"),
        "ENHANCED_CURRENT_BINDING",
    )


def prepare(session, request, *, root):
    """候选prepare保留最新其他五维，新增三维仍作为部分评分追加包。"""
    from datetime import UTC

    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json, canonical_source
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    code, raw = request["fund_code"], deepcopy(request["inputs"])
    previous = current_inputs(session, [code]).get(code)
    _require(previous and str(previous["as_of_date"]) == request["as_of_date"], "ENHANCED_CURRENT_PERIOD_REQUIRED")
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
                "stability_calculation",
                "gold_components_calculation",
            ]
        ),
        "ENHANCED_PARTIAL_SNAPSHOT_REQUIRED",
    )
    for h, ref in raw["sources"].items():
        raw["sources"][h] = canonical_source(root, ref)
    result = calculate(raw, root=root)
    doc = bundle["reviewed_document"]
    for key in ["return", "risk", "stability"]:
        explanation = request["explanations"][key].strip()
        _require(len(explanation) >= 40, "ENHANCED_EXPLANATION_REQUIRED")
        point = result["dimensions"][key]
        doc["dimensions"][key] = {"lower": point, "upper": point, "candidate": point, "reason": explanation}
        bundle.setdefault("public_explanations", {})[key] = explanation
        bundle.get("public_metrics", {}).pop(key, None)
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, doc),
        enhanced_performance_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    known = {r["sha256"] for r in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(ref for h, ref in raw["sources"].items() if h not in known)
    validate_sources(bundle, datetime.now(UTC), root=root)
    checked = snapshot_calculate(bundle, root=root)
    _require(checked["status"] == "PARTIAL" and checked["score"] is None, "ENHANCED_CANNOT_CLAIM_COMPLETE")
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}
