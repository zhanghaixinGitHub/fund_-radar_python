"""007950广谱主动收益、稳定性两个维度的实际原件编译候选。

仅依赖正式服务的Originals及冻结R5。调用者必须提供sources目录内的
全部实际计算原件；旧研究路径仅可作为来源说明，绝不读取或导入。
参照准入属于冻结审阅契约；本模块从实际净值、分红与定报重新编译
观察，不接受分数、收益、状态或已算好的观察作为计算输入。
"""

import json
import re
from collections import OrderedDict
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal as D
from decimal import localcontext
from fractions import Fraction as F
from pathlib import Path

from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS

VERSION = "R5_ACTIVE_BROAD_007950_ORIGINALS_20261030_2"
SELECTION_HASH = "44dbb31c04b0f58844ba61a7270c2741742a28698933b06644716ade2c491fe8"
MATERIAL_HASH = "1bf32b91e477aeabe47a9ecce87549f9ae83ea485a6183259037528812b8d09a"
CODE = "007950"
ALIASES = ["007950", "001917", "CN_PUBLIC_FUND:001917"]
SELECTED = {"return", "stability"}
DOCUMENT_KEYS = {"return": "return", "stability": "stability"}
TIME_CONTRACT = {
    "kind": "RETROSPECTIVE_HISTORICAL_QUALITY_RESEARCH",
    "valuation_window_end": "as_of_date",
    "nav_and_market_publication_boundary": "research_on",
    "fundamental_disclosure_boundary": "as_of_date",
    "not_historical_asof_tradable_signal": True,
}


def require(ok, code):
    if not ok:
        raise ValueError("ACTIVE_BROAD_" + code)


def iso(value):
    require(isinstance(value, str) and bool(re.fullmatch(r"\d{8}|\d{4}-\d{2}-\d{2}", value)), "DATE_SCHEMA")
    return date.fromisoformat(value).isoformat()


def decimal(value):
    return D(value.numerator) / D(value.denominator)


def json_ready(value):
    """审计中的原始数字精确转字符串，计算完毕后才转换，不经float损失精度。"""
    if isinstance(value, (D, F)):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(child) for child in value]
    return value


def raw_json(source, digest):
    require(digest in source.raw, "UNARCHIVED_ORIGINAL")
    return json.loads(source.raw[digest], parse_float=D)


def originals(inputs, root):
    """每次重新验所有字节；价格公告可晚于估值日，基础事实仍限9月末。"""
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == CODE
        and inputs.get("share_class") == "C"
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("holding_date") == "2026-06-30",
        "VERSION_OR_TARGET_SHARE",
    )
    require(
        inputs.get("time_contract") == TIME_CONTRACT and inputs.get("research_on") == "2026-10-29",
        "EXPLICIT_TIME_CONTRACT",
    )
    require(set(inputs.get("selected_dimensions", [])) == SELECTED, "EXPLICIT_TWO_COMPONENT_SCOPE")
    require(
        not any(k in inputs for k in ["score", "dimensions", "observations", "packet", "quality_metrics", "returns"]),
        "PRECOMPUTED_INPUT_FORBIDDEN",
    )
    price_refs = set(inputs["target"]["licensed_nav_refs"]) | {
        inputs["target"]["issuer_nav_ref"],
        inputs["target"]["issuer_page_ref"],
    }
    price_refs.update(h for m in inputs["peers"] for h in m["nav_refs"])
    checked = {}
    for h, ref in inputs["sources"].items():
        item = deepcopy(ref)
        if ref.get("published_on"):
            stamp = iso(ref["published_on"])
            limit = inputs["research_on"] if h in price_refs else inputs["as_of_date"]
            require(stamp <= limit, "DOCUMENT_KNOWLEDGE_BOUNDARY")
            item["published_on"] = stamp
        checked[h] = item
    return Originals({**inputs, "as_of_date": inputs["research_on"], "sources": checked}, Path(root))


def _table(source, digest, identity):
    raw = raw_json(source, digest)
    require(
        type(raw.get("code")) is int and raw["code"] == 0 and isinstance(raw.get("data"), dict), "PROVIDER_RESPONSE"
    )
    fields = raw["data"]["fields"]
    items = raw["data"]["items"]
    require(
        isinstance(fields, list)
        and len(fields) == len(set(fields))
        and isinstance(items, list)
        and "ts_code" in fields
        and raw["data"].get("has_more") is False,
        "PROVIDER_SCHEMA_OR_OPEN_PAGE",
    )
    rows = [dict(zip(fields, row, strict=True)) for row in items]
    require(all(r["ts_code"] == identity for r in rows), "WRONG_SHARE_ORIGINAL_ROW")
    return rows


def _event_receipt(inputs, source, member):
    """旧实际收藏回执绑定代码、原回包和原行；没有HTTP参数就如实保留其边界。"""
    receipt = raw_json(source, member["dividend_receipt_ref"])
    require(
        receipt["code"] == member["code"] and {r["sha256"] for r in receipt["sources"]} == set(member["dividend_refs"]),
        "DIVIDEND_RECEIPT_IDENTITY",
    )
    observed = [r for h in member["dividend_refs"] for r in _table(source, h, member["provider_share_id"])]
    require(receipt["rows"] == observed, "DIVIDEND_RECEIPT_ORIGINAL_ROWS")
    require(
        all(
            r.get("status", r.get("http_status")) == 200 and iso(r["received_at"][:10]) <= inputs["research_on"]
            for r in receipt["sources"]
        ),
        "DIVIDEND_RECEIPT_SUCCESS_AND_RESEARCH_DATE",
    )
    return {
        "receipt_ref": member["dividend_receipt_ref"],
        "original_refs": member["dividend_refs"],
        "empty": not observed,
        "http_params_archived": any("params" in r for r in receipt["sources"]),
        "identity_route": "ACTUAL_COLLECTION_RECEIPT_CODE_RESPONSE_HASH_ROWS",
    }


def _calendar(inputs, source):
    """由实际静态官方休市定义重建交易日，既不借净值有无也不猜工作日。"""
    days = []
    for digest in inputs["calendar"]["source_refs"]:
        definition = raw_json(source, digest)
        closed = set()
        for year in definition["years"]:
            for start, end in year["closed_ranges"]:
                a, b = date.fromisoformat(start), date.fromisoformat(end)
                closed.update(a + timedelta(days=i) for i in range((b - a).days + 1))
        start, end = date.fromisoformat(definition["coverage_start"]), date.fromisoformat(definition["coverage_end"])
        for i in range((end - start).days + 1):
            day = start + timedelta(days=i)
            if day.weekday() < 5 and day not in closed and date(2023, 9, 28) <= day <= date(2026, 9, 30):
                days.append(day.isoformat())
    require(
        days == sorted(set(days)) and len(days) == 727 and days == inputs["calendar"]["dates"], "FIXED_REAL_CALENDAR"
    )
    months = list({d[:7]: d for d in days}.values())
    require(len(months) == 37 and months == inputs["calendar"]["month_ends"], "ACTUAL_MONTH_ENDS")
    exchange = raw_json(source, inputs["calendar"]["exchange_ref"])
    require(
        type(exchange.get("code")) is int and exchange["code"] == 0 and exchange["data"].get("has_more") is False,
        "INDEPENDENT_CALENDAR_RESPONSE",
    )
    fields = exchange["data"]["fields"]
    require(len(fields) == len(set(fields)) and {"exchange", "cal_date", "is_open"} <= set(fields), "CALENDAR_SCHEMA")
    rows = [dict(zip(fields, row, strict=True)) for row in exchange["data"]["items"]]
    rows = [r for r in rows if "2023-09-28" <= iso(r["cal_date"]) <= "2026-09-30"]
    require(
        all(r["exchange"] == "SSE" and type(r["is_open"]) is int and r["is_open"] in (0, 1) for r in rows),
        "CALENDAR_EXCHANGE_OR_FLAG",
    )
    require(
        len({r["cal_date"] for r in rows}) == len(rows) == (date(2026, 9, 30) - date(2023, 9, 28)).days + 1,
        "CALENDAR_ALL_ACTUAL_DATES_REQUIRED",
    )
    require(sorted(iso(r["cal_date"]) for r in rows if r["is_open"] == 1) == days, "INDEPENDENT_CALENDAR_CONFLICT")
    return days, months


def _nav(source, refs, identity, days, research):
    points = {}
    announcements = {}
    late = []
    for digest in refs:
        rows = _table(source, digest, identity)
        require(all(k in r for r in rows for k in ["nav_date", "ann_date", "unit_nav"]), "NAV_FIELDS")
        for row in rows:
            day = iso(row["nav_date"])
            require(day <= days[-1], "NAV_VALUATION_AFTER_CUTOFF")
            if day not in days:
                continue
            announced = iso(row["ann_date"])
            require(day <= days[-1] and announced <= research, "NAV_PUBLICATION_NOT_AVAILABLE")
            nav = F(str(row["unit_nav"]))
            require(nav > 0, "NAV_NONPOSITIVE")
            require(day not in points or points[day] == nav, "NAV_CONFLICT")
            points[day] = nav
            announcements[day] = announced
            if announced > days[-1]:
                late.append({"nav_date": day, "original_ann_date": announced, "known_on_valuation_cutoff": False})
    return points, announcements, late


def _dividends(source, refs, identity, days, research):
    """同一实施事件按真实原行去重；LOF采用明确净值生效日，保留名义除息日。"""
    groups = {}
    for digest in refs:
        for row in _table(source, digest, identity):
            require(all(k in row for k in ["div_proc", "ex_date", "ann_date", "div_cash"]), "DIVIDEND_FIELDS")
            require(row["div_proc"] in {"实施", "预案", "取消", "终止", "未实施"}, "DIVIDEND_STATUS_UNKNOWN")
            if row["div_proc"] != "实施":
                continue
            require(row["ex_date"], "IMPLEMENTED_DIVIDEND_EX_DATE_REQUIRED")
            nominal = iso(row["ex_date"])
            net = iso(row["net_ex_date"]) if row.get("net_ex_date") else None
            day = net or nominal
            if not days[0] < day <= days[-1]:
                continue
            require(
                iso(row["ann_date"]) <= days[-1] and iso(row.get("imp_anndate")) <= days[-1],
                "DIVIDEND_IMPLEMENTATION_NOT_AVAILABLE",
            )
            cash = F(str(row["div_cash"]))
            require(cash > 0, "DIVIDEND_NONPOSITIVE")
            key = (nominal, cash)
            groups.setdefault(key, []).append((net, row, digest))
    cash_by = {}
    proof = []
    for key, rows in groups.items():
        nets = {net for net, _, _ in rows if net}
        require(len(nets) <= 1, "DIVIDEND_NET_DAY_CONFLICT")
        day = next(iter(nets)) if nets else key[0]
        cash = key[1]
        require(day in days and (day not in cash_by or cash_by[day] == cash), "DIVIDEND_DAY_OR_AMOUNT_CONFLICT")
        cash_by[day] = cash
        proof.append(
            {
                "effective_nav_day": day,
                "nominal_ex_day": key[0],
                "cash_per_share": str(cash),
                "original_rows": [{**r, "source_hash": h} for _, r, h in rows],
            }
        )
    return cash_by, proof


def _growth(points, cash, days, start, end, allow_missing=False):
    if start not in points or end not in points:
        require(allow_missing, "REQUIRED_NAV_ENDPOINT_MISSING")
        return None
    growth = points[end] / points[start]
    for day, value in cash.items():
        if start < day <= end:
            previous = days[days.index(day) - 1]
            require(previous in points and points[previous] > value, "DIVIDEND_PREVIOUS_NAV_MISSING")
            growth *= points[previous] / (points[previous] - value)
    return growth - 1


def _target(inputs, source, days):
    target = inputs["target"]
    points, ann, late = _nav(source, target["licensed_nav_refs"], CODE + ".OF", days, inputs["research_on"])
    require(set(points) == set(days), "TARGET_REQUIRED_DAY_MISSING")
    issuer = {}
    for digest, paged in [(target["issuer_nav_ref"], False), (target["issuer_page_ref"], True)]:
        raw = raw_json(source, digest)
        require(raw.get("code") == 0, "ISSUER_NAV_RESPONSE")
        rows = raw["data"]["list"] if paged else raw["data"]
        if paged:
            require(raw["data"]["total"] == len(rows), "ISSUER_NAV_PAGE_NOT_CLOSED")
        observed = {}
        for row in rows:
            require(row["productCode"] == CODE, "ISSUER_WRONG_SHARE")
            day = iso(row["navDate"])
            require(day not in observed, "ISSUER_NAV_DUPLICATE")
            observed[day] = F(str(row["relatePrice"]))
            require(
                F(str(row["adjustNav"])) == observed[day] == F(str(row["cumulativeNet"])),
                "TARGET_UNRECONCILED_CORPORATE_ACTION",
            )
        require(
            set(days) <= observed.keys() and all(observed[d] == points[d] for d in days),
            "OWN_NAV_DOUBLE_SOURCE_CONFLICT",
        )
        if issuer:
            require(issuer == observed, "ISSUER_TWO_ENDPOINTS_CONFLICT")
        issuer = observed
    for key in ["dividend_ref", "split_ref"]:
        raw = raw_json(source, target[key])
        require(raw.get("code") == 0 and raw.get("data") == [], "TARGET_CORPORATE_ACTION_REVIEW_CHANGED")
        receipt = raw_json(source, target[key.replace("_ref", "_receipt_ref")])
        endpoint = "getFundMelonmd" if key == "dividend_ref" else "getFundSplit"
        require(
            receipt["params"] == {"productCode": CODE}
            and receipt["identity"] == CODE + "-C"
            and receipt["sha256"] == target[key]
            and receipt["url"].endswith("/" + endpoint)
            and receipt["status"] == 200
            and iso(receipt["acquired_at"][:10]) <= inputs["research_on"],
            "TARGET_EMPTY_EVENT_REAL_REQUEST_BINDING",
        )
        # 两个接口的真实空响应字节相同，不能只凭共同响应SHA推断各自请求身份。
        # 分别保留原始采集回执，由真实接口和productCode绑定到本C份额。
        receipt = raw_json(source, target[key.replace("_ref", "_receipt_ref")])
        endpoint = "getFundMelonmd" if key == "dividend_ref" else "getFundSplit"
        require(
            receipt.get("url") == "https://www.cmfchina.com/ws-business-server/fund/" + endpoint
            and receipt.get("params") == {"productCode": CODE}
            and receipt.get("identity") == CODE + "-C"
            and receipt.get("status") == 200
            and receipt.get("sha256") == target[key]
            and receipt.get("acquired_at"),
            "TARGET_CORPORATE_ACTION_REQUEST_BINDING",
        )
    return points, {}, {"announcement_dates": ann, "later_announced": late, "double_source_sessions": len(days)}


def _selection(inputs, source):
    from app.services.fund_rating_active_broad_admission import compile_admission

    require(inputs["selection_ref"] == SELECTION_HASH, "UNREVIEWED_REFERENCE_VERSION")
    historical = raw_json(source, SELECTION_HASH)
    review = compile_admission(inputs, source)
    require(
        historical["target"] == CODE and historical["as_of"] == inputs["as_of_date"],
        "REFERENCE_SCOPE",
    )
    require(
        historical["selection_before_new_scores"] is True and historical["no_complete_case_deletion"] is True,
        "REFERENCE_OUTCOME_SELECTION",
    )
    ret, stable = review["return_codes"], review["stability_codes"]
    require(
        ret == sorted(set(ret))
        and stable == sorted(set(stable))
        and set(stable) <= set(ret)
        and not set(ret) & set(ALIASES),
        "FIXED_PRODUCT_REFERENCE",
    )
    members = inputs["peers"]
    require([m["code"] for m in members] == ret, "REFERENCE_MEMBER_CHANGED")
    require(
        all(
            m["provider_share_id"].split(".")[0] == m["code"]
            and m["provider_share_id"].split(".")[1] in ["OF", "SZ", "SH"]
            for m in members
        ),
        "REFERENCE_SHARE_ID",
    )
    # 旧名单只保存历史对照；新名单必须由全505原件重编产生，新增成员也不能漏掉。
    review["historical_manifest_comparison"] = {
        kind: {
            "added": sorted(set(review[kind + "_codes"]) - set(historical[kind + "_codes"])),
            "removed": sorted(set(historical[kind + "_codes"]) - set(review[kind + "_codes"])),
        }
        for kind in ("return", "stability")
    }
    review["stability_missing_sessions"] = historical["stability_missing_sessions"]
    return review


def _material_review(inputs, source):
    """False来自固定逐事项审阅；监管整改、申购上限及基准调整均不能被抹去。"""
    require(inputs.get("material_event_pending") is False, "EXPLICIT_MATERIAL_REVIEW_REQUIRED")
    require(inputs.get("material_review_ref") == MATERIAL_HASH, "UNREVIEWED_MATERIAL_VERSION")
    review = raw_json(source, MATERIAL_HASH)
    require(
        review["fund_code"] == CODE
        and review["share_class"] == "C"
        and review["primary_ref"] == inputs["primary_ref"]
        and review["manager_name"] == inputs["manager_name"]
        and review["as_of"] == inputs["as_of_date"]
        and review["review_closed"] is True
        and review["material_event_pending"] is False,
        "MATERIAL_REVIEW_SCOPE",
    )
    catalog = raw_json(source, review["catalog_ref"])
    require(catalog["code"] == 0, "MATERIAL_CATALOG_RESPONSE")
    data = catalog["data"]
    listed = data["list"]
    require(
        data["pageNum"] == 1 and len(listed) == data["pageSize"] and min(r["pubDate"] for r in listed) < "2026-01-01",
        "MATERIAL_CATALOG_COVERAGE",
    )
    selected = [r for r in listed if "2026-01-01" <= r["pubDate"] <= inputs["as_of_date"]]
    events = review["events"]
    require(
        len(events) == 20 and {e["id"] for e in events} == {r["adId"] for r in selected},
        "MATERIAL_EVENT_CATALOG_NOT_CLOSED",
    )
    for event in events:
        row = next(r for r in selected if r["adId"] == event["id"])
        require(
            row["productId"] == CODE and row["pubDate"] == event["published_on"] and row["title"] == event["title"],
            "MATERIAL_EVENT_IDENTITY",
        )
        require(
            event["unresolved"] is False
            and event["affects_selected_release"] is False
            and len(event["judgment"]) >= 30
            and event["claims"],
            "UNKNOWN_OR_UNSOURCED_EVENT",
        )
        for claim in event["claims"]:
            source.claim(claim)
    governance = review["governance"]
    require(
        governance["state"] == "REMEDIATION_DISCLOSED"
        and len(governance["known_applicable_events"]) == 1
        and governance["known_applicable_events"][0]["date"] == "2025-10-27",
        "MANAGER_REMEDIATION_MUST_BE_PRESERVED",
    )
    for claim in governance["statements"] + [governance["trustee_execution"]] + governance["known_applicable_events"]:
        source.claim(claim)
    require(review["primary_material_claims"], "PRIMARY_MATERIAL_SECTIONS_REQUIRED")
    for claim in review["primary_material_claims"]:
        require(claim["source_hash"] == inputs["primary_ref"], "PRIMARY_MATERIAL_BINDING")
        source.claim(claim)
    classified = raw_json(source, inputs["material_event_classification_ref"])
    require(
        classified["fund_code"] == CODE
        and classified["as_of_date"] == inputs["as_of_date"]
        and classified["review_ref"] == MATERIAL_HASH
        and classified["review_closed"] is True,
        "MATERIAL_CLASSIFICATION_BINDING",
    )
    fund_events = classified["events"]
    require(
        len(fund_events) == len(events) + len(governance["known_applicable_events"])
        and len({e["id"] for e in fund_events}) == len(fund_events),
        "MATERIAL_CLASSIFICATION_COMPLETENESS",
    )
    by_id = {e["id"]: e for e in fund_events}
    for event in events:
        typed = by_id[event["id"]]
        require(
            typed["source_refs"] == sorted({event["source_hash"], *[c["source_hash"] for c in event["claims"]]})
            and typed["published_on"] == event["published_on"]
            and typed["judgment"] == event["judgment"]
            and typed["applicability"] == "FUND_CONFIRMED",
            "MATERIAL_CLASSIFICATION_EVENT_ORIGINALS",
        )
        require((typed["status"] == "UNRESOLVED") is event["unresolved"], "MATERIAL_UNRESOLVED_STATE_CONFLICT")
    for number, event in enumerate(governance["known_applicable_events"]):
        typed = by_id["MANAGER_GOVERNANCE_" + str(number)]
        require(
            typed["applicability"] == "MANAGER_ONLY"
            and typed["status"] == governance["state"]
            and typed["severity"] == "GOVERNANCE_MEASURE"
            and typed["source_refs"] == [event["source_hash"]]
            and typed["event_date"] == event["date"],
            "MANAGER_GOVERNANCE_ORIGINAL_BINDING",
        )
    for typed in fund_events:
        require(
            typed["applicability"] in {"FUND_CONFIRMED", "MANAGER_ONLY"}
            and typed["status"] in {"NO_UNRESOLVED_FUND_FAILURE_IDENTIFIED", "UNRESOLVED", "REMEDIATION_DISCLOSED"}
            and typed["severity"] in {"NOT_MATERIAL", "MATERIAL", "SEVERE", "GOVERNANCE_MEASURE"}
            and typed["source_refs"]
            and set(typed["source_refs"]) <= source.raw.keys(),
            "MATERIAL_TYPED_EVENT_SCHEMA",
        )
    pending = any(
        e["applicability"] == "FUND_CONFIRMED"
        and e["status"] == "UNRESOLVED"
        and e["severity"] in {"MATERIAL", "SEVERE"}
        for e in fund_events
    )
    require(
        pending is review["material_event_pending"] and pending is inputs["material_event_pending"],
        "SOURCE_DERIVED_FUND_MATERIAL_PENDING_CONFLICT",
    )
    require(not pending, "FUND_CONFIRMED_UNRESOLVED_MATERIAL_EVENT")
    return {
        "source_hash": MATERIAL_HASH,
        "reviewed_catalog_events": len(events),
        "governance_state": governance["state"],
        "known_penalty_preserved": True,
        "material_event_pending": pending,
        "material_event_classification_ref": inputs["material_event_classification_ref"],
        "material_events": fund_events,
        "known_business_restrictions_and_contract_changes": [
            deepcopy(e)
            for e in events
            if any(term in e["title"] for term in ("限制", "限额", "大额", "暂停", "转换", "基准", "修订"))
        ],
        "scope": "本基金本期公开事项及六期管理声明；不替代持股公司逐家质量审阅",
    }


def _compile(inputs, source):
    review = _selection(inputs, source)
    days, months = _calendar(inputs, source)
    own, own_cash, own_time = _target(inputs, source, days)
    funds = {}
    time_audit = {CODE: own_time}
    event_audit = {}
    refs = set()
    for member in inputs["peers"]:
        code = member["code"]
        identity = member["provider_share_id"]
        points, ann, late = _nav(source, member["nav_refs"], identity, days, inputs["research_on"])
        receipt_audit = _event_receipt(inputs, source, member)
        cash, events = _dividends(source, member["dividend_refs"], identity, days, inputs["research_on"])
        if code in review["stability_codes"]:
            require(
                sorted(set(days) - points.keys()) == review["stability_missing_sessions"].get(code, []),
                "STABILITY_ORIGINAL_AVAILABILITY_CHANGED",
            )
        funds[code] = (points, cash)
        event_audit[code] = events
        time_audit[code] = {
            "later_announced": late,
            "nav_rows": len(points),
            "dividend_request_identity": receipt_audit,
        }
        refs.update(member["nav_refs"] + member["dividend_refs"])
    refs.update(
        inputs["target"]["licensed_nav_refs"]
        + [
            inputs["target"]["issuer_nav_ref"],
            inputs["target"]["issuer_page_ref"],
            inputs["target"]["dividend_ref"],
            inputs["target"]["split_ref"],
        ]
    )
    reference_ids = ["FROZEN_ACTIVE_PRODUCT_REP:" + c for c in review["return_codes"]]
    nodes = {}
    true_returns = {}
    with localcontext() as context:
        context.prec = 70
        for code, (points, cash) in {**funds, CODE: (own, own_cash)}.items():
            one = _growth(points, cash, days, "2025-09-30", days[-1])
            three = _growth(points, cash, days, days[0], days[-1])
            true_returns[code] = {
                "return_1y": str(decimal(one)),
                "return_3y": str(
                    (1 + decimal(three))
                    ** (D("365.25") / D((date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days))
                    - 1
                ),
            }
        for kind in ["return_1y", "return_3y"]:
            nodes[kind] = {
                "kind": "rank",
                "direction": "higher",
                "target": true_returns[CODE][kind],
                "peers": [
                    {"product_id": pid, "value": true_returns[c][kind]}
                    for pid, c in zip(reference_ids, review["return_codes"], strict=True)
                ],
                "expected_products": reference_ids,
                "source_refs": sorted(refs),
            }
        windows = {}
        bounds = {"lower": {}, "upper": {}}
        for kind, intervals, unit in [
            ("monthly_success", [(months[i], months[i + 1]) for i in range(36)], F(1, 120)),
            ("annual_success", [(months[i], months[i + 12]) for i in range(25)], F(1, 10)),
        ]:
            rows = []
            for start, end in intervals:
                known = []
                missing = []
                for code in review["stability_codes"]:
                    value = _growth(*funds[code], days, start, end, allow_missing=True)
                    (missing if value is None else known).append(code if value is None else value)
                known.sort()
                n = len(review["stability_codes"])
                k = len(missing)
                middle = [n // 2 - 1, n // 2]
                require(min(middle) >= k and len(known) > max(middle), "MEDIAN_BOUNDS_NOT_FINITE")
                low = sum(known[i - k] for i in middle) / 2
                high = sum(known[i] for i in middle) / 2
                target = _growth(own, own_cash, days, start, end)

                def point(gap, threshold=unit):
                    return 100 if gap > threshold else 50 if gap < -threshold else 75

                minimum, maximum = point(target - high), point(target - low)
                require(minimum == maximum, "STABILITY_WINDOW_NOT_IDENTIFIED")
                rows.append(
                    {
                        "start": start,
                        "end": end,
                        "target_ratio": str(decimal(target)),
                        "reference_lower_ratio": str(decimal(low)),
                        "reference_upper_ratio": str(decimal(high)),
                        "target_exact": str(target),
                        "reference_lower_exact": str(low),
                        "reference_upper_exact": str(high),
                        "missing_peer_returns": missing,
                        "quality_minimum": minimum,
                        "quality_maximum": maximum,
                    }
                )
            windows[kind] = rows
            for boundary in ["lower", "upper"]:
                bounds[boundary][kind] = {
                    "kind": "windows",
                    "expected_windows": len(rows),
                    "source_refs": sorted(refs),
                    "rows": [
                        {
                            "end": r["end"],
                            "target": r["target_ratio"],
                            "reference": r["reference_" + boundary + "_ratio"],
                        }
                        for r in rows
                    ],
                }
    return {
        "base_nodes": nodes,
        "stability_boundary_nodes": bounds,
        "window_bounds": windows,
        "audit": {
            "return_reference_count": len(review["return_codes"]),
            "stability_reference_count": len(review["stability_codes"]),
            "reference_scope": "既存1986份额目录505法律产品原件重编的目标适配；非全市场普查",
            "source_admission_replay": review,
            "true_returns": true_returns,
            "nav_time_audit": time_audit,
            "dividend_event_audit": event_audit,
            "no_imputation": True,
            "no_complete_case_deletion": True,
            "missing_values_not_replaced_by_median_or_midpoint": True,
            "boundaries_are_not_observed_peer_medians": True,
        },
    }


_COMPILED = OrderedDict()


def _compile_cached(inputs, source, root_name):
    """只缓存派生观察；冷调用复用本次已验全部字节，不复制约0.5GB原件。

    每次calculate仍先originals重核全部SHA和事项，不以缓存掩盖原件改动。
    最多保留两个不可变输入版本，不保存Originals或PDF对象，返回方另做deepcopy。
    """
    key = (json.dumps(inputs, ensure_ascii=False, sort_keys=True), root_name)
    if key not in _COMPILED:
        result = _compile(inputs, source)
        if len(_COMPILED) >= 2:
            _COMPILED.popitem(last=False)
        _COMPILED[key] = result
    return _COMPILED[key]


def calculate(inputs, *, root):
    """每次核真实原字节后重放原R5；缺值界定的每窗同档才给稳定性点值。"""
    source = originals(inputs, Path(root))
    material = _material_review(inputs, source)
    compiled = deepcopy(_compile_cached(inputs, source, str(Path(root).resolve())))
    registry = {h: {**r, "path": str(Path(root) / r["file"])} for h, r in source.sources.items()}
    replays = {}
    for boundary in ["lower", "upper"]:
        packet = {
            "basis": ACTIVE_BASIS,
            "fund_code": CODE,
            "as_of_date": inputs["as_of_date"],
            "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
            "quality_metrics": {
                "target_product_aliases": ALIASES,
                "material_event_pending": material["material_event_pending"],
                "observations": {**compiled["base_nodes"], **compiled["stability_boundary_nodes"][boundary]},
            },
            "r4_evidence_registry": registry,
            "raw_archive": [{"kind": r.get("archive_kind", "ORIGINAL"), "sha256": h} for h, r in registry.items()],
            "quality_public": {"dataDates": {"nav": "2026-09-30", "holdings": "2026-06-30"}},
        }
        replays[boundary] = _calculate_frozen(
            packet, PROTOCOL, {"material_events": {CODE: material["material_events"]}}, selected_dimensions=SELECTED
        )
    require(replays["lower"]["dimensions"] == replays["upper"]["dimensions"], "R5_BOUNDARY_POINT_CONFLICT")
    require(
        all(v.get(k) is None for v in replays.values() for k in ["score", "raw_score", "grade"]), "TOTAL_SCORE_LEAK"
    )
    return json_ready(
        {
            "dimensions": replays["lower"]["dimensions"],
            "complete_score": None,
            "grade": None,
            "observations": {
                "exact": compiled["base_nodes"],
                "stability_identification_bounds": compiled["window_bounds"],
            },
            "audit": {**compiled["audit"], "material_event_review": material},
            "standard_boundary_replays": replays,
        }
    )


def check_binding(bundle, inputs):
    """禁止同管理人、同系列或旧主报告代替本C份额当前真实主报告。"""
    require(
        inputs.get("fund_code") == CODE
        and inputs.get("share_class") == "C"
        and bundle.get("family") == "ACTIVE_EQUITY"
        and bundle.get("fund_code") == inputs.get("fund_code")
        and bundle.get("share_class") == inputs.get("share_class")
        and str(bundle.get("as_of_date")) == inputs.get("as_of_date")
        and str(bundle.get("holding_date")) == inputs.get("holding_date")
        and bundle["identity"]["legal_name"] == inputs.get("legal_name")
        and bundle["identity"]["manager_name"] == inputs.get("manager_name")
        and bundle["original_report_archive"]["sha256"] == inputs.get("primary_ref"),
        "CURRENT_BINDING",
    )
    periods = inputs["source_admission"]["target_quarters"]
    require(
        sum(row["period"] == inputs["holding_date"] for row in periods) == 1
        and inputs["source_admission"]["primary_report_ref"] == inputs["primary_ref"],
        "CURRENT_PRIMARY_REPORT_ACTUAL_QUARTER_BINDING",
    )


def check_snapshot(bundle, dimensions, *, root, compute=None):
    """标准快照重新由原件求两维，并核每个lower/upper/candidate，不信候选点分。"""
    raw = bundle["active_broad_components_calculation"]["inputs"]
    check_binding(bundle, raw)
    result = (compute or calculate)(raw, root=root)
    require(
        set(result["dimensions"]) == SELECTED and result.get("complete_score") is None and result.get("grade") is None,
        "TWO_COMPONENT_RESULT_SCOPE",
    )
    for key, component in DOCUMENT_KEYS.items():
        expected = D(result["dimensions"][component])
        require(expected.is_finite(), "NONFINITE_REPLAY_POINT")
        for bound in ("lower", "upper", "candidate"):
            value = dimensions[key].get(bound)
            require(
                value is not None and D(str(value)).is_finite() and abs(D(str(value)) - expected) <= D("1e-55"),
                "DIMENSION_ORIGINAL_RECALCULATION:" + key + ":" + bound,
            )
    return result


def prepare(session, request, *, root):
    """只读最新current；新增两维文档/原件包，其他维度及核验时间原样保留。"""
    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    raw = deepcopy(request["inputs"])
    code = request["fund_code"]
    previous = current_inputs(session, [code]).get(code)
    require(previous and str(previous["as_of_date"]) == request["as_of_date"], "LATEST_CURRENT_PERIOD_REQUIRED")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    check_binding(bundle, raw)
    require(
        bundle.get("basis") == BASIS
        and not any(
            bundle.get(k)
            for k in [
                "r5_calculation",
                "passive_components_calculation",
                "passive_performance_calculation",
                "stability_calculation",
                "gold_components_calculation",
                "enhanced_performance_calculation",
                "bosera_performance_calculation",
            ]
        ),
        "PARTIAL_OR_CONFLICTING_COMPONENT_REQUIRED",
    )
    # 标准prepare入口统一canonical；此层只读取规范sources，绝不再按外部path找资料。
    result = calculate(raw, root=root)
    doc = bundle["reviewed_document"]
    unchanged = {k: deepcopy(v) for k, v in doc["dimensions"].items() if k not in DOCUMENT_KEYS}
    for key, component in DOCUMENT_KEYS.items():
        reason = request["explanations"][component].strip()
        require(len(reason) >= 40, "BUSINESS_EXPLANATION_REQUIRED")
        point = result["dimensions"][component]
        doc["dimensions"][key] = {"lower": point, "upper": point, "candidate": point, "reason": reason}
        bundle.setdefault("public_explanations", {})[key] = reason
        bundle.get("public_metrics", {}).pop(key, None)
    require(
        unchanged == {k: v for k, v in doc["dimensions"].items() if k not in DOCUMENT_KEYS}, "OTHER_DIMENSION_CHANGED"
    )
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, doc),
        active_broad_components_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    known = {r["sha256"] for r in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(r for h, r in raw["sources"].items() if h not in known)
    validate_sources(bundle, datetime.now(UTC), root=root)
    checked = snapshot_calculate(bundle, root=root)
    require(
        checked["status"] == "PARTIAL" and checked["score"] is None and checked.get("grade") is None,
        "CANNOT_CLAIM_COMPLETE",
    )
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}
