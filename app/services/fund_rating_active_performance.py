"""主动基金原件表现编译候选；只读显式归档，不读取临时研究目录或预填分数。

最小已验收范围为同一截止日、固定准入组的主动收益。风险和稳定性使用相同
原始净值与分红入口，必要日/月末缺失即拒绝；不借其他份额或删除缺数据同行。
"""

import json
import re
import unicodedata
from copy import deepcopy
from datetime import date
from decimal import Decimal as D
from decimal import localcontext
from functools import lru_cache
from pathlib import Path
from statistics import median

from app.services.material_content import content_digest, source_bytes, source_exists

VERSION = "R5_ACTIVE_PERFORMANCE_ORIGINALS_20261029_3"
TIME_CONTRACT = "CURRENT_RESEARCH_HISTORICAL_FUND_RATING_NOT_POINT_IN_TIME_TRADING_SIGNAL"
RETURN_ESTIMAND = "R5_FROZEN_DISCLOSED_NAV_GROWTH_20261019"
PROTOCOL_SHA256 = "771606ecb96ebca237b1b58ccc09f0b1aba6d5a7763558ba0c4de7ae379a563c"
PROTOCOL_FORMULA = (
    "G(s,t)=NAV(t)/NAV(s)*PRODUCT[NAV(previous_trading_day)/(NAV(previous_trading_day)-cash_per_share)]；s<ex_date<=t"
)


def _day(value, error):
    """公告日、净值日均须为真实公历日期，不只检查八位数字外形。"""
    text = str(value or "").replace("-", "")
    if not re.fullmatch(r"\d{8}", text):
        raise ValueError(error)
    try:
        date(int(text[:4]), int(text[4:6]), int(text[6:]))
    except ValueError as invalid:
        raise ValueError(error) from invalid
    return text


def _calendar(inputs, root):
    """把共同日历追到原始trade_cal成功回包，逐日对账并拒绝删日或空页。"""
    calendar = _read(inputs, root, inputs["calendar_ref"])
    if not calendar.get("sources"):
        raise ValueError("ACTIVE_CALENDAR_ORIGINAL_REQUIRED")
    originals = []
    for source in calendar["sources"]:
        if source.get("status") != 200 or source.get("api") != "trade_cal":
            raise ValueError("ACTIVE_CALENDAR_SOURCE_NOT_SUCCESS")
        rows = _licensed_rows(_read(inputs, root, source["sha256"]))
        if any(row.get("exchange") != "SSE" for row in rows):
            raise ValueError("ACTIVE_CALENDAR_EXCHANGE_IDENTITY")
        originals.extend(rows)

    def normalize(values):
        return sorted(values, key=lambda row: row["cal_date"])

    if normalize(calendar["rows"]) != normalize(originals):
        raise ValueError("ACTIVE_CALENDAR_DERIVED_ROWS_CONFLICT")
    days = sorted(
        row["cal_date"]
        for row in originals
        if str(row["is_open"]) == "1" and "20230928" <= row["cal_date"] <= "20260930"
    )
    if len(days) != 727 or len(set(days)) != 727 or days[0] != "20230928" or days[-1] != "20260930":
        raise ValueError("ACTIVE_COMMON_CALENDAR")
    return days


def _bytes(inputs, root, digest):
    """只允许root/sources内的实际文件，逐次核摘要，拒绝外部路径和软链接逃逸。"""
    if digest not in inputs["sources"]:
        raise ValueError("ACTIVE_ORIGINAL_SOURCE_REFERENCE_MISSING")
    ref = inputs["sources"][digest]
    path = (root / ref["file"]).resolve()
    if not path.is_relative_to(root / "sources") or not source_exists(path):
        raise ValueError("ACTIVE_CANONICAL_SOURCE_REQUIRED")
    raw = source_bytes(path)
    if content_digest(raw) != digest:
        raise ValueError("ACTIVE_ORIGINAL_HASH_CHANGED")
    return raw


def _read(inputs, root, digest):
    return json.loads(_bytes(inputs, root, digest))


def _mandate_identity(inputs, root, report, product):
    """区分实际PDF和完整公告转载，全文必须与原法律产品及本代码绑定。"""
    raw = _bytes(inputs, root, report["source"]["sha256"])
    if raw.startswith(b"%PDF"):
        from app.services.material_content import document_reader as PdfReader

        # 摘要已覆盖完整原件字节；身份只需找到明确代码及法律产品名称，命中即停止解析，
        # 不为重复身份复核把数百页附注在共享16GB主机上反复展开。未命中时仍检查全文。
        compact = ""
        for page in PdfReader(raw).pages:
            compact += re.sub(r"\s+", "", page.extract_text() or "")
            if product["fund_code"] in compact and product["legal_name"] in compact:
                return True
        return False
    if raw.lstrip().startswith(b"<"):
        from bs4 import BeautifulSoup

        text = BeautifulSoup(raw.decode("utf-8", errors="replace"), "html.parser").get_text(" ", strip=True)
        compact = re.sub(r"\s+", "", text)
        return len(text) > 1500 and product["fund_code"] in compact and product["legal_name"] in compact
    original = json.loads(raw)
    data = original.get("data") or {}
    if original.get("success") != 1 or product["fund_code"] not in {s["stock"] for s in data.get("security", [])}:
        return False
    if len(data.get("notice_content", "")) >= 1500:
        return True
    # 原转载空正文且已另取实际PDF时核其真实字节；不把空正文当资料永久不存在。
    if report.get("pdf_source"):
        copied = {**report, "source": report["pdf_source"]}
        return _mandate_identity(inputs, root, copied, product)
    return False


def _report_publication(inputs, root, report):
    """缺目录日期时从原件送出日期恢复，不把年度标题、取得日期当公开日。"""
    if report.get("published_on"):
        return report["published_on"]
    raw = _bytes(inputs, root, report["source"]["sha256"])
    if raw.startswith(b"%PDF"):
        from app.services.material_content import document_reader as PdfReader

        pages = [p.extract_text() or "" for p in PdfReader(raw).pages]
    elif raw.lstrip().startswith(b"<"):
        from bs4 import BeautifulSoup

        pages = [BeautifulSoup(raw.decode("utf-8", errors="replace"), "html.parser").get_text(" ", strip=True)]
    else:
        original = json.loads(raw)
        data = original.get("data") or {}
        if data.get("notice_date"):
            return data["notice_date"][:10]
        pages = [data.get("notice_content", "")]
    from app.services.fund_rating_public_market import publication

    return publication(pages)


def _licensed_rows(raw):
    # 授权失败、截断页和空壳响应均不是“没有分红”；必须先核供应商成功与分页闭合。
    if type(raw.get("code")) is not int or raw["code"] != 0:
        raise ValueError("ACTIVE_LICENSED_RESPONSE_NOT_SUCCESS")
    data = raw.get("data") or {}
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("fields"), list)
        or not isinstance(data.get("items"), list)
    ):
        raise ValueError("ACTIVE_LICENSED_RESPONSE_SHAPE")
    if data.get("has_more") is not False:
        raise ValueError("ACTIVE_LICENSED_PAGINATION_NOT_CLOSED")
    if not data["fields"] or len(set(data["fields"])) != len(data["fields"]):
        raise ValueError("ACTIVE_LICENSED_FIELDS_REQUIRED")
    return [dict(zip(data["fields"], row, strict=True)) for row in data["items"]]


def _nav_time_scope(inputs):
    """研究历史净值时分开估值日和可取得日；不声称是月末当时已知信号。"""
    if inputs.get("time_contract") != TIME_CONTRACT:
        raise ValueError("ACTIVE_EXPLICIT_NAV_TIME_CONTRACT_REQUIRED")
    cutoff = _day(inputs.get("as_of_date"), "ACTIVE_VALUATION_CUTOFF_REQUIRED")
    research = _day(inputs.get("research_on"), "ACTIVE_RESEARCH_DATE_REQUIRED")
    if research < cutoff:
        raise ValueError("ACTIVE_RESEARCH_BEFORE_VALUATION_CUTOFF")
    return cutoff, research


def _require_time_contract(inputs, root):
    """研究日需与封存的日期裁定一致，基本面/合同仍遵守原月末公开边界。"""
    cutoff, research = _nav_time_scope(inputs)
    reference = inputs.get("time_contract_ref")
    if not reference:
        raise ValueError("ACTIVE_NAV_TIME_DECISION_ORIGINAL_REQUIRED")
    decision = _read(inputs, root, reference)
    if (
        decision.get("purpose") != TIME_CONTRACT
        or _day(decision.get("valuation_cutoff"), "ACTIVE_TIME_DECISION_SCOPE") != cutoff
        or _day(decision.get("research_on"), "ACTIVE_TIME_DECISION_SCOPE") != research
        or decision.get("calendar_weights_formula_changed") is not False
        or decision.get("not_authorized_for_point_in_time_prediction_or_backtest") is not True
    ):
        raise ValueError("ACTIVE_NAV_TIME_DECISION_SCOPE_CONFLICT")
    return {
        "contract": TIME_CONTRACT,
        "valuation_cutoff": inputs["as_of_date"],
        "research_on": inputs["research_on"],
        "decision_ref": reference,
        "fundamental_publication_cutoff": inputs["as_of_date"],
        "point_in_time_trading_signal": False,
    }


def _require_material_review(inputs, root):
    """重读本基金V2复核的pending原值、34源、证据引文及全部事件引用图。"""
    if inputs.get("material_event_pending") is not False or not inputs.get("material_event_review_ref"):
        raise ValueError("ACTIVE_MATERIAL_EVENT_REVIEW_REQUIRED")
    review = _read(inputs, root, inputs["material_event_review_ref"])
    if (
        review.get("fund_code") != inputs["fund_code"]
        or review.get("share_class") != inputs["share_class"]
        or review.get("as_of") != inputs["as_of_date"]
        or review.get("legal_product_aliases") != inputs["target_aliases"]
    ):
        raise ValueError("ACTIVE_MATERIAL_EVENT_REVIEW_IDENTITY")
    # 字段明确为pending；缺失或未知绝不能通过.get(...,False)变成无未决事项。
    if (
        "pending" not in review
        or review["pending"] is not False
        or review.get("review_closed") is not True
        or review["pending"] is not inputs["material_event_pending"]
    ):
        raise ValueError("ACTIVE_MATERIAL_EVENT_REVIEW_NOT_CLOSED")
    if (
        _day(str(review.get("recorded_at", ""))[:10], "ACTIVE_MATERIAL_REVIEW_DATE_REQUIRED")
        > _nav_time_scope(inputs)[1]
    ):
        raise ValueError("ACTIVE_MATERIAL_REVIEW_AFTER_RESEARCH_DATE")
    registry = review.get("source_registry")
    evidence, events, anchors = review.get("evidence"), review.get("events"), review.get("anchors")
    if (
        not isinstance(registry, dict)
        or not registry
        or not isinstance(evidence, list)
        or not evidence
        or not isinstance(events, list)
        or not anchors
    ):
        raise ValueError("ACTIVE_MATERIAL_EVENT_ORIGINAL_ANCHORS_REQUIRED")
    evidence_by_id = {row["id"]: row for row in evidence}
    if len(evidence_by_id) != len(evidence) or not set(anchors) <= evidence_by_id.keys():
        raise ValueError("ACTIVE_MATERIAL_EVENT_EVIDENCE_GRAPH")
    originals = {}
    for digest, ref in registry.items():
        if ref.get("sha256") != digest:
            raise ValueError("ACTIVE_MATERIAL_REGISTRY_HASH_CONFLICT")
        originals[digest] = json.loads(_bytes(inputs, root, digest))

    def compact(value):
        return re.sub(r"\s+", "", str(value))

    cutoff = inputs["as_of_date"].replace("-", "")
    for row in evidence:
        published = _day(row.get("published_on"), "ACTIVE_MATERIAL_EVENT_PUBLICATION_REQUIRED")
        if published > cutoff:
            raise ValueError("ACTIVE_MATERIAL_EVENT_FUTURE_EVIDENCE")
        if row.get("source_ref") not in originals or not row.get("quote"):
            raise ValueError("ACTIVE_MATERIAL_EVENT_EVIDENCE_SOURCE_REQUIRED")
        original = originals[row["source_ref"]]
        data = original.get("data") or {}
        if original.get("success") != 1 or not data.get("notice_content"):
            raise ValueError("ACTIVE_MATERIAL_NOTICE_BODY_NOT_SUCCESS")
        if _day(str(data.get("notice_date", ""))[:10], "ACTIVE_MATERIAL_ORIGINAL_PUBLICATION_REQUIRED") != published:
            raise ValueError("ACTIVE_MATERIAL_PUBLICATION_NOT_BOUND_TO_ORIGINAL")
        if compact(row["quote"]) not in compact(data["notice_content"]):
            raise ValueError("ACTIVE_MATERIAL_EVENT_QUOTE_NOT_IN_ORIGINAL")
        if row["id"] in anchors:
            codes = {str(v.get("stock")) for v in data.get("security", [])}
            if not codes.intersection(inputs["target_aliases"]) or compact(
                inputs["target"]["legal_name"]
            ) not in compact(data["notice_content"]):
                raise ValueError("ACTIVE_MATERIAL_PRIMARY_ANCHOR_IDENTITY")
    identifiers = set()
    for event in events:
        if not event.get("id") or event["id"] in identifiers:
            raise ValueError("ACTIVE_MATERIAL_EVENT_ID_REQUIRED_OR_DUPLICATED")
        identifiers.add(event["id"])
        if (
            "pending_for_fund_performance_gate" not in event
            or event["pending_for_fund_performance_gate"] is not review["pending"]
        ):
            raise ValueError("ACTIVE_MATERIAL_EVENT_PENDING_GRAPH_CONFLICT")
        day = event.get("published_on") or event.get("event_date")
        if _day(day, "ACTIVE_MATERIAL_EVENT_DATE_REQUIRED") > cutoff:
            raise ValueError("ACTIVE_MATERIAL_EVENT_AFTER_CUTOFF")
        linked_evidence, linked_sources = event.get("evidence", []), event.get("sources", [])
        if (
            not linked_evidence
            and not linked_sources
            or not set(linked_evidence) <= evidence_by_id.keys()
            or not set(linked_sources) <= originals.keys()
        ):
            raise ValueError("ACTIVE_MATERIAL_EVENT_EVIDENCE_GRAPH")
    catalog = review.get("catalogue", {})
    pages, catalog_rows = [], []
    observation_count = 0
    for digest, ref in registry.items():
        original = originals[digest]
        if ref.get("role") == "COMPLETE_TARGET_ANNOUNCEMENT_CATALOG":
            if original.get("ErrCode") != 0 or not isinstance(original.get("Data"), list):
                raise ValueError("ACTIVE_MATERIAL_CATALOG_NOT_SUCCESS")
            if any(row.get("FUNDCODE") != inputs["fund_code"] for row in original["Data"]):
                raise ValueError("ACTIVE_MATERIAL_CATALOG_IDENTITY")
            pages.append(original.get("PageIndex"))
            catalog_rows.extend(original["Data"])
        elif ref.get("kind") == "WEB_PRIMARY_TEXT_OBSERVATION":
            # 网页直接观察只用于其明确的其他法律产品排除，不伪装成已存原PDF字节。
            if (
                original.get("raw_pdf_bytes_archived") is not False
                or not original.get("original_identity_quote")
                or original.get("original_identity_quote") != original.get("applicable_legal_product")
                or original.get("applicable_legal_product") == inputs["target"]["legal_name"]
                or _day(original.get("published_on"), "ACTIVE_MATERIAL_OBSERVATION_PUBLICATION") > cutoff
            ):
                raise ValueError("ACTIVE_MATERIAL_WEB_OBSERVATION_SCOPE")
            observation_count += 1
    if (
        catalog.get("closed_to_start") is not True
        or catalog.get("as_of") != inputs["as_of_date"]
        or not pages
        or sorted(pages) != list(range(1, len(pages) + 1))
        or len(pages) != catalog.get("captured_pages")
    ):
        raise ValueError("ACTIVE_MATERIAL_CATALOG_CLOSURE_REQUIRED")
    in_scope = [row for row in catalog_rows if catalog["from"] <= str(row["PUBLISHDATE"])[:10] <= catalog["as_of"]]
    if (
        len({row["ID"] for row in in_scope}) != len(in_scope)
        or len(in_scope) != catalog.get("count")
        or not any(str(row["PUBLISHDATE"])[:10] < catalog["from"] for row in catalog_rows)
    ):
        raise ValueError("ACTIVE_MATERIAL_CATALOG_COUNT_OR_START_CONFLICT")
    return {
        "review_ref": inputs["material_event_review_ref"],
        "review_version": review.get("version"),
        "pending": review["pending"],
        "review_closed": review["review_closed"],
        "source_count": len(registry),
        "evidence_count": len(evidence),
        "event_count": len(events),
        "catalog_count": len(in_scope),
        "original_quotes_and_dates_reverified": True,
        "web_observations_not_claimed_as_pdf_bytes": observation_count,
        "no_independent_regulatory_release_claim": review.get("no_independent_regulatory_release_claim"),
    }


def _product(inputs, root, product, *, audit=None):
    """重读本代码单位净值与全部分红，优先净值除息日，重复事件按现金/宣告日去重。"""
    code = product["fund_code"]
    cutoff, research = _nav_time_scope(inputs)
    if audit is not None:
        audit.update(
            fund_code=code,
            valuation_cutoff=inputs["as_of_date"],
            research_on=inputs["research_on"],
            late_announcements=[],
            nav_observations=0,
            actual_dates_retained_in_originals=True,
            point_in_time_trading_signal=False,
        )
    points, events = {}, []
    if product["format"] == "LICENSED_SHARE_HISTORY_AGGREGATE":
        raw = _read(inputs, root, product["history_ref"])
        if (
            raw["fund_code"] != code
            or raw.get("dividend_records") is None
            or not raw.get("event_refresh", {}).get("dividends_verified_at")
        ):
            raise ValueError("ACTIVE_SHARE_OR_EVENT_CLOSURE")
        if raw.get("event_refresh", {}).get("status") != "SUCCEEDED":
            raise ValueError("ACTIVE_SHARE_EVENT_REFRESH_NOT_SUCCESS")
        rows = [
            {"nav_date": r["nav_date"].replace("-", ""), "ann_date": r.get("ann_date"), "unit_nav": r["unit_nav"]}
            for r in raw["history"]["rows"]
        ]
        event_rows = raw["dividend_records"]
    elif product["format"] == "TUSHARE_NAV_AND_DIVIDEND_ORIGINALS":
        rows = []
        for digest in product["nav_refs"]:
            for row in _licensed_rows(_read(inputs, root, digest)):
                if row["ts_code"] != product["ts_code"] or row["ts_code"].split(".")[0] != code:
                    raise ValueError("ACTIVE_NAV_SHARE_MISMATCH")
                rows.append(row)
        receipt = _read(inputs, root, product["dividend_receipt_ref"])
        if receipt["code"] != code:
            raise ValueError("ACTIVE_DIVIDEND_SHARE_MISMATCH")
        original_events = []
        for digest in product["dividend_refs"]:
            original_events.extend(_licensed_rows(_read(inputs, root, digest)))
        # 空回包也需本代码请求回执绑定；不把其他基金空回包随意借用。
        if {r["sha256"] for r in receipt["sources"]} != set(product["dividend_refs"]):
            raise ValueError("ACTIVE_DIVIDEND_RECEIPT_BINDING")
        if receipt["rows"] != original_events:
            raise ValueError("ACTIVE_DIVIDEND_RECEIPT_ROWS_CONFLICT")
        event_rows = original_events
    else:
        raise ValueError("ACTIVE_SOURCE_FORMAT_UNSUPPORTED")
    for row in rows:
        day = _day(row["nav_date"], "ACTIVE_NAV_DATE_FORMAT")
        if day > cutoff:
            raise ValueError("ACTIVE_NAV_VALUATION_AFTER_CUTOFF")
        announcement = _day(row.get("ann_date"), "ACTIVE_NAV_ANNOUNCEMENT_REQUIRED")
        if announcement > research:
            raise ValueError("ACTIVE_NAV_ANNOUNCED_AFTER_RESEARCH_DATE")
        if audit is not None:
            audit["nav_observations"] += 1
            if announcement > cutoff:
                audit["late_announcements"].append(
                    {
                        "valuation_date": day,
                        "announced_on": announcement,
                        "known_on_valuation_cutoff": False,
                        "usable_only_for_current_historical_research": True,
                    }
                )
        nav = D(str(row["unit_nav"]))
        if nav <= 0 or not nav.is_finite():
            raise ValueError("ACTIVE_POSITIVE_NAV_REQUIRED")
        if day in points and points[day] != nav:
            raise ValueError("ACTIVE_DUPLICATE_NAV_CONFLICT")
        points[day] = nav
    grouped = {}
    cash_by_ex_date = {}
    for event in event_rows:
        event_code = event.get("fund_code") or (event.get("ts_code") or "").split(".")[0]
        if event_code and event_code != code:
            raise ValueError("ACTIVE_DIVIDEND_WRONG_CODE")
        status_values = {str(event[k]) for k in ["div_proc", "process_status"] if event.get(k) is not None}
        if len(status_values) != 1:
            raise ValueError("ACTIVE_DIVIDEND_STATUS_UNKNOWN_OR_CONFLICT")
        status = next(iter(status_values))
        if status not in {"实施", "预案", "取消", "终止", "未实施"}:
            raise ValueError("ACTIVE_DIVIDEND_STATUS_UNKNOWN_OR_CONFLICT")
        if status != "实施":
            continue
        cash_values = {D(str(event[k])) for k in ["div_cash", "cash_dividend"] if event.get(k) is not None}
        if len(cash_values) != 1 or not all(v.is_finite() and v >= 0 for v in cash_values):
            raise ValueError("ACTIVE_DIVIDEND_CASH_MISSING_OR_CONFLICT")
        if not event.get("ex_date"):
            raise ValueError("ACTIVE_DIVIDEND_EX_DATE_MISSING")
        declared = str(event["ex_date"]).replace("-", "")
        ann = str(event.get("ann_date") or "").replace("-", "")
        if not ann:
            raise ValueError("ACTIVE_DIVIDEND_ANNOUNCEMENT_REQUIRED")
        declared = _day(declared, "ACTIVE_DIVIDEND_DATE_FORMAT")
        ann = _day(ann, "ACTIVE_DIVIDEND_DATE_FORMAT")
        if ann > inputs["as_of_date"].replace("-", ""):
            if declared <= inputs["as_of_date"].replace("-", ""):
                raise ValueError("ACTIVE_DIVIDEND_ANNOUNCED_AFTER_CUTOFF")
            continue
        effective_values = {str(event[k]).replace("-", "") for k in ["net_ex_date", "nav_ex_date"] if event.get(k)}
        if len(effective_values) > 1 or any(not re.fullmatch(r"\d{8}", v) for v in effective_values):
            raise ValueError("ACTIVE_DIVIDEND_EFFECTIVE_DATE_CONFLICT")
        effective_values = {_day(value, "ACTIVE_DIVIDEND_EFFECTIVE_DATE_CONFLICT") for value in effective_values}
        cash = next(iter(cash_values))
        if declared in cash_by_ex_date and cash_by_ex_date[declared] != cash:
            raise ValueError("ACTIVE_DIVIDEND_CASH_EVENT_CONFLICT")
        cash_by_ex_date[declared] = cash
        # Decimal作键：0.1与0.10是同一金额，不能按字符串分成两次事件。
        key = (declared, cash)
        grouped.setdefault(key, []).append({"effective_dates": effective_values, "original": event, "ann_date": ann})
    for (declared, cash), records in grouped.items():
        effective = set().union(*(r["effective_dates"] for r in records))
        if len(effective) > 1:
            raise ValueError("ACTIVE_DIVIDEND_EFFECTIVE_DATE_CONFLICT")
        events.append(
            {
                "date": next(iter(effective)) if effective else declared,
                "cash": D(cash),
                "ann_dates": sorted({r["ann_date"] for r in records if r["ann_date"]}),
                "declared_ex_date": declared,
                "process_status": "实施",
                "original_records": [r["original"] for r in records],
            }
        )
    return points, events


def _cash_reinvestment_growth(points, events, days, start, end):
    """独立审计对照：除息日现金再投资，与通用total_return一致，不参与R5评分。"""
    if start not in points or end not in points:
        raise ValueError("ACTIVE_RETURN_ENDPOINT_MISSING")
    value = points[end] / points[start]
    for event in events:
        if start < event["date"] <= end:
            if event["date"] not in days or event["date"] not in points:
                raise ValueError("ACTIVE_DIVIDEND_EX_DATE_NAV_MISSING")
            value *= (points[event["date"]] + event["cash"]) / points[event["date"]]
    return value


def _growth(points, events, days, start, end):
    """按冻结披露统计式计算净值增长，缺实际前一交易日净值即拒绝。

    前一交易日来自显式共同日历，不能用上一条稀疏披露日、其他份额或插值替代。
    现金必须小于前日净值；无分红时仍仅需真实窗口端点。
    """
    if start not in points or end not in points:
        raise ValueError("ACTIVE_RETURN_ENDPOINT_MISSING")
    value = points[end] / points[start]
    for event in events:
        if start < event["date"] <= end:
            if event["date"] not in days or days.index(event["date"]) == 0:
                raise ValueError("ACTIVE_DIVIDEND_PREVIOUS_CALENDAR_DAY_MISSING")
            previous = days[days.index(event["date"]) - 1]
            if previous not in points:
                raise ValueError("ACTIVE_DIVIDEND_PREVIOUS_DAY_NAV_MISSING")
            nav, cash = points[previous], event["cash"]
            if not nav.is_finite() or not cash.is_finite() or cash < 0 or nav <= cash:
                raise ValueError("ACTIVE_DIVIDEND_PREVIOUS_NAV_NOT_ABOVE_CASH")
            value *= nav / (nav - cash)
    return value


def compile_observations(inputs, *, root):
    """核验固定准入与法律产品去重后编译原始收益、风险或稳定性观察。

    准入属于已核、已冻结的事实判断，编译器重放名单、逐期暴露门槛和原件
    摘要，不把基金名称关键词当完整策略分类，也不在看到分数后挑比较对象。
    """
    from app.services.fund_rating_r5 import exposure_factor, peer_gap_score, signed_gap_score, window_frequency_score

    root = Path(root).resolve()
    if inputs.get("version") != VERSION or inputs["as_of_date"] != "2026-09-30":
        raise ValueError("ACTIVE_INPUT_VERSION_OR_PERIOD")
    if inputs.get("return_estimand") != RETURN_ESTIMAND:
        raise ValueError("ACTIVE_EXPLICIT_RETURN_ESTIMAND_REQUIRED")
    if inputs.get("statistic_protocol_ref") != PROTOCOL_SHA256:
        raise ValueError("ACTIVE_FROZEN_STATISTIC_PROTOCOL_REQUIRED")
    protocol = _read(inputs, root, PROTOCOL_SHA256)
    if protocol.get("direct_formula") != PROTOCOL_FORMULA:
        raise ValueError("ACTIVE_FROZEN_STATISTIC_FORMULA_CHANGED")
    selected = set(inputs["selected_dimensions"])
    if not selected or not selected <= {"return", "risk", "stability"}:
        raise ValueError("ACTIVE_COMPONENT_SCOPE")
    if any(k in inputs for k in ["score", "dimensions", "precomputed_scores"]):
        raise ValueError("ACTIVE_PRECOMPUTED_SCORE_FORBIDDEN")
    target = inputs["target"]
    if target["fund_code"] != inputs["fund_code"] or target["share_class"] != inputs["share_class"]:
        raise ValueError("ACTIVE_TARGET_SHARE_BINDING")
    time_contract_audit = _require_time_contract(inputs, root)
    material_event_audit = _require_material_review(inputs, root)
    peers = inputs["peers"]
    codes = [p["fund_code"] for p in peers]
    ids = [p["legal_product_id"] for p in peers]
    if (
        not peers
        or len(set(codes)) != len(codes)
        or len(set(ids)) != len(ids)
        or len({p["legal_name"] for p in peers}) != len(peers)
        or target["legal_name"] in {p["legal_name"] for p in peers}
        or target["legal_product_id"] in ids
    ):
        raise ValueError("ACTIVE_LEGAL_PRODUCT_PEER_CLOSURE")
    freeze = _read(inputs, root, inputs["cohort_ref"])
    cohort_key = "return" if selected == {"return"} else "risk_stability"
    if freeze["target_code"] != inputs["fund_code"] or freeze["target_aliases"] != inputs["target_aliases"]:
        raise ValueError("ACTIVE_COHORT_TARGET_BINDING")
    rows = freeze[cohort_key]["rows"]
    if any(r["state"].endswith(("PENDING", "REQUIRED")) for r in rows):
        raise ValueError("ACTIVE_COHORT_ADMISSION_NOT_CLOSED")
    expected = sorted(r["code"] for r in rows if r["state"] == "ADMITTED")
    if sorted(codes) != expected or set(codes) & set(inputs["target_aliases"]):
        raise ValueError("ACTIVE_FIXED_COHORT_MEMBERS_CHANGED")
    admission = _read(inputs, root, inputs["admission_ref"])
    lookup = {r["code"]: r for r in admission}
    # 从本产品12份季报重读实际股票金额及A/C净资产，重建本目标的仓位基准。
    exposure_target = {}
    money = r"(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}"
    for quarter in inputs["target_quarters"]:
        original = _read(inputs, root, quarter["source_ref"])
        data = original.get("data") or {}
        if (
            inputs["fund_code"] not in {s["stock"] for s in data.get("security", [])}
            or data["notice_date"][:10] > inputs["as_of_date"]
        ):
            raise ValueError("ACTIVE_QUARTER_IDENTITY_OR_DATE")
        text = unicodedata.normalize("NFKC", data["notice_content"])
        table = text[text.rfind("期末基金资产组合情况") :]
        stock = re.search(r"其中\s*[:：]\s*股票\s+(" + money + ")", table)
        nav = re.search(r"4[.、]?\s*期末基金资产净值\s+((" + money + r"\s*)+)", text)
        if not stock or not nav:
            raise ValueError("ACTIVE_QUARTER_FINANCIAL_ROWS")
        columns = re.findall(money, nav[1])
        if len(columns) != len(inputs["target_aliases"]):
            raise ValueError("ACTIVE_QUARTER_SHARE_COLUMNS")
        exposure_target[quarter["period_end"]] = (
            D(stock[1].replace(",", "")) / sum(D(v.replace(",", "")) for v in columns) * 100
        )
    if len(exposure_target) != 12 or any(
        str(v) != freeze["target_quarter_exposure"].get(k) for k, v in exposure_target.items()
    ):
        # 同样的金额换算可在不同Decimal上下文尾位舍入，只容许运算尾数差，非资料容差。
        if len(exposure_target) != 12 or any(
            abs(v - D(freeze["target_quarter_exposure"].get(k, "Infinity"))) > D("1e-24")
            for k, v in exposure_target.items()
        ):
            raise ValueError("ACTIVE_TARGET_EXPOSURE_REPLAY_CONFLICT")
    from app.services.fund_rating_active_exposure import replay_exposures

    exposure, exposure_audit = replay_exposures(inputs, root, _read, _bytes)
    thresholds = freeze[cohort_key]["thresholds"]
    replay_members = []
    for item in exposure:
        if item["code"] in inputs["target_aliases"]:
            continue
        quarters = item["quarters"]
        if len(quarters) != 12 or item["gaps"]:
            continue
        upper = [
            abs(D(q["equity_nav_pct"]) - exposure_target[q["date"]]) + D(q["rounding_radius_pp"]) for q in quarters
        ]
        region = item["region_status_by_cutoff"][str(thresholds["domestic_min_pct"])]
        admitted = lookup.get(item["code"], {}).get("decision") == "COMPARABLE_MANDATE_EVIDENCE_AVAILABLE"
        if (
            max(upper) <= thresholds["max_gap_pp"]
            and sum(upper) / 12 <= thresholds["mean_gap_pp"]
            and region.startswith("REGION_PASS")
            and admitted
        ):
            replay_members.append(item["code"])
    if sorted(replay_members) != sorted(codes):
        raise ValueError("ACTIVE_EXPOSURE_ADMISSION_REPLAY_CONFLICT")
    admission_audit = []
    for p in peers:
        item = lookup[p["fund_code"]]
        if (
            item["decision"] != "COMPARABLE_MANDATE_EVIDENCE_AVAILABLE"
            or p["legal_name"] != item["legal_name"]
            or p["fund_code"] not in item["share_codes"]
        ):
            raise ValueError("ACTIVE_PEER_MANDATE_NOT_ADMITTED")
        reports = item["current_originals"] + item["window_originals"]
        for report in reports:
            if _report_publication(inputs, root, report) > inputs["as_of_date"]:
                raise ValueError("ACTIVE_FUTURE_CONTRACT_SOURCE")
            if not _mandate_identity(inputs, root, report, p):
                raise ValueError("ACTIVE_MANDATE_ORIGINAL_IDENTITY")
        admission_audit.append(
            {"fund_code": p["fund_code"], "legal_product_id": p["legal_product_id"], "reports": len(reports)}
        )
    days = _calendar(inputs, root)
    with localcontext() as ctx:
        ctx.prec = 60
        products = [target] + peers
        nav_time_audit = {p["fund_code"]: {} for p in products}
        raw = {p["fund_code"]: _product(inputs, root, p, audit=nav_time_audit[p["fund_code"]]) for p in products}
        observations, scores, audit = (
            {},
            {},
            {
                "admission": admission_audit,
                "calendar_points": len(days),
                "exposure_original_replay": exposure_audit,
                "material_event_review": material_event_audit,
                "time_contract": time_contract_audit,
                "nav_time_audit": nav_time_audit,
            },
        )
        if "return" in selected:
            values = {}
            years = D("365.25") / D((date(2026, 9, 30) - date(2023, 9, 28)).days)
            for p in products:
                points, events = raw[p["fund_code"]]
                values[p["fund_code"]] = [
                    _growth(points, events, days, "20250930", "20260930") - 1,
                    _growth(points, events, days, days[0], days[-1]) ** years - 1,
                ]
            subs = []
            for i, name in enumerate(["return_1y", "return_3y"]):
                target_value = values[target["fund_code"]][i]
                peer_values = [values[c][i] for c in codes]
                observations[name] = {
                    "kind": "rank",
                    "target": str(target_value),
                    "direction": "higher",
                    "peers": [
                        {"product_id": p["legal_product_id"], "value": str(values[p["fund_code"]][i])} for p in peers
                    ],
                    "expected_products": ids,
                }
                subs.append(signed_gap_score(target_value - median(peer_values), ".10"))
            scores["return"] = str(subs[0] * D(".3") + subs[1] * D(".7"))
        if "risk" in selected:
            if len(peers) < 2:
                raise ValueError("ACTIVE_RISK_LEAVE_ONE_OUT_NEEDS_TWO_PEERS")
            daily = {}
            for p in products:
                points, events = raw[p["fund_code"]]
                missing = sorted(set(days) - set(points))
                if missing:
                    raise ValueError("ACTIVE_REQUIRED_DAILY_NAV_MISSING:" + p["fund_code"] + ":" + str(len(missing)))
                daily[p["fund_code"]] = [
                    _growth(points, events, days, a, b) - 1 for a, b in zip(days, days[1:], strict=False)
                ]
            sums = [sum(daily[c][i] for c in codes) for i in range(len(days) - 1)]
            metrics = {}
            for p in products:
                own = p["fund_code"]
                include = own in codes
                count = len(codes) - int(include)
                base = [(s - (daily[own][i] if include else 0)) / count for i, s in enumerate(sums)]
                wealth = benchmark = peak = D(1)
                dd = square = D(0)
                for f, b in zip(daily[own], base, strict=True):
                    wealth *= 1 + f
                    benchmark *= 1 + b
                    relative = wealth / benchmark
                    peak = max(peak, relative)
                    dd = max(dd, 1 - relative / peak)
                    square += min(f - b, D(0)) ** 2
                metrics[own] = [dd, (D(252) / D(len(base)) * square).sqrt()]
            terms = []
            for i, name in enumerate(["extra_drawdown", "extra_downside"]):
                x = metrics[target["fund_code"]][i]
                pv = [metrics[c][i] for c in codes]
                observations[name] = {
                    "kind": "rank",
                    "target": str(x),
                    "direction": "lower",
                    "peers": [
                        {"product_id": p["legal_product_id"], "value": str(metrics[p["fund_code"]][i])} for p in peers
                    ],
                    "expected_products": ids,
                }
                terms.append(peer_gap_score(x, pv) * exposure_factor(name, x))
            scores["risk"] = str(terms[0] * D(".7") + terms[1] * D(".3"))
        if "stability" in selected:
            months = []
            for day in days:
                if not months or day[:6] != months[-1][:6]:
                    months.append(day)
                else:
                    months[-1] = day
            assert len(months) == 37
            terms = []
            for horizon, name, expected in [(1, "monthly_success", 36), (12, "annual_success", 25)]:
                rows = []
                pointscores = []
                unit = D(".10") / (12 if horizon == 1 else 1)
                for a, b in zip(months, months[horizon:], strict=False):
                    returns = {p["fund_code"]: _growth(*raw[p["fund_code"]], days, a, b) - 1 for p in products}
                    reference = median(returns[c] for c in codes)
                    value = returns[target["fund_code"]]
                    rows.append({"start": a, "end": b, "target": str(value), "reference": str(reference)})
                    pointscores.append(window_frequency_score(value - reference, unit))
                if len(rows) != expected:
                    raise ValueError("ACTIVE_WINDOW_COUNT")
                observations[name] = {"kind": "windows", "expected_windows": expected, "rows": rows}
                terms.append(sum(pointscores) / len(pointscores))
            scores["stability"] = str(terms[0] * D(".3") + terms[1] * D(".7"))
    return {
        "version": VERSION,
        "fund_code": inputs["fund_code"],
        "as_of_date": inputs["as_of_date"],
        "return_estimand": RETURN_ESTIMAND,
        "statistic_protocol_sha256": PROTOCOL_SHA256,
        "statistic_protocol_original_formula": PROTOCOL_FORMULA,
        "selected_dimensions": sorted(selected),
        "observations": observations,
        "component_scores": scores,
        "complete_score": None,
        "grade": None,
        "audit": audit,
    }


def compile_stability_bounds(inputs, *, root):
    """原频率档允许用真实上下界唯一判定；原始中位数仍保留范围，不制造参考单值。

    正净值增长的未公开端点只给(-1,+∞)这一数学可行域。枚举仓位准入跨界
    成员的全部可能名单，任一窗口或名单分支得分不唯一即拒绝点分。
    """
    from copy import deepcopy
    from itertools import product as combinations

    from bs4 import BeautifulSoup

    from app.services.fund_rating_r5 import window_frequency_score

    root = Path(root).resolve()
    base = deepcopy(inputs)
    base["selected_dimensions"] = ["return"]
    # 原件、目标份额、父组法律产品及12期实际暴露复用相同严格回放入口。
    base_compiled = compile_observations(base, root=root)
    freeze = _read(inputs, root, inputs["cohort_ref"])
    from app.services.fund_rating_active_exposure import replay_exposures

    exposures, original_exposure_audit = replay_exposures(inputs, root, _read, _bytes)
    parent = {p["fund_code"]: p for p in inputs["peers"]}
    target_weights = {k: D(v) for k, v in freeze["target_quarter_exposure"].items()}
    refinements = {}
    for refinement in inputs.get("exposure_refinements", []):
        raw = _bytes(inputs, root, refinement["source_ref"])
        text = raw.decode("utf-8", errors="strict")
        if refinement.get("kind") == "PRIMARY_QUARTER_FINANCIAL_AMOUNTS":
            original = json.loads(raw)
            data = original.get("data") or {}
            code = refinement["fund_code"]
            if original.get("success") != 1 or code not in {row["stock"] for row in data.get("security", [])}:
                raise ValueError("ACTIVE_REFINEMENT_IDENTITY")
            if _day(data.get("notice_date", "")[:10], "ACTIVE_REFINEMENT_PUBLICATION_REQUIRED") > inputs[
                "as_of_date"
            ].replace("-", ""):
                raise ValueError("ACTIVE_REFINEMENT_FUTURE_ORIGINAL")
            content = data.get("notice_content", "")
            money = r"(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}"
            nav = re.search(r"4\s*[.、]?\s*期末基金资产净值\s+((?:" + money + r"\s*)+)", content)
            heads = list(re.finditer(r"5\.1\s+报告期末基金资产组合情况", content))
            if not nav or not heads:
                raise ValueError("ACTIVE_REFINEMENT_FINANCIAL_ROWS_REQUIRED")
            stock = re.search(r"1\s+权益投资\s+(" + money + r")", content[heads[-1].end() :])
            values = re.findall(money, nav[1])
            if not stock or len(values) != len(data["security"]):
                raise ValueError("ACTIVE_REFINEMENT_SHARE_NAV_COLUMNS")
            period = date.fromisoformat(refinement["period_end"])
            report_quarter = re.search(r"(\d{4})年第([1-4])季度报告", data.get("notice_title", ""))
            quarter_ends = ["03-31", "06-30", "09-30", "12-31"]
            if (
                not report_quarter
                or refinement["period_end"] != report_quarter[1] + "-" + quarter_ends[int(report_quarter[2]) - 1]
            ):
                raise ValueError("ACTIVE_REFINEMENT_PERIOD_IDENTITY")
            period_pattern = rf"{period.year}年0?{period.month}月0?{period.day}日"
            if not re.search(period_pattern, re.sub(r"\s+", "", content)):
                raise ValueError("ACTIVE_REFINEMENT_PERIOD_IDENTITY")
            value = D(stock[1].replace(",", "")) / sum(D(v.replace(",", "")) for v in values) * 100
            refinements[code] = {refinement["period_end"]: {"point": value, "radius": D(0)}}
            continue
        if refinement["fund_code"] not in text:
            raise ValueError("ACTIVE_REFINEMENT_IDENTITY")
        table = next(
            (t for t in BeautifulSoup(text, "html.parser").find_all("table") if "股票占净比" in t.get_text()), None
        )
        if table is None:
            raise ValueError("ACTIVE_REFINEMENT_STOCK_NAV_HEADER")
        points = {}
        for tr in table.find_all("tr"):
            cells = [c.get_text(strip=True) for c in tr.find_all("td")]
            if len(cells) >= 2 and cells[0] in target_weights:
                if not re.fullmatch(r"\d+\.\d{2}%", cells[1]):
                    raise ValueError("ACTIVE_REFINEMENT_PERCENT_PRECISION")
                points[cells[0]] = D(cells[1][:-1])
        if set(points) != set(target_weights):
            raise ValueError("ACTIVE_REFINEMENT_ALL_QUARTERS")
        refinements[refinement["fund_code"]] = {
            day: {"point": value, "radius": D(".005")} for day, value in points.items()
        }
    required = []
    optional = []
    exposure_audit = []
    for item in exposures:
        code = item["code"]
        if code not in parent:
            continue
        if len(item["quarters"]) != 12 or item["gaps"]:
            raise ValueError("ACTIVE_STABILITY_EXPOSURE_GAP")
        if not item["region_status_by_cutoff"]["90"].startswith("REGION_PASS"):
            continue
        lows = []
        highs = []
        for quarter in item["quarters"]:
            day = quarter["date"]
            point = D(quarter["equity_nav_pct"])
            radius = D(quarter["rounding_radius_pp"])
            if day in refinements.get(code, {}):
                point = refinements[code][day]["point"]
                radius = refinements[code][day]["radius"]
            delta = abs(point - target_weights[day])
            lows.append(max(D(0), delta - radius))
            highs.append(delta + radius)
        if sum(lows) / 12 > D(5) or max(lows) > D(10):
            state = "EXCLUDED_EXPOSURE"
        elif sum(highs) / 12 <= D(5) and max(highs) <= D(10):
            required.append(code)
            state = "REQUIRED"
        else:
            optional.append(code)
            state = "POSSIBLE"
        exposure_audit.append(
            {
                "fund_code": code,
                "state": state,
                "mean_lower": str(sum(lows) / 12),
                "mean_upper": str(sum(highs) / 12),
                "maximum_lower": str(max(lows)),
                "maximum_upper": str(max(highs)),
            }
        )
    if len(optional) > 8:
        raise ValueError("ACTIVE_STABILITY_BRANCH_CAPACITY_NOT_SUPPORTED")
    all_products = {**parent, inputs["fund_code"]: inputs["target"]}
    nav_time_audit = {c: {} for c in required + optional + [inputs["fund_code"]]}
    raw = {
        c: _product(inputs, root, all_products[c], audit=nav_time_audit[c])
        for c in required + optional + [inputs["fund_code"]]
    }
    days = _calendar(inputs, root)
    months = []
    for day in days:
        if not months or day[:6] != months[-1][:6]:
            months.append(day)
        else:
            months[-1] = day
    if len(months) != 37:
        raise ValueError("ACTIVE_STABILITY_MONTH_COUNT")
    branches = []
    with localcontext() as context:
        context.prec = 60
        for flags in combinations([False, True], repeat=len(optional)):
            codes = required + [c for c, flag in zip(optional, flags, strict=True) if flag]
            if not codes:
                raise ValueError("ACTIVE_STABILITY_NO_REFERENCE")
            observations = {}
            terms = []
            for horizon, name, expected in [(1, "monthly_success", 36), (12, "annual_success", 25)]:
                rows = []
                scores = []
                unit = D(".10") / (12 if horizon == 1 else 1)
                for start, end in zip(months, months[horizon:], strict=False):
                    target = _growth(*raw[inputs["fund_code"]], days, start, end) - 1
                    lower = []
                    upper = []
                    missing = []
                    for code in codes:
                        try:
                            value = _growth(*raw[code], days, start, end) - 1
                        except ValueError as error:
                            if str(error) not in {
                                "ACTIVE_RETURN_ENDPOINT_MISSING",
                                "ACTIVE_DIVIDEND_PREVIOUS_CALENDAR_DAY_MISSING",
                                "ACTIVE_DIVIDEND_PREVIOUS_DAY_NAV_MISSING",
                            }:
                                raise
                            lower.append(D(-1))
                            upper.append(D("Infinity"))
                            missing.append({"fund_code": code, "dependency": str(error)})
                        else:
                            lower.append(value)
                            upper.append(value)
                    lo = median(lower)
                    hi = median(upper)
                    score_lo = window_frequency_score(target - hi, unit)
                    score_hi = window_frequency_score(target - lo, unit)
                    if score_lo != score_hi:
                        raise ValueError("ACTIVE_STABILITY_FREQUENCY_NOT_UNIQUE:" + start + ":" + end)
                    scores.append(score_lo)
                    rows.append(
                        {
                            "start": start,
                            "end": end,
                            "target_ratio": str(target),
                            "reference_lower_ratio": str(lo),
                            "reference_upper_ratio": str(hi),
                            "unique_frequency_point": str(score_lo),
                            "missing_peer_values": missing,
                        }
                    )
                if len(rows) != expected:
                    raise ValueError("ACTIVE_STABILITY_WINDOW_COUNT")
                observations[name] = {"kind": "bounded_windows", "expected_windows": expected, "rows": rows}
                terms.append(sum(scores) / len(scores))
            score = terms[0] * D(".3") + terms[1] * D(".7")
            branches.append(
                {
                    "optional_members": [c for c, flag in zip(optional, flags, strict=True) if flag],
                    "required_members": required,
                    "peer_count": len(codes),
                    "component_score": str(score),
                    "observations": observations,
                }
            )
    scores = {b["component_score"] for b in branches}
    if len(scores) != 1:
        raise ValueError("ACTIVE_STABILITY_MEMBERSHIP_RESULT_NOT_UNIQUE")
    return {
        "version": VERSION,
        "fund_code": inputs["fund_code"],
        "as_of_date": inputs["as_of_date"],
        "return_estimand": RETURN_ESTIMAND,
        "statistic_protocol_sha256": PROTOCOL_SHA256,
        "statistic_protocol_original_formula": PROTOCOL_FORMULA,
        "component_scores": {"stability": next(iter(scores))},
        "complete_score": None,
        "grade": None,
        "branches": branches,
        "exposure_audit": exposure_audit,
        "exposure_original_replay": original_exposure_audit,
        "time_contract": base_compiled["audit"]["time_contract"],
        "material_event_review": base_compiled["audit"]["material_event_review"],
        "nav_time_audit": nav_time_audit,
        "unknown_values_imputed": False,
        "formula": "原R5频率函数与36月/25年窗口及30/70，所有上下界及可行准入分支同点才返回点分",
    }


def calculate(inputs, *, root):
    """正式分项入口：每次核全部源字节，仅缓存同一原件包的解析，不延长日期。"""
    from app.services.fund_rating_inputs import canonical

    root = Path(root).resolve()
    for digest in inputs["sources"]:
        _bytes(inputs, root, digest)
    return deepcopy(_calculate_cached(canonical(inputs), str(root)))


@lru_cache(maxsize=4)
def _calculate_cached(serialized, root_name):
    """从原始观察/上下界重放原公式，候选分值不能作为本函数的评分输入。"""
    from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen, window_frequency_score
    from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS, policy

    inputs, root = json.loads(serialized), Path(root_name)
    mode = inputs.get("component_mode")
    if mode == "OBSERVATIONS":
        raw = compile_observations(inputs, root=root)
        observations = deepcopy(raw["observations"])
        for node in observations.values():
            node["source_refs"] = sorted(inputs["sources"])
        aliases = set(inputs["target_aliases"])
        aliases.update("CN_PUBLIC_FUND:" + code for code in inputs["target_aliases"])
        aliases.add(inputs["target"]["legal_product_id"])
        packet = {
            "fund_code": inputs["fund_code"],
            "as_of_date": inputs["as_of_date"],
            "basis": ACTIVE_BASIS,
            "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
            "quality_public": {"dataDates": {"nav": inputs["as_of_date"], "research": inputs["research_on"]}},
            "quality_metrics": {
                "observations": observations,
                "target_product_aliases": sorted(aliases),
                "material_event_pending": raw["audit"]["material_event_review"]["pending"],
            },
            "r4_evidence_registry": {
                h: {**ref, "path": str(root / ref["file"])} for h, ref in inputs["sources"].items()
            },
            "raw_archive": [{"sha256": h, "kind": ref.get("kind", "ORIGINAL")} for h, ref in inputs["sources"].items()],
        }
        scored = _calculate_frozen(packet, PROTOCOL, {}, selected_dimensions=set(inputs["selected_dimensions"]))
        dimensions = scored["dimensions"]
        if any(abs(D(value) - D(raw["component_scores"][key])) > D("1e-55") for key, value in dimensions.items()):
            raise ValueError("ACTIVE_STANDARD_RAW_OBSERVATIONS_SCORE_CONFLICT")
    elif mode == "STABILITY_BOUNDS":
        if set(inputs["selected_dimensions"]) != {"stability"}:
            raise ValueError("ACTIVE_BOUNDED_COMPONENT_SCOPE")
        raw = compile_stability_bounds(inputs, root=root)
        branch_points = []
        terms = policy(ACTIVE_BASIS)["terms"]["stability"]
        with localcontext() as context:
            context.prec = 60
            for branch in raw["branches"]:
                point = D(0)
                if set(branch["observations"]) != {name for name, _ in terms}:
                    raise ValueError("ACTIVE_BOUND_ORIGINAL_TERMS_MISSING")
                for name, weight in terms:
                    node = branch["observations"][name]
                    expected = 36 if name == "monthly_success" else 25
                    if node["expected_windows"] != expected or len(node["rows"]) != expected:
                        raise ValueError("ACTIVE_BOUND_ORIGINAL_WINDOW_COUNT")
                    values = []
                    unit = D(PROTOCOL["parameters"]["annual_return_gap_unit"]) / (12 if expected == 36 else 1)
                    for row in node["rows"]:
                        lower, upper = D(row["reference_lower_ratio"]), D(row["reference_upper_ratio"])
                        if lower > upper:
                            raise ValueError("ACTIVE_BOUND_REFERENCE_REVERSED")
                        target = D(row["target_ratio"])
                        lo = window_frequency_score(target - upper, unit)
                        hi = window_frequency_score(target - lower, unit)
                        if lo != hi or lo != D(row["unique_frequency_point"]):
                            raise ValueError("ACTIVE_BOUND_FREQUENCY_NOT_IDENTIFIED")
                        values.append(lo)
                    point += D(weight) / 100 * sum(values) / len(values)
                if abs(point - D(branch["component_score"])) > D("1e-55"):
                    raise ValueError("ACTIVE_BOUND_BRANCH_RECOMPUTATION_CONFLICT")
                branch_points.append(point)
        if not branch_points or len(set(branch_points)) != 1:
            raise ValueError("ACTIVE_BOUND_MEMBERSHIP_SCORE_NOT_IDENTIFIED")
        dimensions = {"stability": str(branch_points[0])}
        scored, packet = None, None
    else:
        raise ValueError("ACTIVE_COMPONENT_MODE_REQUIRED")
    if scored is not None and any(scored[key] is not None for key in ["score", "raw_score", "grade"]):
        raise ValueError("ACTIVE_COMPONENT_COMPLETE_TOTAL_FORBIDDEN")
    return {
        "version": VERSION,
        "fund_code": inputs["fund_code"],
        "as_of_date": inputs["as_of_date"],
        "research_on": inputs["research_on"],
        "dimensions": dimensions,
        "complete_score": None,
        "grade": None,
        "standard_calculation": scored,
        "packet": packet,
        "raw_replay": raw,
        "publication_scope": "PARTIAL_COMPONENTS_ONLY",
    }
