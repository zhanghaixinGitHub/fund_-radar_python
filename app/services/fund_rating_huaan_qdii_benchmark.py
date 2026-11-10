"""华安纳斯达克100联接本C的合同基准、风险及目标执行原件编译。"""

from __future__ import annotations

import calendar as month_calendar
import json
import re
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

from app.services.fund_rating_direct_index_math import performance
from app.services.fund_rating_huaan_qdii_relative import (
    Originals,
    _contracts,
    _events,
    _evidence,
    _history,
    check_binding as check_relative_binding,
    compact,
)
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS

VERSION = "R5_HUAAN_QDII_BENCHMARK_20261110_1"
DIMENSIONS = {"risk", "efficiency"}
WINDOW_LABELS = (("过去三个月", 3), ("过去六个月", 6), ("过去一年", 12))
REPORT_ENDS = {"2024": "2024-12-31", "2025": "2025-12-31", "2026": "2026-06-30"}


def _index(source: Originals, inputs: dict) -> dict[str, D]:
    raw = source.json(inputs["index_ref"])
    rows = raw.get("aaData")
    if not isinstance(rows, list) or raw.get("iTotalRecords") != len(rows) or raw.get("iTotalDisplayRecords") != len(rows):
        raise ValueError("HUAAN_QDII_INDEX_RESPONSE_INCOMPLETE")
    result = {}
    for row in rows:
        match = re.fullmatch(r"/Date\((\d+)\)/", row.get("TradeDate") or "")
        if not match or row["TradeDate"] != row.get("TimeStamp") or row.get("Currency") != "USD":
            raise ValueError("HUAAN_QDII_INDEX_IDENTITY")
        day = datetime.fromtimestamp(int(match[1]) / 1000, ZoneInfo("America/New_York")).date().isoformat()
        value = row.get("Value")
        if day in result or value is None or D(str(value)) <= 0:
            raise ValueError("HUAAN_QDII_INDEX_DATE_OR_VALUE")
        result[day] = D(str(value))
    if not result or min(result) > inputs["window_start"] or max(result) < inputs["as_of_date"]:
        raise ValueError("HUAAN_QDII_INDEX_WINDOW")
    return result


def _fx(source: Originals, inputs: dict) -> dict[str, D]:
    result = {}
    for digest in inputs["fx_refs"]:
        soup = BeautifulSoup(source.raw[digest].decode("utf-8-sig"), "html.parser")
        table = soup.find("table", id="InfoTable")
        if table is None:
            raise ValueError("HUAAN_QDII_FX_TABLE")
        rows = [[cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])] for row in table.find_all("tr")]
        if not rows or rows[0][:2] != ["日期", "美元"]:
            raise ValueError("HUAAN_QDII_FX_COLUMNS")
        for row in rows[1:]:
            day = date.fromisoformat(row[0]).isoformat()
            if len(row) < 2 or day in result or not re.fullmatch(r"\d+\.\d+", row[1]):
                raise ValueError("HUAAN_QDII_FX_DATE_OR_DUPLICATE")
            result[day] = D(row[1]) / 100
    if not result or min(result) > inputs["window_start"] or max(result) < inputs["as_of_date"]:
        raise ValueError("HUAAN_QDII_FX_WINDOW")
    return result


def _latest(values: dict[str, D], day: str) -> tuple[str, D]:
    eligible = [key for key in values if key <= day]
    if not eligible:
        raise ValueError("HUAAN_QDII_NO_ACTUAL_PRIOR_VALUE")
    key = max(eligible)
    return key, values[key]


def _month_start(end: str, months: int) -> str:
    year, month, day = map(int, end.split("-"))
    month -= months
    while month <= 0:
        month += 12
        year -= 1
    end_is_month_end = day == month_calendar.monthrange(int(end[:4]), int(end[5:7]))[1]
    start_day = month_calendar.monthrange(year, month)[1] if end_is_month_end else min(
        day, month_calendar.monthrange(year, month)[1]
    )
    return date(year, month, start_day).isoformat()


def _published_windows(source: Originals, inputs: dict, days: list[str]) -> list[dict]:
    checks = []
    for year, end in REPORT_ENDS.items():
        digest = inputs["members"][inputs["fund_code"]]["contract_refs"][year]
        plain = compact(source.text(digest))
        table_end = plain.find("3.2.2")
        table_start = plain.rfind("华安纳斯达克100ETF联接(QDII)C", 0, table_end)
        if table_start < 0 or table_end < 0:
            raise ValueError("HUAAN_QDII_C_BENCHMARK_TABLE:" + year)
        table = plain[table_start:table_end]
        for label, months in WINDOW_LABELS:
            match = re.search(label + r"(-?[\d.]+)%(-?[\d.]+)%(-?[\d.]+)%(-?[\d.]+)%", table)
            if not match:
                raise ValueError("HUAAN_QDII_PUBLISHED_BENCHMARK_MISSING:" + year + ":" + label)
            nominal = _month_start(end, months)
            first = max(day for day in days if day <= nominal)
            last = max(day for day in days if day <= end)
            checks.append(
                {
                    "report_year": year,
                    "label": label,
                    "source_hash": digest,
                    "source_quote": match[0],
                    "nominal_start": nominal,
                    "start": first,
                    "end": last,
                    "published_percent": match[3],
                }
            )
    if len(checks) != 9:
        raise ValueError("HUAAN_QDII_NINE_PUBLISHED_WINDOWS_REQUIRED")
    return checks


def _benchmark(source: Originals, inputs: dict, days: list[str]) -> dict:
    protocol = source.json(inputs["benchmark_protocol_ref"])
    if (
        protocol.get("fund_code") != inputs["fund_code"]
        or protocol.get("as_of") != inputs["as_of_date"]
        or protocol.get("common_days") != "OWN_727_ISSUER_CROSS_CHECKED_VALUATION_DATES"
        or protocol.get("index") != "NDX_OFFICIAL_EOD"
        or protocol.get("fx") != "SAFE_PBOC_USD_CNY_MIDPOINT_LATEST_PUBLICATION"
        or protocol.get("annual_cash_rate") != ".0035"
        or protocol.get("no_imputation") is True
    ):
        raise ValueError("HUAAN_QDII_BENCHMARK_PROTOCOL")
    rate_text = compact(source.text(inputs["cash_rate_ref"]))
    if "活期存款0.350.35" not in rate_text:
        raise ValueError("HUAAN_QDII_CASH_RATE_ORIGINAL")
    indices, fx = _index(source, inputs), _fx(source, inputs)
    prices, provenance = {}, {}
    for day in days:
        index_day, index = _latest(indices, day)
        fx_day, rate = _latest(fx, day)
        prices[day] = index * rate
        provenance[day] = {"index_date": index_day, "fx_date": fx_day}
    with localcontext() as context:
        context.prec = 70
        levels = {days[0]: D(1)}
        for previous, current in zip(days, days[1:]):
            gap = (date.fromisoformat(current) - date.fromisoformat(previous)).days
            cash = D(".05") * D(protocol["annual_cash_rate"]) * gap / 365
            levels[current] = levels[previous] * (1 + D(".95") * (prices[current] / prices[previous] - 1) + cash)
        checks = _published_windows(source, inputs, days)
        for check in checks:
            calculated = (levels[check["end"]] / levels[check["start"]] - 1) * 100
            published = D(check["published_percent"])
            check["calculated_percent"] = str(calculated)
            check["published_rounding_interval"] = [str(published - D(".005")), str(published + D(".005"))]
            check["pass"] = published - D(".005") <= calculated <= published + D(".005")
        if not all(check["pass"] for check in checks):
            raise ValueError("HUAAN_QDII_PUBLISHED_BENCHMARK_PRECISION_CONFLICT")
    return {
        "levels": [str(levels[day]) for day in days],
        "source_dates": provenance,
        "published_window_checks": checks,
        "index_rows": len(indices),
        "fx_rows": len(fx),
        "annual_cash_rate": protocol["annual_cash_rate"],
        "scope": "合同构造参照；九项定报同窗收益按原0.01个百分点展示精度验收，非基金管理人逐日原值。",
    }


def calculate(inputs: dict, *, root: Path) -> dict:
    from app.services.fund_rating_huaan_qdii_relative import VERSION as RELATIVE_VERSION

    if (
        inputs.get("version") != RELATIVE_VERSION
        or inputs.get("fund_code") != "014978"
        or inputs.get("as_of_date") != "2026-09-30"
        or inputs.get("window_start") != "2023-09-28"
        or inputs.get("basis") != PASSIVE_BASIS
        or set(inputs.get("requested_dimensions", [])) != {"return", "stability"}
    ):
        raise ValueError("HUAAN_QDII_BENCHMARK_SCOPE")
    source = Originals(inputs, root)
    event_review = _events(source, inputs)
    if event_review["material_event_pending"]:
        raise ValueError("HUAAN_QDII_BENCHMARK_MATERIAL_EVENT_PENDING")
    _evidence(source, inputs)
    contracts = _contracts(source, inputs)
    history = _history(source, inputs, contracts["peers"])
    benchmark = _benchmark(source, inputs, history["dates"])
    values = {}
    with localcontext() as context:
        context.prec = 70
        for member, nav in history["values"].items():
            values[member] = performance(
                history["dates"], history["month_ends"], nav, benchmark["levels"]
            )
    code, peers = inputs["fund_code"], contracts["peers"]
    all_refs = sorted(source.refs)
    observations = {}
    for name in ["extra_drawdown", "extra_downside", "tracking_error"]:
        observations[name] = {
            "kind": "rank",
            "direction": "lower",
            "target": values[code][name],
            "expected_products": peers,
            "peers": [{"product_id": peer, "value": values[peer][name]} for peer in peers],
            "source_refs": all_refs,
        }
    registry = {
        digest: {"path": source.path(digest), "sha256": digest, "published_on": ref.get("available_on")}
        for digest, ref in source.refs.items()
    }
    packet = {
        "basis": PASSIVE_BASIS,
        "fund_code": code,
        "as_of_date": inputs["as_of_date"],
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_metrics": {
            "observations": observations,
            "target_product_aliases": inputs["target_product_aliases"],
            "material_event_pending": False,
        },
        "r4_evidence_registry": registry,
        "raw_archive": [
            {"kind": "ORIGINAL", "sha256": digest} for digest, ref in source.refs.items() if ref["kind"] == "ORIGINAL"
        ],
        "quality_public": {
            "dataDates": {"nav": inputs["as_of_date"], "holdings": "2026-06-30", "manager": "2026-06-30", "fees": "2026-09-18"}
        },
    }
    tracking_ref = inputs["evidence"]["own_tracking_contract"]["source_refs"][0]
    tracking_quote = inputs["evidence"]["own_tracking_contract"]["quotes"][0]
    match = re.search(r"年跟踪误差不超过([\d.]+)%", compact(tracking_quote))
    if not match:
        raise ValueError("HUAAN_QDII_TRACKING_CONTRACT_LIMIT")
    anchor = {
        "path": source.path(tracking_ref),
        "sha256": tracking_ref,
        "published_on": source.refs[tracking_ref]["available_on"],
        "source_verified": True,
        "annual_tracking_error_limit_ratio": str(D(match[1]) / 100),
        "actual_clause": tracking_quote,
    }
    anchors = {"material_events": {code: event_review["material_event_anchors"]}, code: anchor}
    computed = _calculate_frozen(packet, PROTOCOL, anchors, selected_dimensions=DIMENSIONS)
    return {
        "version": VERSION,
        "fund_code": code,
        "as_of_date": inputs["as_of_date"],
        "status": "READY_FOR_INTEGRATION",
        "dimensions": computed["dimensions"],
        "subscores": computed["subscores"],
        "score": None,
        "grade": None,
        "publishable": False,
        "observations": observations,
        "compiled_anchors": anchors,
        "audit": {
            "benchmark": benchmark,
            "contracts": contracts["contracts"],
            "performance": values,
            "calendar_count": len(history["dates"]),
            "peers": peers,
            "actual_event_request_identities": history["actual_event_request_identities"],
        },
    }


def prepare(session, request: dict, *, root: Path) -> dict:
    from app.services.fund_rating_pipeline import archive_json, current_inputs
    from app.services.fund_rating_reviewed_snapshot import BASIS, calculate as snapshot_calculate, validate_sources

    code = request.get("fund_code")
    if code != "014978" or request.get("as_of_date") != "2026-09-30":
        raise ValueError("HUAAN_QDII_BENCHMARK_REQUEST_SCOPE")
    previous = current_inputs(session, [code]).get(code)
    if not previous or str(previous["as_of_date"]) != request["as_of_date"]:
        raise ValueError("HUAAN_QDII_BENCHMARK_CURRENT_PERIOD")
    bundle = deepcopy(
        next(member["bundle"] for member in previous["input_snapshot"]["members"] if member["fund_code"] == code)
    )
    if bundle.get("basis") != BASIS or bundle.get("huaan_qdii_benchmark_calculation"):
        raise ValueError("HUAAN_QDII_BENCHMARK_CURRENT_SNAPSHOT")
    relative = bundle.get("huaan_qdii_relative_calculation")
    if not relative:
        raise ValueError("HUAAN_QDII_RELATIVE_COMPONENT_REQUIRED")
    raw = deepcopy(relative["inputs"])
    check_relative_binding(bundle, raw)
    document = bundle["reviewed_document"]
    points = {
        name
        for name, node in document["dimensions"].items()
        if node is not None and (not isinstance(node, dict) or node.get("candidate") is not None)
    }
    if points != {"return", "stability", "management", "fee"}:
        raise ValueError("HUAAN_QDII_EXPECTED_FOUR_POINT_BUNDLE")
    calculated = calculate(raw, root=root)
    for name in sorted(DIMENSIONS):
        snapshot_name = "execution" if name == "efficiency" else name
        reason = request["explanations"][name].strip()
        if len(reason) < 40:
            raise ValueError("HUAAN_QDII_BENCHMARK_EXPLANATION")
        point = calculated["dimensions"][name]
        document["dimensions"][snapshot_name] = {"lower": point, "upper": point, "candidate": point, "reason": reason}
        bundle.setdefault("public_explanations", {})[snapshot_name] = reason
        bundle.get("public_metrics", {}).pop(snapshot_name, None)
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, document),
        huaan_qdii_benchmark_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    validate_sources(bundle, datetime.now(ZoneInfo("UTC")), root=root)
    checked = snapshot_calculate(bundle, root=root)
    if checked["status"] != "PARTIAL" or checked["point_weight"] != "80" or checked["score"] is not None:
        raise ValueError("HUAAN_QDII_BENCHMARK_MUST_REMAIN_PARTIAL")
    return {
        "basis": BASIS,
        "entries": [{key: value for key, value in bundle.items() if key not in {"basis", "reviewed_document"}}],
    }


def replay_snapshot(bundle: dict, dimensions: dict, *, root: Path) -> dict:
    raw = bundle["huaan_qdii_benchmark_calculation"]["inputs"]
    check_relative_binding(bundle, raw)
    if raw != bundle.get("huaan_qdii_relative_calculation", {}).get("inputs"):
        raise ValueError("HUAAN_QDII_BENCHMARK_RELATIVE_INPUT_MISMATCH")
    result = calculate(raw, root=root)
    for name, point in result["dimensions"].items():
        snapshot_name = "execution" if name == "efficiency" else name
        expected = D(str(point))
        node = dimensions.get(snapshot_name)
        if not isinstance(node, dict):
            raise ValueError("HUAAN_QDII_BENCHMARK_SNAPSHOT_DIMENSION:" + name)
        for bound in ("lower", "upper", "candidate"):
            value = node.get(bound)
            if value is None or abs(D(str(value)) - expected) > D("1e-55"):
                raise ValueError("HUAAN_QDII_BENCHMARK_SNAPSHOT_RECALCULATION:" + name + ":" + bound)
    return result
