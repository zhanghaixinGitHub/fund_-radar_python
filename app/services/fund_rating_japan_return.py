"""019449日本主动QDII收益原件组件；仅重放收益，不采集、不发布。

参照为独立法律产品Matthews Japan Active ETF（JPAN）。12季实际股票仓位沿用
既有收益准入10/20个百分点界，不把该参照扩到风险、稳定性、持仓或流动性。
"""
from __future__ import annotations

import json
import re
import unicodedata
import zipfile
from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup

from app.services.fund_rating_active_execution import Originals
from app.services.fund_rating_active_performance import _growth
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS

VERSION = "R5_JAPAN_QDII_RETURN_ORIGINALS_20261110_1"
FIELD = "japan_return_calculation"
PEER_ID = "US_SEC_SERIES:S000081865"
EXPECTED_QUARTERS = [
    "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
    "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30",
]


def require(condition, code):
    if not condition:
        raise ValueError("R5_JAPAN_RETURN_" + code)


def plain(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)))


def xml_text(node, name, namespace):
    value = node.findtext("n:" + name, namespaces=namespace)
    require(value is not None, "NPORT_FIELD:" + name)
    return value


def peer_quarter(raw: bytes, expected_period: str) -> dict:
    namespace = {"n": "http://www.sec.gov/edgar/nport"}
    root = ET.fromstring(raw)
    general = root.find("n:formData/n:genInfo", namespace)
    fund = root.find("n:formData/n:fundInfo", namespace)
    require(general is not None and fund is not None, "NPORT_STRUCTURE")
    require(xml_text(general, "seriesId", namespace) == "S000081865", "PEER_SERIES")
    require(xml_text(general, "seriesName", namespace) == "Matthews Japan Active ETF", "PEER_NAME")
    actual = xml_text(general, "repPdDate", namespace)
    date.fromisoformat(actual)
    require(actual[:7] == expected_period[:7], "PEER_QUARTER:" + expected_period)
    positions = []
    for position in root.findall(".//n:invstOrSec", namespace):
        row = {name: xml_text(position, name, namespace) for name in ["valUSD", "assetCat", "issuerCat", "payoffProfile"]}
        positions.append(row)
    stock = sum(
        (D(row["valUSD"]) for row in positions
         if row["assetCat"] in {"EC", "EP"} and row["issuerCat"] == "CORP" and row["payoffProfile"] == "Long"),
        D(0),
    )
    nav = D(xml_text(fund, "netAssets", namespace))
    assets = D(xml_text(fund, "totAssets", namespace))
    liabilities = D(xml_text(fund, "totLiabs", namespace))
    require(positions and stock > 0 and nav > 0 and assets - liabilities == nav, "NPORT_RECONCILIATION")
    return {"quarter_label": expected_period, "actual_report_date": actual, "stock_USD": str(stock),
            "all_class_NAV_USD": str(nav), "stock_NAV_ratio": str(stock / nav), "position_rows": len(positions)}


def legal_review(source: Originals, inputs: dict) -> list[dict]:
    reviews = []
    for digest in inputs["peer_legal_refs"]:
        text = source.body(digest)
        compact = plain(text)
        require(
            "MatthewsJapanActiveETF" in compact
            and "Long-termcapitalappreciation" in compact
            and "atleast80%ofitsnetassets" in compact
            and "commonandpreferredstocksofcompanieslocatedinJapan" in compact
            and "companiesofanymarketcapitalization" in compact,
            "PEER_LEGAL_SCOPE:" + digest,
        )
        reviews.append({"source_ref": digest, "filed_on": source.sources[digest]["published_on"]})
    require(len(reviews) == 5 and len({r["filed_on"] for r in reviews}) == 5, "PEER_LEGAL_PERIOD_COVERAGE")
    require(min(r["filed_on"] for r in reviews) <= "2023-09-28", "PEER_INITIAL_LEGAL_SCOPE")
    return reviews


def target_history(source: Originals, inputs: dict) -> tuple[dict, dict]:
    provider = source.json(inputs["target_provider_ref"])
    detail = source.json(inputs["target_detail_ref"])
    nav = source.json(inputs["target_nav_ref"])
    from app.services.fund_rating_public_japan import issuer_history

    history = issuer_history(provider, detail, nav, date.fromisoformat(inputs["as_of_date"]))
    info = detail["data"]["product_info"]
    envelope = nav["data"]["product_nav"][info["product_id"]]
    rows = {}
    for row in envelope["list"]:
        day = date(int(row["nav_date"][:4]), int(row["nav_date"][4:6]), int(row["nav_date"][6:])).isoformat()
        if inputs["window_start"] <= day <= inputs["as_of_date"]:
            require(day not in rows, "TARGET_NAV_DUPLICATE")
            require(
                D(row["relate_price"]) == D(row["cumulative_net"]) == D(row["adjust_nav"]) > 0,
                "TARGET_DISTRIBUTION_OR_SPLIT_REVIEW",
            )
            rows[day] = D(row["adjust_nav"])
    require(set(history["dates"]) == set(rows), "TARGET_OFFICIAL_DATE_SCOPE")
    return rows, history["issuer_valuation_verification"]


def peer_product(source: Originals, inputs: dict) -> tuple[list[str], dict, list[dict], str]:
    raw = source.raw[inputs["peer_product_ref"]]
    soup = BeautifulSoup(raw.decode("utf-8"), "html.parser")
    charts = [json.loads(node["chart-data"], parse_float=D) for node in soup.select(".series-chart[chart-data]")]
    matches = [chart for chart in charts if any(series["name"] == "NAV" for series in chart["series"])]
    require(len(matches) == 1, "PEER_NAV_CHART_UNIQUE")
    chart = matches[0]
    nav = next(series for series in chart["series"] if series["name"] == "NAV")
    require(len(chart["categories"]) == len(nav["data"]) and nav["data"], "PEER_NAV_CHART_ROWS")
    points = {}
    for label, value in zip(chart["categories"], nav["data"], strict=True):
        day = datetime.strptime(label, "%m/%d/%Y").date().isoformat()
        require(day not in points and value > 0, "PEER_NAV_IDENTITY_OR_VALUE")
        points[day] = value
    require(points.get(inputs["window_start"]) and points.get(inputs["as_of_date"]), "PEER_NAV_ENDPOINT")
    text = soup.get_text(" ", strip=True)
    convention = (
        "Total Return reflects the reinvestment of distributions on ex-date for NAV returns "
        "and payment date for Market Price returns."
    )
    require(convention in text, "PEER_REINVESTMENT_CONVENTION")
    rows = [[cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"])]
            for tr in soup.select("#tblDistributionsFundDetail tr")]
    require(rows and rows[0][:4] == ["Ex Date", "Record Date", "Pay Date", "Reinvest Date"], "PEER_DIVIDEND_TABLE")
    events = []
    for row in rows[1:]:
        require(len(row) == 9, "PEER_DIVIDEND_ROW")
        dates = [datetime.strptime(value, "%m/%d/%Y").date().isoformat() for value in row[:4]]
        amounts = [D(value.removeprefix("$")) for value in row[4:8]]
        require(sum(amounts[:3]) == amounts[3] and dates[0] == dates[3], "PEER_DIVIDEND_RECONCILIATION")
        require(dates[0] in points and amounts[3] >= 0, "PEER_DIVIDEND_NAV")
        if inputs["window_start"] <= dates[0] <= inputs["as_of_date"]:
            events.append({"date": dates[0], "cash_USD": amounts[3], "original_cells": row})
    require(len(events) == 3, "PEER_DIVIDEND_WINDOW_COVERAGE")
    return sorted(day for day in points if inputs["window_start"] <= day <= inputs["as_of_date"]), points, events, convention


def fx_history(source: Originals, inputs: dict) -> dict:
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(__import__("io").BytesIO(source.raw[inputs["fx_history_ref"]])) as archive:
        shared = ["".join(item.itertext()) for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall("m:sheetData/m:row", namespace):
        values = []
        for cell in row:
            value = cell.find("m:v", namespace)
            text = value.text if value is not None else "".join(cell.itertext())
            values.append(shared[int(text)] if cell.get("t") == "s" else text)
        rows.append(values)
    require(rows[0][:4] == ["日期", "美元", "欧元", "日元"], "FX_HISTORICAL_HEADER")
    result = {}
    for row in rows[1:]:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", row[0]) and inputs["window_start"] <= row[0] <= "2025-12-31":
            result[row[0]] = D(row[1]) / 100
    current = BeautifulSoup(source.raw[inputs["fx_current_ref"]].decode("utf-8"), "html.parser")
    tables = []
    for table in current.find_all("table"):
        rows = [[cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
                for row in table.find_all("tr")]
        if rows and rows[0] and rows[0][0] == "日期" and len(rows[0]) == 26:
            tables.append(rows)
    require(len(tables) == 1, "FX_CURRENT_TABLE_UNIQUE")
    for row in tables[0][1:]:
        if len(row) == 26 and "2026-01-01" <= row[0] <= inputs["as_of_date"]:
            require(row[0] not in result, "FX_DUPLICATE_DATE")
            result[row[0]] = D(row[1]) / 100
    require(result and all(value > 0 for value in result.values()), "FX_VALUE")
    return result


def calculate(inputs: dict, *, root: Path) -> dict:
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "019449"
        and inputs.get("share_class") == "C"
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("holding_date") == "2026-06-30"
        and inputs.get("window_start") == "2023-09-28",
        "INPUT_SCOPE",
    )
    require(not any(key in inputs for key in ["score", "dimensions", "observations", "complete_score", "grade"]), "PRECOMPUTED")
    require(set(inputs["peer_quarter_refs"]) == set(EXPECTED_QUARTERS), "PEER_QUARTER_SCOPE")
    source = Originals(inputs, root)
    from app.services.fund_rating_japan_active_execution import calculate as calculate_execution

    execution = calculate_execution(inputs["execution_inputs"], root=root)
    require(execution["material_event_review"]["material_event_pending"] is False, "MATERIAL_REVIEW")
    legal = legal_review(source, inputs)
    target, target_audit = target_history(source, inputs)
    peer_days, peer, events, convention = peer_product(source, inputs)
    fx = fx_history(source, inputs)
    common = sorted(set(target) & set(peer_days) & set(fx))
    require(common[0] == inputs["window_start"] and common[-1] == inputs["as_of_date"], "COMMON_ENDPOINT")
    require("2025-09-30" in common, "ONE_YEAR_ENDPOINT")
    peer_cny = {day: peer[day] * fx[day] for day in common}
    cash_events = []
    for event in events:
        day = event["date"]
        require(day in common and common.index(day) > 0, "DIVIDEND_COMMON_DATE")
        previous = common[common.index(day) - 1]
        cash_events.append({"date": day, "cash": event["cash_USD"] * fx[previous], "original_cells": event["original_cells"]})
    quarters = []
    target_rows = {row["period"]: row for row in execution["report_ledger"]}
    for period in EXPECTED_QUARTERS:
        row = peer_quarter(source.raw[inputs["peer_quarter_refs"][period]], period)
        own = D(target_rows[period]["stock_NAV_ratio"])
        difference = abs(D(row["stock_NAV_ratio"]) - own) * 100
        quarters.append({**row, "target_stock_NAV_ratio": str(own), "absolute_difference_pp": str(difference),
                         "source_ref": inputs["peer_quarter_refs"][period]})
    mean = sum(D(row["absolute_difference_pp"]) for row in quarters) / 12
    maximum = max(D(row["absolute_difference_pp"]) for row in quarters)
    require(mean <= 10 and maximum <= 20, "RETURN_PEER_STOCK_BUDGET")
    with localcontext() as context:
        context.prec = 70
        annualization = D("365.25") / D((date.fromisoformat(inputs["as_of_date"]) - date.fromisoformat(inputs["window_start"])).days)
        own_1y = target[inputs["as_of_date"]] / target["2025-09-30"] - 1
        own_3y = (target[inputs["as_of_date"]] / target[inputs["window_start"]]) ** annualization - 1
        peer_1y = _growth(peer_cny, cash_events, common, "2025-09-30", inputs["as_of_date"]) - 1
        peer_3y = _growth(peer_cny, cash_events, common, inputs["window_start"], inputs["as_of_date"]) ** annualization - 1
        observations = {}
        for name, own, reference in [("return_1y", own_1y, peer_1y), ("return_3y", own_3y, peer_3y)]:
            observations[name] = {"kind": "rank", "direction": "higher", "target": str(own),
                                  "peers": [{"product_id": PEER_ID, "value": str(reference)}], "expected_products": [PEER_ID]}
        registry = {digest: {**ref, "path": str((root / ref["file"]).resolve())}
                    for digest, ref in source.sources.items()}
        packet = {
            "fund_code": "019449", "basis": ACTIVE_BASIS, "as_of_date": inputs["as_of_date"],
            "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
            "quality_metrics": {"target_product_aliases": ["CN_PUBLIC_FUND:007280", "CN_PUBLIC_FUND:019449"],
                                "material_event_pending": False, "observations": observations},
            "r4_evidence_registry": registry,
            "quality_public": {"dataDates": {"nav": inputs["as_of_date"], "holdings": inputs["holding_date"]}},
            "raw_archive": [{"sha256": digest, "kind": "ORIGINAL"} for digest in sorted(source.sources)],
        }
        result = _calculate_frozen(packet, PROTOCOL, execution["anchors"], selected_dimensions={"return"})
    require(result["score"] is None and result["grade"] is None, "COMPONENT_ONLY")
    return {
        "version": VERSION, "dimensions": result["dimensions"], "subscores": result["subscores"],
        "observations": observations, "complete_score": None, "grade": None,
        "audit": {
            "peer": {"legal_product_id": PEER_ID, "ticker": "JPAN", "native_currency": "USD"},
            "legal_reviews": legal, "quarters": quarters,
            "stock_budget_mean_difference_pp": str(mean), "stock_budget_max_difference_pp": str(maximum),
            "common_valuation_fx_days": len(common), "target_nav": target_audit,
            "peer_distributions": cash_events, "peer_total_return_convention": convention,
            "fx_conversion": "JPAN USD NAV × same-day SAFE/CFETS CNY per USD; distribution cash uses previous common day's fixing",
            "return_estimand": "R5_FROZEN_DISCLOSED_NAV_GROWTH_20261019",
            "risk_stability_holdings_liquidity_not_admitted_by_this_component": True,
        },
    }


def check_binding(bundle: dict, inputs: dict) -> None:
    require(
        bundle.get("basis") == "PUBLIC_REVIEWED_QUALITY_SNAPSHOT_V3"
        and bundle.get("family") == "QDII_EQUITY"
        and bundle.get("fund_code") == inputs.get("fund_code") == "019449"
        and bundle.get("share_class") == inputs.get("share_class") == "C"
        and str(bundle.get("as_of_date")) == inputs.get("as_of_date") == "2026-09-30"
        and str(bundle.get("holding_date")) == inputs.get("holding_date") == "2026-06-30"
        and inputs.get("legal_name") == bundle.get("identity", {}).get("legal_name")
        and inputs.get("manager_name") == bundle.get("identity", {}).get("manager_name")
        and inputs.get("primary_ref") == bundle.get("original_report_archive", {}).get("sha256"),
        "SNAPSHOT_BINDING",
    )
    require(bundle.get("japan_active_execution_calculation", {}).get("inputs") == inputs.get("execution_inputs"),
            "EXECUTION_INPUT_BINDING")
    require(not any(bundle.get(key) for key in ["r5_calculation", "passive_components_calculation",
                                                  "active_performance_calculations"]), "CONFLICTING_COMPONENT")


def prepare(session, request: dict, *, root: Path) -> dict:
    """只读当前019449快照并准备单收益组件；不提交数据库。"""
    from app.services.fund_rating_pipeline import archive_json, current_inputs
    from app.services.fund_rating_pipeline_prepare import canonical_source
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    code = request.get("fund_code")
    previous = current_inputs(session, [code]).get(code) if code == "019449" else None
    require(previous is not None and previous["batch_id"] == request.get("expected_current_ref")
            and str(previous["as_of_date"]) == request.get("as_of_date") == "2026-09-30", "CURRENT_PERIOD_AND_REF")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    raw = deepcopy(request["inputs"])
    raw["execution_inputs"] = deepcopy(bundle["japan_active_execution_calculation"]["inputs"])
    check_binding(bundle, raw)
    require(FIELD not in bundle and raw["as_of_date"] == request["as_of_date"], "COMPONENT_OR_PERIOD")
    raw["sources"] = {digest: canonical_source(root, ref) for digest, ref in raw["sources"].items()}
    calculated = calculate(raw, root=root)
    explanation = request.get("explanation", "").strip()
    require(len(explanation) >= 40, "PUBLIC_EXPLANATION")
    point = calculated["dimensions"]["return"]
    document = bundle["reviewed_document"]
    document["dimensions"]["return"] = {"lower": point, "upper": point, "candidate": point, "reason": explanation}
    bundle.setdefault("public_explanations", {})["return"] = explanation
    bundle.get("public_metrics", {}).pop("return", None)
    bundle["data_dates"] = {**bundle.get("data_dates", {}), "nav": raw["as_of_date"]}
    bundle["comparison_product_count"] = 1
    bundle.update(research_format="single", research_archive=archive_json(root, document),
                  **{FIELD: {"inputs": raw, "input_archive": archive_json(root, raw)}})
    archives = {ref["sha256"]: ref for ref in bundle.get("extra_archives", [])}
    archives.update({digest: {key: ref[key] for key in ["file", "sha256", "kind"]}
                     for digest, ref in raw["sources"].items()})
    bundle["extra_archives"] = list(archives.values())
    validate_sources(bundle, datetime.now(UTC), root=root)
    checked = snapshot_calculate(bundle, root=root)
    require(checked["status"] == "PARTIAL" and checked["score"] is None and checked["grade"] is None,
            "COMPONENT_CANNOT_CLAIM_COMPLETE")
    return {"basis": BASIS, "entries": [{key: value for key, value in bundle.items()
                                         if key not in {"basis", "reviewed_document"}}]}


def replay_snapshot(bundle: dict, dimensions: dict, *, root: Path) -> dict:
    raw = bundle[FIELD]["inputs"]
    check_binding(bundle, raw)
    result = calculate(raw, root=root)
    point = D(result["dimensions"]["return"])
    for bound in ("lower", "upper", "candidate"):
        value = dimensions["return"].get(bound)
        require(value is not None and D(str(value)) == point, "SNAPSHOT_RECALCULATION:" + bound)
    return result
