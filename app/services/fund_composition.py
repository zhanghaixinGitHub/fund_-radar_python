"""账户图表使用的公共持仓资料；个人金额留在Java，这里不接收任何用户信息。"""

import json
import re
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.services.fund_materials import snapshot

PROJECT = Path(__file__).resolve().parents[2]


def _latest(reports: list[dict], field: str) -> dict | None:
    rows = [r for r in reports if r.get(field)]
    return max(rows, key=lambda r: (r["endDate"], r["publishedDate"]), default=None)


def _source(report: dict, fund_code: str, fund_name: str | None = None) -> dict:
    return {
        "fundCode": fund_code,
        "fundName": fund_name,
        "title": report["title"],
        "endDate": report["endDate"],
        "publishedDate": report["publishedDate"],
        "sourceUrl": report.get("sourceUrl"),
        "fullDisclosure": bool(report.get("fullDisclosure")),
    }


def _assets(data: dict | None) -> dict | None:
    report = _latest((data or {}).get("reports", []), "assets")
    if report is None:
        return None
    names = {
        "股票": "STOCK",
        "权益投资": "STOCK",
        "债券": "BOND",
        "固定收益投资": "BOND",
        "基金": "FUND",
        "基金投资": "FUND",
    }
    grouped: dict[str, dict] = {}
    for row in report["assets"]:
        if row["name"] == "合计":
            continue
        key = names.get(row["name"], "CASH_OTHER")
        labels = {"STOCK": "股票", "BOND": "债券", "FUND": "基金", "CASH_OTHER": "现金及其他"}
        orders = {"STOCK": 10, "BOND": 20, "FUND": 30, "CASH_OTHER": 40}
        item = grouped.setdefault(key, {"name": labels[key], "order": orders[key], "weightPct": 0.0})
        item["weightPct"] += float(row["weightPct"])
    total = sum(row["weightPct"] for row in grouped.values())
    if not grouped or abs(total - 100) > 0.2:
        return None
    return {
        **_source(report, data["fundCode"]),
        "items": [
            {"key": key, "name": value["name"], "weightPct": value["weightPct"]}
            for key, value in sorted(grouped.items(), key=lambda row: row[1]["order"])
        ],
    }


def _industries(data: dict | None) -> dict | None:
    report = _latest((data or {}).get("reports", []), "industries")
    if report is None:
        return None
    stock_weight = report.get("stockWeightPct")
    industry_weight = sum(float(row["weightPct"]) for row in report["industries"])
    if stock_weight is None:
        stock_weight = industry_weight
    stock_weight = float(stock_weight)
    if stock_weight <= 0 or industry_weight <= 0 or industry_weight > stock_weight + 0.02:
        return None
    return {
        **_source(report, data["fundCode"]),
        "stockWeightPct": stock_weight,
        "items": [
            {"key": f"INDUSTRY_{index}", "name": row["name"], "weightPct": float(row["weightPct"])}
            for index, row in enumerate(report["industries"])
        ],
    }


@lru_cache(maxsize=1)
def _theme_registry() -> dict:
    path = PROJECT / "app" / "data" / "fund_security_themes_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    definitions = {row["id"]: row for row in data["themes"]}
    if len(definitions) != len(data["themes"]):
        raise ValueError("THEME_DEFINITION_DUPLICATE")
    for security, row in data["securities"].items():
        if not security or not row.get("name") or not row.get("themes"):
            raise ValueError("THEME_SECURITY_INVALID")
        for assignment in row["themes"]:
            theme_id = assignment if isinstance(assignment, str) else assignment["id"]
            if theme_id not in definitions:
                raise ValueError("THEME_SECURITY_UNKNOWN_DEFINITION")
    return data


def _theme_assignments(stock_code: str, stock_name: str) -> list[dict]:
    registry = _theme_registry()
    row = registry["securities"].get(stock_code)
    if row is None or row["name"] != stock_name:
        return []
    definitions = {item["id"]: item for item in registry["themes"]}
    results = []
    for assignment in row["themes"]:
        override = {} if isinstance(assignment, str) else assignment
        theme_id = assignment if isinstance(assignment, str) else assignment["id"]
        definition = definitions[theme_id]
        results.append(
            {
                "id": theme_id,
                "name": definition["name"],
                "sourceName": override.get("sourceName") or definition["sourceName"],
                "sourceUrl": override.get("sourceUrl") or definition.get("sourceUrl"),
                "classificationDate": override.get("classificationDate") or definition["classificationDate"],
                "method": override.get("method") or definition["method"],
            }
        )
    return results


def _normalized_name(value: str) -> str:
    return re.sub(r"[\s（）()\-—_]", "", value).casefold()


def _chain(code: str) -> tuple[list[dict], str | None]:
    """只读已核验工作流身份，不在页面请求中补采集或重新解析原件。"""
    try:
        from app.services.fund_exposure_common import read
        from app.services.fund_workflow_materials import registered_materials

        root = Path(get_settings().fund_insight_directory) / "market-workflow"
        parent = registered_materials(code, root=root)
        material = read(Path(parent["materials_path"]))
        chain = material.get("exposure_chain") or [code]
        if chain[0] != code or len(chain) > 8 or len(chain) != len(set(chain)):
            raise ValueError("COMPOSITION_EXPOSURE_CHAIN_INVALID")
        rows = []
        for target in chain:
            if target == code:
                rows.append({"fundCode": code, "fundName": material.get("fund_name")})
                continue
            target_data = snapshot(target)
            target_report = _latest((target_data or {}).get("reports", []), "holdings")
            target_name = (
                re.split(r"20\d{2}\s*年", target_report["title"])[0].strip()
                if target_report is not None
                else None
            )
            rows.append({"fundCode": target, "fundName": target_name})
        return rows, None
    except (ValueError, OSError, KeyError, TypeError, IndexError):
        return [{"fundCode": code, "fundName": None}], "底层基金关系待核验，未纳入穿透统计。"


def _themes(code: str, data: dict | None, chain: list[dict]) -> dict | None:
    reports = (data or {}).get("reports", [])
    report = _latest(reports, "holdings")
    fund_report = next(
        (
            row
            for row in reports
            if any("基金" in group.get("name", "") for group in row.get("investmentGroups") or [])
        ),
        None,
    )
    candidates = [row for row in (report, fund_report) if row is not None]
    if not candidates:
        return None
    parent = max(candidates, key=lambda row: (row["endDate"], row["publishedDate"]))
    holdings = [
        {
            **row,
            "weightPct": float(row["weightPct"]),
            "sourceFundCode": code,
            "themes": _theme_assignments(row["stockCode"], row["stockName"]),
        }
        for row in parent.get("holdings", [])
    ]
    sources = [_source(parent, code, chain[0].get("fundName"))]
    used_target = False
    for target in chain[1:]:
        target_data = snapshot(target["fundCode"])
        target_report = _latest((target_data or {}).get("reports", []), "holdings")
        target_name = target.get("fundName")
        if target_data is None or target_report is None or not target_name:
            continue
        normalized_target = _normalized_name(target_name)
        matched = None
        for group in parent.get("investmentGroups") or []:
            if "基金" not in group.get("name", ""):
                continue
            matched = next(
                (
                    row
                    for row in group.get("holdings", [])
                    if _normalized_name(row.get("stockName", "")) == normalized_target
                ),
                None,
            )
            if matched is not None:
                break
        if matched is None:
            continue
        parent_weight = float(matched["weightPct"])
        for row in target_report["holdings"]:
            holdings.append(
                {
                    **row,
                    "weightPct": parent_weight * float(row["weightPct"]) / 100,
                    "sourceFundCode": target["fundCode"],
                    "themes": _theme_assignments(row["stockCode"], row["stockName"]),
                }
            )
        sources.append(_source(target_report, target["fundCode"], target_name))
        used_target = True
    if not holdings:
        return None
    return {
        "endDate": parent["endDate"],
        "publishedDate": parent["publishedDate"],
        "fullDisclosure": bool(parent.get("fullDisclosure")) and all(row["fullDisclosure"] for row in sources),
        "disclosedWeightPct": sum(row["weightPct"] for row in holdings),
        "holdings": holdings,
        "sources": sources,
        "lookthrough": used_target,
        "themeVersion": _theme_registry()["version"],
    }


def batch(codes: list[str]) -> dict:
    if not codes or len(codes) > 100 or len(codes) != len(set(codes)):
        raise ValueError("COMPOSITION_FUND_CODES_INVALID")
    funds = []
    for code in codes:
        if not re.fullmatch(r"\d{6}", code):
            raise ValueError("INVALID_FUND_CODE")
        data = snapshot(code)
        chain, warning = _chain(code)
        funds.append(
            {
                "fundCode": code,
                "available": data is not None,
                "assets": _assets(data),
                "industries": _industries(data),
                "themes": _themes(code, data, chain),
                "warning": warning,
            }
        )
    registry = _theme_registry()
    return {
        "version": 1,
        "funds": funds,
        "themeDefinitions": [
            {"id": row["id"], "name": row["name"], "definition": row["definition"]} for row in registry["themes"]
        ],
    }
