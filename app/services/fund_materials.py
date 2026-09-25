"""公开基金资料只读查询；只返回页面白名单字段，不暴露原文件路径或采集凭据。"""

import re
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.services.fund_exposure_common import read


@lru_cache(maxsize=8)
def _snapshot(path: str, modified_ns: int) -> dict:
    """同版本快照只校验一次，替换文件后自动失效；避免翻页时反复读取数万条资料。"""
    return read(Path(path))


def snapshot(code: str) -> dict | None:
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError("INVALID_FUND_CODE")
    p = Path(get_settings().fund_material_directory) / f"{code}.json"
    return _snapshot(str(p.resolve()), p.stat().st_mtime_ns) if p.exists() else None


def overview(code: str, report_id: str | None = None, stock_code: str | None = None) -> dict:
    """报告和公司身份必须属于该基金资料集；其他基金明确返回尚无补充资料。"""
    data = snapshot(code)
    if data is None:
        return {
            "available": False,
            "fundCode": code,
            "asOfDate": None,
            "notice": "这只基金的持仓、公司经营及公告资料暂未整理。",
            "reports": [],
            "report": None,
            "companies": [],
            "company": None,
        }
    report = next((r for r in data["reports"] if r["id"] == report_id), None) if report_id else data["reports"][0]
    if report is None:
        raise ValueError("MATERIAL_REPORT_NOT_FOUND")
    company = next((c for c in data["companies"] if c["stockCode"] == stock_code), None) if stock_code else None
    if stock_code and company is None:
        raise ValueError("MATERIAL_COMPANY_NOT_FOUND")
    return {
        "available": True,
        "fundCode": code,
        "asOfDate": data["asOfDate"],
        "notice": data["notice"],
        "reports": [
            {k: r[k] for k in ("id", "title", "endDate", "publishedDate", "fullDisclosure", "sourceUrl")}
            for r in data["reports"]
        ],
        "report": report,
        "companies": [{k: c[k] for k in ("stockCode", "stockName", "latestHeld")} for c in data["companies"]],
        "company": company,
    }


def documents(
    code: str,
    *,
    page: int = 1,
    size: int = 20,
    kind: str = "all",
    stock_code: str | None = None,
    keyword: str = "",
    latest_only: bool = False,
) -> dict:
    """先按明确范围筛选再分页；历史关联与最新披露持仓关联分别标注。"""
    data = snapshot(code)
    rows = [] if data is None else data["documents"]
    rows = [
        r
        for r in rows
        if (kind == "all" or r["kind"] == kind)
        and (not stock_code or r["stockCode"] == stock_code)
        and (not keyword or keyword.casefold() in r["title"].casefold())
        and (not latest_only or r["kind"] != "company" or r["latestHeld"])
    ]
    return {
        "items": rows[(page - 1) * size : page * size],
        "total": len(rows),
        "page": page,
        "pageSize": size,
        "asOfDate": data["asOfDate"] if data else None,
    }
