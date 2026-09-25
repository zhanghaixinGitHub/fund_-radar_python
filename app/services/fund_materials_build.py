"""将已核验的 002112 研究资料转换为页面只读快照；不会抓取来源或调用大模型。"""

import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from app.core.config import get_settings
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import ROOT, now, read, save
from app.services.fund_materials_store import source_files, source_path


def iso_date(value) -> str | None:
    """来源的 YYYYMMDD 转日期；未知值不伪装为采集当天。"""
    s = str(value or "")
    try:
        return datetime.strptime(s, "%Y%m%d").date().isoformat() if len(s) == 8 else None
    except ValueError:
        return None


def safe_url(value) -> str | None:
    """页面只接收公开 http(s) 来源链接，不允许脚本、凭据或本地文件地址。"""
    s = str(value or "")
    p = urlparse(s)
    return s if p.scheme in {"https", "http"} and p.hostname and not p.username and not p.password else None


def financial_rows(root: Path, code: str, api: str) -> list[dict]:
    """优先完整字段版本；两个版本不能相加，文件哈希异常直接停止发布。"""
    for folder in ("enriched-financials", "financials"):
        p = source_path(root, f"supplement/{folder}/{code}/{api}.json")
        if p.exists():
            return read(p)["rows"]
    return []


def select_statement(rows: list[dict], end: str, cutoff: str) -> dict:
    """仅选同期间合并累计报表（1/4/5）；同版不同值保留冲突，不任意挑一行。"""
    eligible = [
        r
        for r in rows
        if r.get("end_date") == end
        and str(r.get("report_type", "1")) in {"1", "4", "5"}
        and (r.get("f_ann_date") or r.get("ann_date"))
        and str(r.get("f_ann_date") or r.get("ann_date")) <= cutoff
    ]
    if not eligible:
        return {}

    def key(r):
        # 同日优先调整后合并表，其次普通合并表，最后才是来源保留的调整前版本。
        return (str(r.get("f_ann_date") or r.get("ann_date")), {"1": 1, "4": 2, "5": 0}[str(r.get("report_type", "1"))])

    best = max(map(key, eligible))
    chosen = [r for r in eligible if key(r) == best]
    result = dict(chosen[0])
    for field in result:
        if len({json.dumps(r.get(field), sort_keys=True) for r in chosen}) > 1:
            result[field] = None
    return result


def company_financials(root: Path, code: str, cutoff: str, quotes: dict | None = None) -> dict:
    """金额单位为元，增幅为百分数；按期间对齐，不把不同季度数字拼成一份财报。"""
    sources = {
        api: financial_rows(root, code, api)
        for api in ("income", "cashflow", "balancesheet", "fina_indicator", "fina_mainbz")
    }
    periods = sorted(
        {r["end_date"] for r in sources["income"] if r.get("end_date") and r["end_date"] <= cutoff}, reverse=True
    )[:8]
    history = []
    for end in periods:
        income = select_statement(sources["income"], end, cutoff)
        if not income:
            continue
        cash = select_statement(sources["cashflow"], end, cutoff)
        balance = select_statement(sources["balancesheet"], end, cutoff)
        indicators = select_statement(sources["fina_indicator"], end, cutoff)
        history.append(
            {
                "endDate": iso_date(end),
                "publishedDate": iso_date(income.get("f_ann_date") or income.get("ann_date")),
                "revenue": income.get("revenue"),
                "netProfit": income.get("n_income_attr_p"),
                "operatingCashflow": cash.get("n_cashflow_act"),
                "totalAssets": balance.get("total_assets"),
                "totalLiabilities": balance.get("total_liab"),
                "revenueGrowthPct": indicators.get("or_yoy"),
                "netProfitGrowthPct": indicators.get("netprofit_yoy"),
            }
        )
    # 主营构成的产品和地区可能重复覆盖总收入，只展示原项目，绝不混合相加。
    business = sources["fina_mainbz"]
    latest = max((r.get("end_date", "") for r in business if r.get("end_date", "") <= cutoff), default="")
    parts, seen = [], set()
    for row in business:
        if row.get("end_date") != latest:
            continue
        key = (row.get("bz_item"), row.get("bz_sales"), row.get("curr_type"))
        if key in seen:
            continue
        seen.add(key)
        parts.append(
            {
                "name": row.get("bz_item"),
                "sales": row.get("bz_sales"),
                "currency": row.get("curr_type"),
                "endDate": iso_date(latest),
            }
        )
    disclosures = []
    for api, label in (
        ("forecast", "业绩预告"),
        ("express", "业绩快报"),
        ("fina_audit", "审计意见"),
        ("dividend", "公司分红"),
        ("disclosure_date", "财报披露安排"),
    ):
        rows = [r for r in financial_rows(root, code, api) if r.get("ann_date") and r["ann_date"] <= cutoff]
        rows = sorted(
            {digest(r): r for r in rows}.values(), key=lambda r: (r["ann_date"], r.get("end_date", "")), reverse=True
        )
        for r in rows[:2]:
            if api == "forecast":
                lower, upper = r.get("net_profit_min"), r.get("net_profit_max")
                summary = str(r.get("type") or "预告类型暂缺")
                if lower is not None and upper is not None:
                    summary += f"；预计净利润 {lower:,.2f} 至 {upper:,.2f} 万元（预告范围，并非正式业绩）"
            elif api == "express":
                summary = "业绩快报，最终数字以正式财报为准"
                if r.get("revenue") is not None:
                    summary += f"；营业收入 {r['revenue'] / 100_000_000:,.2f} 亿元"
            elif api == "fina_audit":
                summary = str(r.get("audit_result") or "审计意见暂缺") + "；" + str(r.get("audit_agency") or "机构暂缺")
            elif api == "dividend":
                summary = "方案进度：" + str(r.get("div_proc") or "暂缺")
                if r.get("pay_date"):
                    summary += "；来源派息日期：" + str(iso_date(r["pay_date"]))
            else:
                summary = "来源计划披露日：" + str(iso_date(r.get("pre_date")) or "暂缺")
                actual = r.get("actual_date")
                summary += "；来源实际披露日：" + str(iso_date(actual) if actual and actual <= cutoff else "暂缺")
            disclosures.append(
                {
                    "category": label,
                    "endDate": iso_date(r.get("end_date")),
                    "publishedDate": iso_date(r["ann_date"]),
                    "summary": summary,
                }
            )
    disclosures.sort(key=lambda r: r["publishedDate"], reverse=True)
    q = (quotes or {}).get(code, {})
    return {
        "history": history,
        "business": parts,
        "disclosures": disclosures,
        "quote": {"date": q.get("_date"), "close": q.get("close"), "changePct": q.get("pct_chg")},
        "sourceName": "Tushare · 公司披露资料",
        "notice": "财务金额为元，利润和现金流为年初至报告期末累计值；主营项目可能按产品或地区分别列示，不能相加。",
    }


def build(root: Path = ROOT, target: Path | None = None, *, checked_through: str | None = None) -> dict:
    """显式离线构建完整快照，成功后原子替换；接口不扫描原 PDF 或发起采集。"""
    plan = read(root / "supplement/plan.json")
    fund = plan["fund_code"]
    cutoff = checked_through or plan["end_date"]
    if not iso_date(cutoff) or cutoff > now().strftime("%Y%m%d"):
        raise ValueError("MATERIAL_CUTOFF_INVALID")
    reports = []
    received = {}
    names = {}
    report_index = read(root / "report-result.json")
    for name in report_index["report_files"]:
        r = read(root / "reports" / (name + ".json"))
        received[r["raw"]["sha256"]] = r["raw"].get("received_at", "")
        holdings = [
            {
                "stockCode": h["stock_code"],
                "stockName": h["stock_name"],
                "weightPct": float(h["nav_weight_pct"]),
                "marketValue": float(h["market_value_cny"]),
            }
            for h in r["holdings"]
        ]
        for h in holdings:
            names[h["stockCode"]] = h["stockName"]
        reports.append(
            {
                "id": r["raw"]["sha256"],
                "title": r["title"]
                + ("（来源目录暂未列出）" if name in report_index.get("not_listed_on_recheck", []) else ""),
                "endDate": r["report_end"],
                "publishedDate": r["published_date"],
                "sourceUrl": safe_url(r["raw"]["url"]),
                "fullDisclosure": r["full_stock_disclosure"],
                "stockWeightPct": float(r["stock_nav_pct"]),
                "disclosedWeightPct": float(r["disclosed_nav_pct"]),
                "holdings": holdings,
                "assets": [
                    {"name": k, "weightPct": float(v["total_asset_pct"]), "value": float(v["value_cny"])}
                    for k, v in r["assets"].items()
                ],
                "industries": [
                    {"name": x["name"], "weightPct": float(x["nav_weight_pct"])} for x in r["reported_industries"]
                ],
            }
        )
    reports.sort(key=lambda r: (r["endDate"], r["publishedDate"], r["fullDisclosure"], received[r["id"]]), reverse=True)
    if not reports or not reports[0]["holdings"]:
        raise ValueError("MATERIAL_REPORTS_INCOMPLETE")
    current = {h["stockCode"]: h for h in reports[0]["holdings"]}
    # 一次载入截至日行情；缺少当日数据保持空值，不用旧价格冒充当日收盘。
    quotes = {}
    for quote_file in sorted((root / "stock-days").glob("*.json"), reverse=True):
        if quote_file.stem > iso_date(cutoff):
            continue
        quotes = read(quote_file).get("rows", {})
        if quotes:
            quotes = {code: {**q, "_date": quote_file.stem} for code, q in quotes.items()}
            break
    companies = [
        {
            "stockCode": code,
            "stockName": current.get(code, {}).get("stockName", name),
            "latestHeld": code in current,
            **company_financials(root, code, cutoff, quotes),
        }
        for code, name in sorted(names.items())
    ]
    documents = []
    for path in source_files(root, "supplement/company-announcements"):
        catalog = read(path)
        for x in catalog["rows"]:
            document = read(source_path(root, "supplement/company-documents/" + digest(x["announcementId"]) + ".json"))
            documents.append(
                {
                    "id": "company-" + x["announcementId"],
                    "kind": "company",
                    "title": x["title_plain"],
                    "stockCode": catalog["code"],
                    "stockName": x["secName"],
                    "publishedDate": x["announced_at_source"][:10],
                    "dateNote": "来源目录本次暂未列出，保留历史原文"
                    if x.get("listing_status") == "NOT_LISTED_ON_RECHECK"
                    else "来源披露日期",
                    "sourceName": "巨潮资讯",
                    "sourceUrl": safe_url(document["receipt"]["url"]),
                    "textComplete": document["text_status"] == "TEXT_EXTRACTED",
                    "latestHeld": catalog["code"] in current,
                }
            )
    seen = set()
    catalog_path = root / "materials-live/supplement/public-catalog.json"
    active_ids = set(read(catalog_path)["items"]) if catalog_path.exists() else None
    for path in source_files(root, "supplement/documents"):
        d = read(path)
        url = safe_url(d.get("url") or d.get("receipt", {}).get("url"))
        if not url or url in seen:
            continue
        seen.add(url)
        c = d["catalog"]
        timestamp = c.get("activationDate")
        published = datetime.fromtimestamp(int(timestamp) / 1000).date().isoformat() if timestamp else None
        documents.append(
            {
                "id": "fund-" + path.stem,
                "kind": "news" if "FUND_PRODUCT_DISCLOSURE" not in d["scopes"] else "fund",
                "title": c["title"],
                "stockCode": None,
                "stockName": None,
                "publishedDate": published,
                "dateNote": "来源目录本次暂未列出，保留历史原文"
                if active_ids is not None and str(c.get("contentId")) not in active_ids
                else "官网目录日期，未证实首次公开时间",
                "sourceName": "德邦基金及公开转载",
                "sourceUrl": url,
                "textComplete": d["text_status"] == "TEXT_EXTRACTED",
                "latestHeld": False,
            }
        )
    documents = list({d["id"]: d for d in documents}.values())
    documents.sort(key=lambda d: (d["publishedDate"] or "", d["id"]), reverse=True)
    # 资料日期来自内容自身。没有新资料时重建不能把旧数据日期改成今天。
    dates = [r["publishedDate"] for r in reports] + [d["publishedDate"] for d in documents]
    dates += [c["quote"]["date"] for c in companies]
    dates += [h["publishedDate"] for c in companies for h in c["history"]]
    actual_date = max(d for d in dates if d and d <= iso_date(cutoff))
    value = {
        "fundCode": fund,
        "asOfDate": actual_date,
        "checkedThrough": iso_date(cutoff),
        "collectedAt": now().isoformat(),
        "reports": reports,
        "companies": companies,
        "documents": documents,
        "notice": "持仓以定期报告为准，不代表实时调仓；公司公告按历史披露持仓关联收集，未覆盖所有新闻。",
    }
    target = target or Path(get_settings().fund_material_directory) / f"{fund}.json"
    # 在发布前验证完整内容可序列化且没有非数值金额；异常由调用方记录，旧文件保持可用。
    json.dumps(value, allow_nan=False)
    save(target, value, replace=True)
    return {"file": str(target), "reports": len(reports), "companies": len(companies), "documents": len(documents)}
