"""从现有留档冻结002112资料；不采集，不读取目标日答案，不导入单次报告结果。"""

from datetime import date, datetime, timedelta
from pathlib import Path

from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import get_engine
from app.repositories import direction_1d as repo
from app.schemas.fund_information_analysis import normalize
from app.services import direction_1d_information as info
from app.services import fund_information_manager as managers
from app.services.direction_1d_protocol import ZONE, digest, features, input_days, window
from app.services.fund_exposure_common import ROOT, read
from app.services.fund_information_contracts import time_context
from app.services.fund_materials_store import source_path

ACTIVE_FILE = ROOT.parent / "direction-1d-information/analysis-active.json"


def enabled(code: str) -> bool:
    return code == "002112" and ACTIVE_FILE.exists() and read(ACTIVE_FILE).get("enabled") is True


def prepare(now) -> dict:
    """快照时间取数据库；各类日期独立保留。缺最新净值不改变预测比较基准。"""
    now = now.astimezone(ZONE)
    w = window(now)
    base = date.fromisoformat(w["base_nav_date"])
    material_path = Path(get_settings().fund_material_directory) / "002112.json"
    material = read(material_path)
    candidates = [r for r in material["reports"] if r["publishedDate"] <= str(now.date())]
    if not candidates:
        raise ValueError("INFORMATION_HOLDINGS_NOT_READY")
    report = max(candidates, key=lambda r: (r["endDate"], r["publishedDate"]))
    with get_engine().begin() as c:
        source = repo.source(c)
        profile = repo.profiles(c, ["002112"])[0]
        # 管理团队及任期表现是持续背景，独立于下方近期公司新闻的时间与数量窗口。
        manager_context = managers.prepare(c, source["source_id"], material, now, base, profile.get("benchmark") or "")
        latest = c.execute(
            text(
                "SELECT max(nav_date) FROM nav_daily WHERE fund_code='002112' "
                "AND source_id=:source AND nav_date<=:base AND updated_at<=:now"
            ),
            {"source": source["source_id"], "base": base, "now": now},
        ).scalar_one()
        rows = repo.navs(c, "002112", source["source_id"], input_days(latest)[0], latest) if latest else []
        rows = [r for r in rows if r["updated_at"] <= now]
        observed = repo.observe(c, "002112", source, rows, now) if rows else []
        sources, versions = info.current_sources(str(base), now, c)
        dividends = [
            dict(r)
            for r in c.execute(
                text(
                    "SELECT ann_date,record_date,ex_date,nav_ex_date,cash_dividend,base_unit,content_hash "
                    "FROM fund_dividend WHERE fund_code='002112' AND source_id=:source AND ann_date<=:today "
                    "AND updated_at<=:now ORDER BY ann_date DESC LIMIT 10"
                ),
                {"source": source["source_id"], "today": now.date(), "now": now},
            ).mappings()
        ]
        shares = [
            dict(r)
            for r in c.execute(
                text(
                    "SELECT trade_date,fund_share,content_hash FROM fund_share_snapshot "
                    "WHERE fund_code='002112' AND source_id=:source AND trade_date<=:base "
                    "AND updated_at<=:now ORDER BY trade_date DESC LIMIT 2"
                ),
                {"source": source["source_id"], "base": base, "now": now},
            ).mappings()
        ]
    chosen = info.recipe.legacy.choose_report(sources["reports"], now.isoformat())
    if not chosen or chosen["report_end"] != report["endDate"]:
        raise ValueError("ANALYSIS_HOLDINGS_MISMATCH")
    weights = info.recipe.legacy.holding_weights(chosen)
    holdings = {h["stockCode"]: h for h in report["holdings"]}
    if any(abs(weights.get(code, -1) * 100 - h["weightPct"]) > 0.011 for code, h in holdings.items()):
        raise ValueError("ANALYSIS_HOLDINGS_MISMATCH")
    quote_day = max((d for d, items in sources["stocks"].items() if items and d <= str(base)), default=None)
    prices = sources["stocks"].get(quote_day, {})
    companies = []
    for company in material["companies"]:
        code = company["stockCode"]
        if code not in holdings:
            continue
        price = prices.get(code)
        companies.append(
            {
                "code": code,
                "name": company["stockName"],
                "weight_pct": holdings[code]["weightPct"],
                "quote": {"date": quote_day, "change_pct": price["pct_chg"]} if price else None,
                "financials": [
                    r for r in company.get("history", []) if r.get("publishedDate", "9999") <= str(now.date())
                ][:2],
                # 没有单独公开时刻的主营资料只作本次已取得背景，不倒灌历史回放。
                "business": company.get("business", [])[:8],
                "source_name": company.get("sourceName", "已保存公司资料"),
            }
        )
    documents, seen, excluded = [], set(), []
    start = str(now.date() - timedelta(days=14))
    for doc in sources["documents"]:
        if doc.get("published_date", "") < start or doc.get("source_hash") in seen:
            continue
        if datetime.fromisoformat(doc["available_at"]) > now or doc["published_date"] > str(now.date()):
            continue
        codes = sorted({link["code"] for link in doc.get("links", []) if link.get("code") in holdings})
        if not codes:
            excluded.append({"id": doc["id"], "reason": "NO_VERIFIED_HOLDING_LINK"})
            continue
        body = "\n".join(doc.get("facts", []))
        scope = "VERIFIED_EXCERPT"
        if doc.get("document_id", "").startswith("company-"):
            identity = doc["document_id"].removeprefix("company-")
            original = read(source_path(ROOT, "supplement/company-documents/" + digest(identity) + ".json"))
            if original["receipt"]["sha256"] == doc["source_hash"]:
                body = "\n".join(original.get("pages", []))
                scope = "FULL_TEXT"
        if not body.strip():
            excluded.append({"id": doc["id"], "reason": "BODY_MISSING"})
            continue
        truncated = len(body) > 18000
        if truncated:
            body = body[:14000] + "\n【中间部分未纳入】\n" + body[-4000:]
        documents.append(
            {
                "id": doc["id"],
                "title": doc["title"],
                "kind": doc["event_type"],
                "date": doc["published_date"],
                "available_at": doc["available_at"],
                "source_hash": doc["source_hash"],
                "url": doc.get("source_url"),
                "codes": codes,
                "body": body,
                "body_scope": scope,
                "body_truncated": truncated,
            }
        )
        seen.add(doc["source_hash"])
    # 基金自身公告只接受原文明确提及本基金的材料，官网目录日期不能充当首次公开时刻。
    for d in material["documents"]:
        if d["kind"] != "fund" or not start <= (d.get("publishedDate") or "") <= str(now.date()):
            continue
        if "基金经理变更" in d["title"]:
            # 已在持续经理背景中核对和引用，不能作为近期新闻再次重复增加影响。
            continue
        original = read(source_path(ROOT, "supplement/documents/" + d["id"].removeprefix("fund-") + ".json"))
        receipt = original.get("receipt", {})
        if not original.get("fund_name_or_code_mentioned") or not receipt.get("received_at"):
            continue
        if datetime.fromisoformat(receipt["received_at"]) > now:
            continue
        body = "\n".join(original.get("pages", []))
        if not body.strip():
            excluded.append({"id": d["id"], "reason": "BODY_MISSING"})
            continue
        documents.append(
            {
                "id": d["id"],
                "title": d["title"],
                "kind": "ANNOUNCEMENT",
                "date": d["publishedDate"],
                "available_at": receipt["received_at"],
                "source_hash": receipt["sha256"],
                "url": d["sourceUrl"],
                "codes": [],
                "relation": "原文明示本基金；目录日期未证实为首次公开时间",
                "body": body[:18000],
                "body_scope": "FULL_TEXT",
                "body_truncated": len(body) > 18000,
            }
        )
    documents = unique_documents(documents)
    # 最近的具体事项优先；上限外资料记录缺口，不假装已阅读。
    documents.sort(key=lambda d: (d["date"], d["id"]), reverse=True)
    overflow = max(0, len(documents) - 100)
    documents = documents[:100]
    market = {}
    for code, items in sources["markets"].items():
        days = [d for d in items if d <= str(base)]
        if days:
            day = max(days)
            market[code] = {"date": day, "change_pct": items[day]["pct_chg"]}
    data = {
        "fund_code": "002112",
        "fund_name": profile["fund_name"],
        "product_family_id": str(profile.get("product_family_id") or "002112"),
        "as_of": now.isoformat(),
        "window": w,
        "source_id": str(source["source_id"]),
        "nav": observed,
        "latest_nav_date": str(latest) if latest else None,
        "report": report,
        "companies": companies,
        "market": market,
        "documents": documents,
        "source_manifest": versions,
        "excluded": excluded,
        "overflow": overflow,
        "material_hash": digest(material),
        "retention_days": source["retention_days"],
        "dividends": [{k: str(v) if v is not None else None for k, v in r.items()} for r in dividends],
        "shares": [{k: str(v) if v is not None else None for k, v in r.items()} for r in shares],
        "manager_context": manager_context,
    }
    data["facts"] = facts(data)
    data["inventory"] = inventory(data)
    data["time_context"] = time_context(data)
    return data


def unique_documents(documents: list[dict]) -> list[dict]:
    """同发行人、同日期、正文相同的转载只进入一次；保留稳定来源，不以文件数量加权。"""
    unique = {}
    for d in sorted(documents, key=lambda d: d["id"]):
        key = digest({"codes": d["codes"], "date": d["date"], "body": normalize(d["body"])})
        unique.setdefault(key, d)
    return list(unique.values())


def facts(data: dict) -> dict:
    """只由原数字生成展示事实；每条有固定身份，可被综合判断引用但不能改数。"""
    values = data["nav"]
    result = {}
    if values:
        last = values[-1]
        summary = f"截至{last['nav_date']}，单位净值{last['unit_nav']}。"
        if len(values) == 61:
            try:
                v = features([r["unit_nav"] for r in values])
                summary += (
                    f"近5个交易日涨跌{v[0] * 100:+.2f}%，近20个交易日{v[1] * 100:+.2f}%；"
                    f"近60日区间位置{v[5] * 100:.1f}%。"
                )
            except ValueError:
                pass
        result["nav"] = {"category": "净值", "text": summary, "source": None}
    total = 0.0
    coverage = 0.0
    for company in data["companies"]:
        quote = company["quote"]
        if quote and quote["change_pct"] is not None:
            impact = company["weight_pct"] * quote["change_pct"] / 100
            total += impact
            coverage += company["weight_pct"]
            result["stock:" + company["code"]] = {
                "category": "持仓行情",
                "text": f"{quote['date']}，{company['name']}涨跌{quote['change_pct']:+.2f}%，"
                f"披露仓位{company['weight_pct']:.2f}%，"
                f"按该仓位估计对当日基金影响{impact:+.2f}个百分点。",
                "source": None,
            }
        if company["financials"]:
            f = company["financials"][0]
            growth = f.get("netProfitGrowthPct")
            if growth is not None:
                result["financial:" + company["code"]] = {
                    "category": "公司经营",
                    "text": f"{company['name']}在{f['publishedDate']}披露，"
                    f"截至{f['endDate']}报告期净利润同比{growth:+.2f}%。"
                    "这是对应报告期的经营数据，不是目标日新发生的变化。",
                    "source": None,
                }
    if coverage:
        leading = sorted(
            (c for c in data["companies"] if c["quote"] and c["quote"]["change_pct"] is not None),
            key=lambda c: abs(c["weight_pct"] * c["quote"]["change_pct"]),
            reverse=True,
        )[:4]
        result["holdings"] = {
            "category": "持仓行情",
            "text": f"按{data['report']['endDate']}披露持仓估算，"
            f"可计算股票合计占净资产{coverage:.2f}%，最近所列行情日合计影响约{total:+.2f}个百分点。"
            "此为已发生行情的旧仓位估算，不是目标日预测涨幅。\n"
            + "\n".join(result["stock:" + c["code"]]["text"] for c in leading),
            "source": None,
        }
    report = data["report"]
    if report.get("industries"):
        result["industry"] = {
            "category": "持仓行情",
            "source": None,
            "text": f"{report['endDate']}披露行业分布："
            + "；".join(f"{r['name']}占净资产{r['weightPct']:.2f}%" for r in report["industries"][:10])
            + "。这是披露时点的配置，不是行业今日涨跌。",
        }
    for row in data.get("dividends", [])[:1]:
        result["fund_dividend"] = {
            "category": "公告",
            "source": None,
            "text": f"所存最近分红公告日期{row['ann_date']}，除息日{row['ex_date'] or row['nav_ex_date'] or '未提供'}，"
            f"现金分红原值{row['cash_dividend']}，每份基数原值{row['base_unit']}。"
            "历史分红不直接作为当日方向信号，除息时单位净值与总回报须区别。",
        }
    for code, value in data["market"].items():
        name = {"000300.SH": "沪深300", "000905.SH": "中证500"}[code]
        result["market:" + code] = {
            "category": "持仓行情",
            "text": f"{value['date']}，{name}涨跌{value['change_pct']:+.2f}%。",
            "source": None,
        }
    if data.get("manager_context"):
        result.update(managers.facts(data["manager_context"]))
    return result


def inventory(data: dict) -> list[dict]:
    """清单只说明已取得范围；当前来源没覆盖的事项保留明确缺口。"""
    definitions = [
        ("基金基本资料", "AVAILABLE", "已核对基金身份与名称"),
        ("历史净值", "AVAILABLE" if data["nav"] else "MISSING", "截至" + str(data["latest_nav_date"])),
        (
            "分红与份额调整",
            "BACKGROUND" if data.get("dividends") else "MISSING",
            "已检查所存分红记录；单位净值除息不等于经济亏损"
            if data.get("dividends")
            else "已检查所存记录，未取得分红事件；不代表没有事件",
        ),
        ("披露持仓", "AVAILABLE", "报告期" + data["report"]["endDate"] + "，非实时仓位"),
        ("资产与行业分布", "BACKGROUND", "保留报告原口径，不与股票净资产权重混算"),
        ("持仓股票行情", "PARTIAL", "使用已保存收盘行情，非实时盘中行情"),
        ("大盘与比较基准", "AVAILABLE" if data["market"] else "MISSING", "已取得指数分别标明日期"),
        ("行业与主题行情", "MISSING", "本次未取得独立行业行情"),
        ("公司主营业务", "BACKGROUND", "使用已保存公司资料，关联需原文验证"),
        ("公司财务", "BACKGROUND", "按各自报告期和公开日使用，不冒充今日新增消息"),
        ("公司公告", "PARTIAL", "近两周已取得的关联正文；部分只取得摘要"),
        ("基金自身公告", "PARTIAL", "检查近两周已保存原文，只有明确涉及本基金的内容参与"),
        ("政策原文", "MISSING", "本次没有通过持仓关联核验的政策原文"),
        ("相关新闻", "MISSING", "本次没有通过持仓关联核验的新闻"),
        (
            "规模与份额",
            "BACKGROUND" if data.get("shares") else "MISSING",
            "已取得份额截至" + data["shares"][0]["trade_date"] + "；份额不直接等同净申购或涨跌"
            if data.get("shares")
            else "本次未取得可用份额记录",
        ),
        ("交易日历与来源", "AVAILABLE", "北京时间，按实际取得时刻冻结资料"),
    ]
    rows = [
        {"id": f"D{i:02}", "name": name, "status": status, "detail": detail}
        for i, (name, status, detail) in enumerate(definitions, 1)
    ]
    manager = data.get("manager_context")
    if manager:
        names = "、".join(m["name"] for m in manager["managers"])
        rows[0].update(
            status="PARTIAL",
            detail=(f"已核对当前经理{names}的任职及可用区间表现；同类和基准比较尚缺。"
                    if manager["current_verified"] else "当前经理及任职表现尚未完整核实。")
            + "".join(manager["gaps"])[:180],
        )
        change = manager.get("latest_change")
        rows[11]["detail"] += (
            f"另保留{change['published_date']}经理变更公告及公开原因，不受两周限制。"
            if change and change["parsed"] else "经理变更及公开原因尚未完整核实。"
        )
    for kind, index in [("POLICY", 12), ("NEWS", 13)]:
        if any(d["kind"] == kind for d in data["documents"]):
            rows[index].update(status="PARTIAL", detail="已取得关联原文，按本次实际解析结果参与")
    if data.get("as_of") and data.get("window"):
        timing = time_context(data)
        rows[5].update(
            status="AVAILABLE"
            if timing["stocks"] and all(q["status"] == "READY" for q in timing["stocks"].values())
            else "PARTIAL",
            detail="；".join(timing["statements"]),
        )
        rows[6].update(
            status="AVAILABLE" if all(q["status"] == "READY" for q in timing["markets"].values()) else "PARTIAL"
        )
    return rows
