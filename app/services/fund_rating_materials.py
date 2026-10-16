"""将002112既有真实资料接入评级冻结输入；不把部分资料准入为完整八维样本。

此模块只由手动更新和本机试算调用。普通查询只读取发布快照。研究目录仅为只读来源，
原始净值、持仓、报价和引用一并冻结进评级批次，避免仅保存一个无法复核的状态文本。
"""

import hashlib
import io
import json
import re
from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

from pypdf import PdfReader
from sqlalchemy import text

from app.services.fund_exposure_common import ROOT, read
from app.services.fund_rating_disclosures import appointed_teams, compact, equity_balance, key_information
from app.services.fund_rating_metrics import equity_structure, history_metrics


def _raw(root: Path, receipt: dict, frozen: datetime) -> bytes:
    """引用只能指向固定来源根内的有界文件；同时核对原文字节和真实取得时间。"""
    # 已有行情回执复用相邻研究目录原文；只允许同一 .local-runs 根，不接受任意本机路径。
    path = (root / receipt["file"]).resolve() if "file" in receipt else Path(receipt["raw_path"]).resolve()
    allowed_root = root if "file" in receipt else root.parent
    if not path.is_relative_to(allowed_root.resolve()) or not path.is_file() or path.stat().st_size > 20_000_000:
        raise ValueError("RATING_MATERIAL_PATH_SIZE")
    acquired = datetime.fromisoformat(receipt["received_at"])
    if acquired.tzinfo is None or acquired > frozen:
        raise ValueError("RATING_MATERIAL_FUTURE")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
        raise ValueError("RATING_MATERIAL_HASH_MISMATCH")
    return raw


def _pages(root, receipt, frozen):
    reader = PdfReader(io.BytesIO(_raw(root, receipt, frozen)))
    if len(reader.pages) > 150:
        raise ValueError("RATING_DOCUMENT_PAGE_LIMIT")
    return [p.extract_text() or "" for p in reader.pages]


def _ref(receipt, title, published):
    return {
        "title": title,
        "url": receipt["url"],
        "publishedOn": published,
    }


def _metric(name, value, unit, period):
    return {"name": name, "value": str(value), "unit": unit, "period": period}


def _documents(root: Path, as_of: date):
    """只枚举已收录的基金法律文件；不扫描公司新闻全文，也不改历史指针。"""
    index = root / "materials-live/supplement/public-result.json"
    if not index.exists():
        index = root / "supplement/public-result.json"
    if not index.exists():
        return []
    result = []
    for item in read(index)["documents"]:
        if "鑫星价值" not in item["title"] or not any(x in item["title"] for x in ("概要", "招募说明书", "经理")):
            continue
        path = (root / item["file"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("RATING_MATERIAL_PATH_SIZE")
        document = read(path)
        published = document["catalog"].get("activationDate")
        # 评价日前的最近版本；后续更新不能用于解释历史时点的经理与收费。
        if published and datetime.fromtimestamp(int(published) / 1000, ZoneInfo("Asia/Shanghai")).date() <= as_of:
            result.append(document)
    return result


def _no_distribution_events(root, nav_rows, days, frozen):
    """只处理当前三年确无分红/折算的案例；一旦出现事件或来源冲突就要求事件适配。

    完整官方公告目录、官方单位/累计净值和独立复权净值三者交叉核对。
    对未返回事件不单独当作“零”；复权倍数容许误差只来自源字段小数舍入。
    """
    catalog_path = root / "materials-live/supplement/public-catalog.json"
    nav_path = root / "supplement/official-nav.json"
    if not catalog_path.exists() or not nav_path.exists():
        return None
    catalog, official = read(catalog_path), read(nav_path)
    checked = datetime.fromisoformat(catalog["at"])
    if checked > frozen or checked.date() < days[-1] or frozen - checked > timedelta(days=31):
        return None
    counts, receipts = {}, []
    for receipt in catalog["receipts"]:
        value = json.loads(_raw(root, receipt, frozen))
        category = receipt["params"]["categoryId"]
        entry = counts.setdefault(category, {"expected": int(value["totalCount"]), "ids": set(), "count": 0})
        if entry["expected"] != int(value["totalCount"]):
            raise ValueError("RATING_DISCLOSURE_CATALOG_CHANGED")
        entry["ids"].update(v["contentId"] for v in value["contents"])
        entry["count"] += len(value["contents"])
        receipts.append(receipt)
    if any(v["expected"] != len(v["ids"]) or v["count"] != v["expected"] for v in counts.values()):
        raise ValueError("RATING_DISCLOSURE_CATALOG_INCOMPLETE")
    for entry in catalog["items"].values():
        title = entry["catalog"]["title"]
        if "FUND_PRODUCT_DISCLOSURE" not in entry["scopes"]:
            continue
        if any(word in title for word in ("折算", "拆分", "份额合并")):
            return None
        if "鑫星价值" in title and any(word in title for word in ("分红", "收益分配")):
            stamp = int(entry["catalog"].get("activationDate") or 0)
            if not stamp or datetime.fromtimestamp(stamp / 1000, frozen.tzinfo).date() >= days[0]:
                return None
    official_rows = {date.fromisoformat(r["date"]): r for r in official["rows"]}
    if len(official_rows) != len(official["rows"]):
        raise ValueError("RATING_OFFICIAL_NAV_DUPLICATE")
    actual = {r["nav_date"]: r for r in nav_rows}
    used = set()
    gaps = set()
    ratios = []
    for day in days:
        row = official_rows.get(day)
        if row is None or day not in actual:
            return None
        if (
            Decimal(row["unit_nav"]) != actual[day]["unit_nav"]
            or Decimal(row["accum_nav"]) != actual[day]["accumulated_nav"]
        ):
            raise ValueError("RATING_NAV_SOURCE_CONFLICT")
        gaps.add(Decimal(row["accum_nav"]) - Decimal(row["unit_nav"]))
        nav, adjusted = actual[day]["unit_nav"], actual[day]["adjusted_nav"]
        if nav is None or adjusted is None or nav <= 0 or adjusted <= 0:
            return None
        # 既有复权来源混有4—8位精度；按最粗四位的半个末位单位核对，不用百分比容差。
        ratios.append(((adjusted - Decimal("0.00005")) / nav, (adjusted + Decimal("0.00005")) / nav))
        receipt = row["receipt"]
        if receipt["sha256"] not in used:
            _raw(root, receipt, frozen)
            used.add(receipt["sha256"])
            receipts.append(receipt)
    if len(gaps) != 1 or max(a for a, _ in ratios) > min(b for _, b in ratios):
        return None
    return {
        "basis": "完整公告目录与官方累计净值差额、独立复权因子交叉核验；本区间无分红折算",
        "checked_at": catalog["at"],
        "dates_checked": len(days),
        "constant_accumulated_gap": str(next(iter(gaps))),
        "receipts": receipts,
    }


def inspect_case(session, as_of: date, frozen: datetime, *, source_root: Path = ROOT) -> dict | None:
    """提取原始业务事实、独立试算与全部阻塞。返回值绝不是可直接发布的评级证据包。

    对外只显示有明确来源的事实。未核验复权事件的净值试算、五日压力情景和代码原因
    仅进入内部冻结输入；这些结果不能越过完整原件、同类样本和规则激活门槛。
    """
    report_index = source_root / "report-result.json"
    if not report_index.exists():
        return None
    reports = []
    for name in read(report_index)["report_files"]:
        if not re.fullmatch(r"[a-f0-9]{64}-[A-Z0-9_]+", name):
            raise ValueError("RATING_REPORT_REFERENCE")
        p = read(source_root / "reports" / (name + ".json"))
        if p["full_stock_disclosure"] and p["report_end"] <= str(as_of):
            reports.append(p)
    if not reports:
        return None
    report = max(reports, key=lambda p: (p["report_end"], p["published_date"]))
    data = {
        "version": "RATING_MATERIALS_002112_V1",
        "fund_code": "002112",
        "source_root": str(source_root.resolve()),
        "as_of_date": str(as_of),
        "sources": [],
        "facts": {},
        "diagnostics": {},
        "blockers": [],
        "dimensions": {},
    }

    def block(code, dimension, message):
        data["blockers"].append({"code": code, "dimension": dimension, "message": message})

    def dimension(key, explanation, metrics, sources):
        data["dimensions"][key] = {"explanation": explanation, "metrics": metrics, "sources": sources}

    pages = _pages(source_root, report["raw"], frozen)
    if not all(code in compact("\n".join(pages[:6])) for code in ("001412", "002112")):
        raise ValueError("RATING_REPORT_FUND_IDENTITY")
    data["sources"].append(report["raw"])
    report_ref = _ref(report["raw"], report["source"]["title"], report["published_date"])
    positions = [{"issuer": h["stock_code"], "amount": h["market_value_cny"]} for h in report["holdings"]]
    if len({p["issuer"] for p in positions}) != len(positions) or len(positions) != report["holding_count"]:
        raise ValueError("RATING_HOLDING_DUPLICATE")
    if len(report["reported_industries"]) == 1:
        industry = report["reported_industries"][0]
        if sum(Decimal(p["amount"]) for p in positions) != Decimal(industry["value_cny"]):
            raise ValueError("RATING_INDUSTRY_RECONCILIATION")
        for p in positions:
            p["industry"] = industry["industry_code"]
    else:
        block("HOLDING_INDUSTRY_MAPPING", "holdings", "完整持仓已收录，各证券与报告行业分类还需逐一核验。")
    try:
        balance = equity_balance(pages, positions)
        data["facts"]["holdings"] = {**balance, "positions": positions, "report_date": report["report_end"]}
    except ValueError as error:
        # 未支持的报表版式是明确资料缺口；字节/引用损坏不在这里吞掉。
        block(str(error), "liquidity", "资产负债表尚不能完整对账，不能以未知资产补成现金。")
        balance = None
    if (as_of - date.fromisoformat(report["report_end"])).days > 210:
        block("HOLDINGS_EXPIRED", "holdings", "完整持仓报告超过当前候选规则允许的有效期。")
    structure = equity_structure(positions) if all(p.get("industry") for p in positions) else {}
    data["diagnostics"].update(structure)
    dimension(
        "holdings",
        "已接入完整股票持仓并核对报告行业合计；尚未形成同类等级。",
        [
            _metric("完整披露股票数", len(positions), "只", report["report_end"]),
            _metric("股票占基金净资产", report["disclosed_nav_pct"], "%", report["report_end"]),
        ],
        [report_ref],
    )

    from app.services.direction_1d_protocol import calendar

    sessions, calendar_hash = calendar()
    trade_days = [d for d in sessions if d <= as_of]
    if not trade_days or trade_days[-1] != as_of:
        block("CASE_VALUATION_DATE", "return", "当前案例适配需要该月最后一个有效交易日的净值。")
        return data
    months = {}
    for d in trade_days:
        months[(d.year, d.month)] = d
    ends = sorted(months.values())[-37:]
    days = [d for d in trade_days if d >= ends[0]]
    nav_rows = (
        session.execute(
            text("""
        SELECT n.nav_date,n.unit_nav,n.accumulated_nav,n.adjusted_nav,n.content_hash,n.updated_at,
          s.source_code FROM nav_daily n JOIN source_registry s ON s.source_id=n.source_id
        WHERE n.fund_code='002112' AND n.nav_date BETWEEN :start AND :end
        ORDER BY n.nav_date,s.source_code
    """),
            {"start": ends[0], "end": as_of},
        )
        .mappings()
        .all()
    )
    if len({r["nav_date"] for r in nav_rows}) != len(nav_rows):
        raise ValueError("RATING_NAV_SOURCE_CONFLICT")
    by_day = {r["nav_date"]: r for r in nav_rows}
    missing = sorted(set(days) - set(by_day))
    data["facts"]["history"] = {
        "calendar_hash": calendar_hash,
        "month_ends": [str(d) for d in ends],
        "dates": [str(d) for d in days],
        "rows": [dict(r) for r in nav_rows],
        "missing_dates": [str(d) for d in missing],
        "extra_disclosure_dates": [str(d) for d in sorted(set(by_day) - set(days))],
        "events_complete": False,
    }
    event_coverage = (
        _no_distribution_events(source_root, nav_rows, days, frozen) if not missing and len(ends) == 37 else None
    )
    if event_coverage:
        data["facts"]["history"].update(
            events_complete=True,
            event_coverage=event_coverage,
            unit_nav=[str(by_day[d]["unit_nav"]) for d in days],
            cash_dividend_per_share=["0"] * len(days),
            split_factor=["1"] * len(days),
        )
    if not missing and len(ends) == 37:
        # 明示只是单位净值试算，绝不将 rf=[]/require_sharpe=False 解释为无风险利率为零。
        trial = history_metrics(days, [by_day[d]["unit_nav"] for d in days], [], ends, 252, require_sharpe=False)
        trial.pop("sharpe")
        data["diagnostics"]["total_return_metrics" if event_coverage else "unit_nav_only_not_admitted"] = trial
    else:
        block("NAV_CALENDAR_GAPS", "return", "近三年净值或37个月末观察点仍有缺口。")
    if not event_coverage:
        block("RETURN_EVENTS_UNVERIFIED", "return", "净值已接入，分红与份额折算的完整事件覆盖尚待核验。")
    from app.core.config import get_settings
    from app.integrations.fund_rating_public import read_risk_free, risk_free_intervals

    rf_root = Path(get_settings().fund_insight_directory) / "ratings" / "risk-free"
    rf = read_risk_free(rf_root, frozen)
    if rf and not missing and len(ends) == 37:
        data["facts"]["risk_free"] = rf
        data["risk_free_root"] = str(rf_root.resolve())
        for term in ("threeMonth", "oneYear"):
            try:
                interval = risk_free_intervals(rf, days, term=term)
                calculated = history_metrics(days, [by_day[d]["unit_nav"] for d in days], interval, ends, 252)
                data["diagnostics"]["sharpe_proxy_" + term] = calculated["sharpe"]
            except ValueError as error:
                block(str(error), "efficiency", "已取得国债收益率，但所需估值区间仍有缺口。")
        dimension(
            "efficiency",
            "已取得同币种国债收益率，风险收益比较仍待完整同类资料核验。",
            [
                _metric("无风险收益参照", "人民币国债（3个月及1年期）", "", f"{rf['start']} 至 {rf['end']}"),
            ],
            [
                {
                    "title": "财政部-中国国债收益率曲线",
                    "url": "https://yield.chinabond.com.cn/cbweb-czb-web/czb/showHistory?locale=cn_ZH&nameType=1",
                    "publishedOn": None,
                }
            ],
        )
        block(
            "RISK_FREE_BASIS_PENDING_VALIDATION",
            "efficiency",
            "国债收益率可作无风险收益代理，期限和计息口径还需同类试算验证。",
        )
    else:
        block("RISK_FREE_SERIES_MISSING", "efficiency", "缺少同币种、同区间的已核验无风险收益序列。")
    block("PEER_SAMPLE_UNVERIFIED", "stability", "尚无满足八维资料要求、完成产品去重的完整同类样本。")
    dimension(
        "return",
        "净值已按交易日历核对，分红及折算事件未核验完整前不作为总回报评分。",
        [
            _metric("已覆盖估值日", len(set(days) & set(by_day)), "日", f"{ends[0]} 至 {as_of}"),
            _metric("最新单位净值", by_day[as_of]["unit_nav"], "元", str(as_of)),
        ]
        if as_of in by_day
        else [],
        [],
    )
    dimension("risk", "已有净值可供复核；完整总回报口径确认前，不发布回撤及波动评价。", [], [])
    if event_coverage:
        t = data["diagnostics"]["total_return_metrics"]
        nav_ref = {
            "title": "德邦鑫星价值C官方历史净值及公告",
            "url": "https://www.dbfund.com.cn/products/hunhe/002112/index.html",
            "publishedOn": None,
        }
        dimension(
            "return",
            "官方净值与完整公告目录交叉核对，近三年区间未发生分红或份额折算；尚未作同类等级比较。",
            [
                _metric("近一年总回报", f"{t['return_1y'] * 100:.4f}", "%", f"截至 {as_of}"),
                _metric("近三年年化总回报", f"{t['return_3y'] * 100:.4f}", "%", f"{ends[0]} 至 {as_of}"),
            ],
            [nav_ref],
        )
        dimension(
            "risk",
            "使用同一区间总回报计算历史回撤和下行波动，尚未形成同类等级。",
            [
                _metric("最大回撤幅度", f"{t['drawdown'] * 100:.4f}", "%", f"{ends[0]} 至 {as_of}"),
                _metric("下行波动", f"{t['downside'] * 100:.4f}", "%", f"{ends[0]} 至 {as_of}"),
            ],
            [nav_ref],
        )
    dimension(
        "stability",
        "已收录近三年各月末的历史表现，用于考察收益是否持续。",
        [
            _metric("历史观察区间", f"{ends[0]} 至 {as_of}", "", "按月观察"),
        ],
        [],
    )

    quotes, missing_quotes = [], []
    for day in days[-20:]:
        path = source_root / "stock-days" / f"{day}.json"
        daily = read(path) if path.exists() else None
        if daily:
            _raw(source_root, daily["receipt"], frozen)
            data["sources"].append(daily["receipt"])
        for p in positions:
            q = daily["rows"].get(p["issuer"]) if daily else None
            if q is None or q.get("amount") is None:
                missing_quotes.append([str(day), p["issuer"]])
            else:
                # Tushare daily amount 为千元；财报为元，必须显式乘1000。
                amount = Decimal(str(q["amount"])) * 1000
                if amount < 0:
                    raise ValueError("RATING_NEGATIVE_TURNOVER")
                quotes.append({"date": str(day), "issuer": p["issuer"], "amount_yuan": str(amount)})
    data["facts"]["turnover"] = {"rows": quotes, "missing": missing_quotes, "source_unit": "THOUSAND_YUAN"}
    if missing_quotes:
        block("TURNOVER_MISSING", "liquidity", "部分股票缺少过去20个交易日的成交额。")
    elif balance:
        sale = sum(
            min(
                Decimal(p["amount"]),
                Decimal("0.5") * median(Decimal(q["amount_yuan"]) for q in quotes if q["issuer"] == p["issuer"]),
            )
            for p in positions
        )
        b = balance["balance_sheet"]
        data["diagnostics"]["liquidity_scenario_not_admitted"] = {
            "assumptions": "仅货币资金视为现金；所有负债立即支付；股票假设均可售；不是实际五日可变现证明",
            "stock_sale_limit_yuan": str(sale),
            "ratio": str(
                max(
                    Decimal(0),
                    min(
                        Decimal(1),
                        (sale + Decimal(b["other_assets"]["货币资金"]) - Decimal(b["liabilities"]))
                        / Decimal(balance["net_assets"]),
                    ),
                )
            ),
        }
    block("HOLDER_EXACT_RATIO_MISSING", "liquidity", "报告仅披露没有单一持有人达到20%，未披露最大持有人精确比例。")
    block("LIQUIDITY_SETTLEMENT_UNVERIFIED", "liquidity", "缺少评价日可售限制及五日内应收应付款的逐项核验依据。")
    dimension(
        "liquidity",
        "已核对报告资产负债及持仓成交情况。",
        [
            *(
                [
                    _metric(
                        "基金净资产",
                        f"{Decimal(balance['net_assets']) / 100_000_000:.2f}",
                        "亿元",
                        report["report_end"],
                    )
                ]
                if balance
                else []
            ),
            *(
                [_metric("报告期单一持有人比例", "低于20%（未披露精确值）", "", report["report_end"])]
                if balance and balance.get("largest_holder_upper_exclusive") == "0.20"
                else []
            ),
        ],
        [report_ref],
    )

    documents = _documents(source_root, as_of)
    team_history = None
    summaries = [d for d in documents if "C份额" in compact(d["catalog"]["title"]) and "概要" in d["catalog"]["title"]]
    if summaries:
        document = max(summaries, key=lambda d: int(d["catalog"].get("activationDate") or 0))
        info = key_information(_pages(source_root, document["receipt"], frozen))
        info["checked_on"] = (
            datetime.fromisoformat(document["receipt"]["received_at"])
            .astimezone(ZoneInfo("Asia/Shanghai"))
            .date()
            .isoformat()
        )
        if date.fromisoformat(info["published_on"]) > as_of:
            raise ValueError("RATING_KEY_INFORMATION_AFTER_VALUATION")
        data["facts"]["key_information"] = info
        data["sources"].append(document["receipt"])
        ref = _ref(document["receipt"], document["catalog"]["title"], info["published_on"])
        dimension(
            "cost",
            "采用官方披露的运作综合费率，已包含其他运作费用；申购与持有365天赎回费均为零。",
            [
                _metric(
                    "运作综合费率（年化）",
                    f"{Decimal(info['fees']['operating_cost']) * 100:.2f}",
                    "%",
                    info["published_on"],
                ),
                _metric(
                    "一万元持有365天费用测算",
                    f"{Decimal(info['cost_1y']) * 10000:.2f}",
                    "元",
                    "无涨跌、按披露费率的标准情景",
                ),
            ],
            [ref],
        )
        dimension(
            "management",
            "官方资料概要已确认当前经理与产品关系，完整团队变更和策略可比范围仍需核验。",
            [
                _metric("基金经理", info["current_manager"], "", info["published_on"]),
                _metric("开始管理日期", info["manager_start"], "", "官方资料概要"),
            ],
            [ref],
        )
        prospectuses = [d for d in documents if "招募说明书" in d["catalog"]["title"]]
        if prospectuses:
            prospectus = max(prospectuses, key=lambda d: int(d["catalog"].get("activationDate") or 0))
            try:
                team_history = appointed_teams(_pages(source_root, prospectus["receipt"], frozen), info, days[0], as_of)
                data["facts"]["manager"] = team_history
                data["sources"].append(prospectus["receipt"])
                dimension(
                    "management",
                    "已按官方招募说明书重建正式任职团队，任免统计口径仍须通过同类方法验证。",
                    [
                        _metric("当前基金经理", info["current_manager"], "", info["published_on"]),
                        _metric("近三年正式团队变更", team_history["team_changes"], "次", f"{ends[0]} 至 {as_of}"),
                        _metric("本届团队连续管理", f"{Decimal(team_history['team_months']):.2f}", "月", str(as_of)),
                    ],
                    [ref, _ref(prospectus["receipt"], prospectus["catalog"]["title"], info["published_on"])],
                )
            except ValueError as error:
                block(str(error), "management", "官方团队记录尚不能形成完整连续的任职区间。")
        if frozen - datetime.fromisoformat(document["receipt"]["received_at"]) > timedelta(days=31):
            block("FEES_VERIFICATION_EXPIRED", "cost", "费用文件超过31天未复核，需检查最新披露。")
    else:
        block("FEE_DOCUMENT_MISSING", "cost", "尚未取得对应份额的最新产品资料概要。")
    if not team_history:
        block("MANAGER_TIMELINE_UNVERIFIED", "management", "完整团队任职变更记录尚未完成逐项核对。")
    block(
        "PEER_CLASSIFICATION_UNVERIFIED", "management", "已确认A/C产品关系；可比策略、仓位范围及全体同类分类尚未核验。"
    )
    allowed = session.execute(
        text("SELECT enabled,authorization_verified_at FROM source_registry WHERE source_code='DBFUND_OFFICIAL_PUBLIC'")
    ).first()
    if not allowed or not allowed.enabled or not allowed.authorization_verified_at:
        block("SOURCE_USE_NOT_REGISTERED", None, "官方原文已取得；评级来源使用范围尚未完成登记核验。")
    return data


def attach_materials(member: dict, material: dict) -> None:
    """保留八维和全部缺口；事实展示不会设置 admitted、分数或等级。"""
    member["preparation"] = material
    member["reasons"] = [b["code"] for b in material["blockers"]]
    public = deepcopy(member["public"])
    public["summary"] = "已接入部分真实资料，仍有必要资料及同类验证未完成，暂未形成综合评级。"
    for dimension in public["dimensions"]:
        if dimension["key"] in material["dimensions"]:
            dimension.update(material["dimensions"][dimension["key"]])
        messages = [b["message"] for b in material["blockers"] if b["dimension"] == dimension["key"]]
        if messages:
            dimension["explanation"] += " " + " ".join(messages)
    facts = material["facts"]
    public["dataDates"]["holdings"] = facts.get("holdings", {}).get("report_date")
    public["dataDates"]["fees"] = facts.get("key_information", {}).get("checked_on")
    public["dataDates"]["manager"] = facts.get("key_information", {}).get("checked_on")
    history = facts.get("history", {})
    if history.get("dates") and not history.get("missing_dates"):
        public["dataDates"]["nav"] = history["dates"][-1]
        public["comparison"]["period"] = f"{history['dates'][0]} 至 {history['dates'][-1]}"
    if facts.get("key_information"):
        public["comparison"]["currency"] = "CNY"
    public["comparison"]["scope"] = "本站已收录基金；A/C产品关系已找到官方依据，同类分类与准入仍待核验"
    public["limitations"] = [
        "净值、报告和费用日期不同，报告持仓不代表评价日实时持仓。",
        "八维资料必须完整；净值试算、流动性假设和资料数量不构成评级。",
        "同类完整样本及规则验证尚未通过，暂未评级不等于偏弱。",
    ]
    member["public"] = public


def archive_materials(material: dict, root: Path, frozen: datetime) -> list[dict]:
    """发布前把实际引用原文复制到评级专用目录；研究目录保持只读，不依赖其长期存在。

    按字节哈希不可变保存，允许相同原文复用。文件写入失败使整类发布回滚，既有指针保留。
    """
    from app.services.fund_exposure_common import _publish

    facts = material.get("facts", {})
    refs = material.get("sources", []) + facts.get("history", {}).get("event_coverage", {}).get("receipts", [])
    pairs = [(Path(material["source_root"]), r) for r in refs]
    if facts.get("risk_free"):
        pairs.extend((Path(material["risk_free_root"]), r) for r in facts["risk_free"]["receipts"])
    archived = {}
    for origin, receipt in pairs:
        key = receipt["sha256"]
        if key in archived:
            continue
        raw = _raw(origin, receipt, frozen)
        suffix = Path(receipt.get("file", receipt.get("raw_path", ""))).suffix
        # 管理人也可能只发布 DOCX 正文；这里只保存已核验原字节，不执行文件内容。
        if suffix not in {".pdf", ".json", ".html", ".docx"}:
            raise ValueError("RATING_RAW_SOURCE_TYPE")
        relative = f"sources/{key}{suffix}"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _publish(path, raw)
        except FileExistsError:
            if hashlib.sha256(path.read_bytes()).hexdigest() != key:
                raise ValueError("RATING_ARCHIVED_SOURCE_CORRUPT") from None
        archived[key] = {"sha256": key, "file": relative}
    return sorted(archived.values(), key=lambda item: item["sha256"])
