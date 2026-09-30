"""E16：只读核对 22/23 日资料与原始报告时间，不重算预测、不读取封存答案。"""

import hashlib
import json
from collections import Counter
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

from app.db.session import get_engine
from app.services.direction_1d_protocol import ZONE, digest
from app.services.fund_exposure_common import ROOT, read, save
from app.services.fund_materials import snapshot
from sqlalchemy import URL, create_engine, text


def before_cutoff(received_at, cutoff):
    """必须有时区且严格早于截止；只有公开日期不能冒充系统已收到。"""
    if not received_at:
        return False
    value = datetime.fromisoformat(str(received_at))
    return value.tzinfo is not None and value < cutoff


def raw_verified(receipt):
    """相对原研究根目录验原件；不修改旧收据，不发起来源请求。"""
    path = (ROOT / receipt["file"]).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]


def core_engine():
    cfg = {}
    for line in Path("C:/ideaProject/workSpace12/.env").read_text(encoding="utf-8-sig").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            cfg[key.strip()] = value.strip()
    host = cfg.get("FUND_CORE_DB_HOST", "localhost")
    if host not in {"localhost", "127.0.0.1"}:
        raise ValueError("CASE_AUDIT_LOCAL_DATABASE_REQUIRED")
    return create_engine(
        URL.create(
            "postgresql+psycopg",
            username=cfg["FUND_CORE_DB_USERNAME"],
            password=cfg["FUND_CORE_DB_PASSWORD"],
            host=host,
            port=int(cfg.get("FUND_CORE_DB_PORT", "54329")),
            database="fund_core",
        ),
        hide_parameters=True,
    )


def audit():
    material = snapshot("002112")
    if not material:
        raise ValueError("CASE_SOURCE_SNAPSHOT_MISSING")
    cases = []
    with get_engine().connect() as ai, core_engine().connect() as core:
        for connection in (ai, core):
            connection.execute(text("SET TRANSACTION READ ONLY"))
            connection.execute(text("SET LOCAL statement_timeout='15s'"))
        for day in (date(2026, 9, 22), date(2026, 9, 23)):
            cutoff = datetime.combine(day, time(15), ZONE)
            # 仅查询案例日前已发生的公开净值；没有读取目标日之后的回算答案或研究标签。
            nav = [
                dict(r)
                for r in ai.execute(
                    text("""
                SELECT nav_date,unit_nav,source_published_at,created_at,updated_at,content_hash
                FROM nav_daily WHERE fund_code='002112' AND nav_date>=:start AND nav_date<:day
                ORDER BY nav_date
            """),
                    {"day": day, "start": day - timedelta(days=45)},
                ).mappings()
            ]
            known = [r for r in nav if r["created_at"] < cutoff and r["updated_at"] < cutoff]
            # 当前行曾在截止后被改过，就不能用当前值证明当时已得，宁缺勿补。
            facts = {
                "nav_as_of": str(known[-1]["nav_date"]) if known else None,
                "known_nav_rows": len(known),
                "daily_change_pct": None,
                "twenty_observation_drawdown_pct": None,
            }
            if len(known) >= 2 and all(r["unit_nav"] > 0 for r in known):
                facts["daily_change_pct"] = str((known[-1]["unit_nav"] / known[-2]["unit_nav"] - 1) * 100)
                peak, drawdown = Decimal(0), Decimal(0)
                for row in known[-20:]:
                    peak = max(peak, row["unit_nav"])
                    drawdown = min(drawdown, row["unit_nav"] / peak - 1)
                facts["twenty_observation_drawdown_pct"] = str(drawdown * 100) if len(known) >= 20 else None
            report = max(
                (r for r in material["reports"] if r["publishedDate"] < str(day)),
                key=lambda r: (r["endDate"], r["publishedDate"]),
            )
            receipts = [read(p) for p in (ROOT / "report-receipts").glob("*.json")]
            receipts = [r for r in receipts if r.get("url") == report["sourceUrl"] and raw_verified(r)]
            receipt = min(receipts, key=lambda r: r["received_at"]) if receipts else None
            holdings = {
                "report_date": report["endDate"],
                "published_date": report["publishedDate"],
                "source_url": report["sourceUrl"],
                "report_hash": digest(report),
                "top_ten_weight_pct": sum(
                    Decimal(str(h["weightPct"]))
                    for h in sorted(report["holdings"], key=lambda h: h["weightPct"], reverse=True)[:10]
                ),
                "first_retained_report_receipt": receipt,
                "already_received_proven": bool(receipt and before_cutoff(receipt["received_at"], cutoff)),
                "receipt_scope": "原 report-receipts 内相同来源且原件哈希通过的最早记录；不推定更早已接入",
            }
            quote_day = day - timedelta(days=1)
            quote_path = ROOT / "stock-days" / f"{quote_day}.json"
            cross = None
            if quote_path.exists():
                quotes = read(quote_path)
                if not raw_verified(quotes["receipt"]):
                    raise ValueError("CASE_QUOTE_ORIGINAL_HASH_MISMATCH")
                covered = [(h, quotes["rows"].get(h["stockCode"])) for h in report["holdings"]]
                covered = [(h, q) for h, q in covered if q and q.get("pct_chg") is not None]
                cross = {
                    "quote_day": str(quote_day),
                    "receipt": quotes["receipt"],
                    "source_hash": digest(quotes),
                    "already_received_proven": before_cutoff(quotes["receipt"]["received_at"], cutoff),
                    "coverage_weight_pct": sum(Decimal(str(h["weightPct"])) for h, _ in covered),
                    "static_change_pct_points": sum(
                        Decimal(str(h["weightPct"])) * Decimal(str(q["pct_chg"])) / 100 for h, q in covered
                    ),
                    "meaning": "今天用已披露历史持仓和当日收盘行情重建的静态关系，不是历史实收输入或净值归因",
                }
            forecasts = [
                dict(r)
                for r in core.execute(
                    text("""
                SELECT forecast_id,protocol,generated_at,stored_at,content_hash FROM direction_1d_forecast
                WHERE fund_code='002112' AND target_nav_date=:day ORDER BY generated_at
            """),
                    {"day": day},
                ).mappings()
            ]
            reports = [
                dict(r)
                for r in core.execute(
                    text("""
                SELECT report_id,generated_at,content_hash,payload->>'generationStatus' status,
                       payload->>'decision' decision,payload->>'validUntil' valid_until,
                       payload->>'strategyVersion' strategy_version
                FROM portfolio_decision_report WHERE fund_code='002112'
                AND generated_at>=:start AND generated_at<:end ORDER BY generated_at
            """),
                    {
                        "start": datetime.combine(day, time(), ZONE),
                        "end": datetime.combine(day + timedelta(days=1), time(), ZONE),
                    },
                ).mappings()
            ]
            ontime = [r for r in reports if r["generated_at"] < cutoff]
            cases.append(
                {
                    "target_date": day,
                    "cutoff": cutoff,
                    "public_nav_rows": nav,
                    "known_nav_facts": facts,
                    "holdings": holdings,
                    "stock_reconstruction": cross,
                    "independent_industry_quote": None,
                    "public_document_inventory_count": sum(
                        d["publishedDate"] < str(day) for d in material["documents"]
                    ),
                    "document_inventory_meaning": (
                        "今天留存目录中公开日早于案例日的记录数；不代表原文实收、语义审核或预测采用"
                    ),
                    "one_day_forecasts": forecasts,
                    "issued_report_scope": (
                        "本地库该基金全部原报告的时间元数据；不包含用户身份、金额，不据此推定当前登录用户的结果"
                    ),
                    "issued_reports": reports,
                    "issued_before_cutoff_count": len(ontime),
                    "issued_before_cutoff_decisions": dict(Counter(r["decision"] for r in ontime)),
                    "first_report_generated_before_cutoff": ontime[0]["generated_at"] if ontime else None,
                    "trade_limit": (
                        "15 点后生成不能视为当日可执行；最早成交还需基金开放状态、渠道截止及实际委托，未擅定份额和费用"
                    ),
                }
            )
    return {
        "audited_at": datetime.now(ZONE),
        "read_only": True,
        "new_fits": 0,
        "cases": cases,
        "material_snapshot_hash": digest(material),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "conclusions": [
            "当时公开、系统已得、今天重建分别记录；今日重建不能改变原预测或证明原提醒已发生。",
            "集中度可支持风险事实，不能单凭该案例证明应卖出、何时买回或策略有效。",
            "单位净值变动仅作原始事实，不冒充已经核验的分红再投资总回报。",
            "22/23 日属于已知案例诊断；没有新增历史回算预测，未读取 2025 封存答案。",
        ],
    }


if __name__ == "__main__":
    result = json.loads(json.dumps(audit(), ensure_ascii=False, default=str))
    target = Path("data/fund-insights/cases") / (datetime.now(ZONE).strftime("%Y%m%d-%H%M%S") + ".json")
    save(target, result)
    print(target)
    print(
        json.dumps(
            [
                {
                    "date": c["target_date"],
                    "known_nav": c["known_nav_facts"],
                    "before_cutoff": c["issued_before_cutoff_count"],
                    "decisions": c["issued_before_cutoff_decisions"],
                    "one_day_forecasts": len(c["one_day_forecasts"]),
                    "stock_reconstruction": c["stock_reconstruction"],
                }
                for c in result["cases"]
            ],
            ensure_ascii=False,
        )
    )
