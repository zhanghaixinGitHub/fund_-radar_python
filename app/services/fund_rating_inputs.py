"""从现有库准备真实覆盖清单，从经核验原始包准备八维；页面不会调用此模块。"""

import hashlib
import json
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from urllib.parse import urlsplit

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models.fund_rating import RatingClassification, RatingEvidence
from app.services.fund_rating_metrics import history_metrics, holdings_metrics, standard_cost, total_return
from app.services.fund_rating_rules import CATEGORY_LABELS, DIMENSIONS
from app.services.fund_rating_specialized import distance, distribution_index, ratio, specialized_holdings, tracking


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str, allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def catalog(session: Session, *, limit: int = 5000) -> list[dict]:
    """有界、批量取数；产品主键未经来源核验，不能据此给同类样本去重。"""
    rows = (
        session.execute(
            text("""
        SELECT f.fund_code, f.fund_name, f.fund_type, f.source_code,
          p.invest_type, p.management_fee, p.custodian_fee, p.found_date,
          p.updated_at AS profile_checked_at, p.content_hash AS profile_hash,
          n.last_nav_date, n.nav_count, n.nav_updated_at, m.manager_count,
          s.enabled AS source_enabled, s.authorization_verified_at
        FROM fund_share_class f
        LEFT JOIN LATERAL (SELECT * FROM fund_profile p WHERE p.fund_code=f.fund_code
          ORDER BY p.updated_at DESC, p.fund_profile_id LIMIT 1) p ON true
        LEFT JOIN (SELECT fund_code,max(nav_date) last_nav_date,count(*) nav_count,
          max(updated_at) nav_updated_at FROM nav_daily GROUP BY fund_code) n ON n.fund_code=f.fund_code
        LEFT JOIN (SELECT fund_code,count(*) manager_count FROM fund_manager_assignment
          GROUP BY fund_code) m ON m.fund_code=f.fund_code
        LEFT JOIN source_registry s ON s.source_code=f.source_code
        WHERE f.status='ACTIVE' ORDER BY f.fund_code LIMIT :limit
    """),
            {"limit": limit + 1},
        )
        .mappings()
        .all()
    )
    if len(rows) > limit:
        raise ValueError("RATING_SCOPE_EXCEEDS_5000")
    return [dict(r) for r in rows]


def provisional_family(row: dict) -> str:
    """仅用于缺口盘点标签；粗分类不是正式可比类别，不得据此入分。"""
    invest = row.get("invest_type") or ""
    if "被动指数" in invest:
        return "INDEX"
    return {
        "STOCK": "ACTIVE_EQUITY",
        "MIXED": "ACTIVE_EQUITY",
        "BOND": "BOND",
        "MONEY": "MONEY",
        "QDII": "OTHER",
        "FOF": "FOF",
    }.get(row["fund_type"], "OTHER")


MISSING = {
    "return": "分红、折算及完整估值日历尚未核验，不能将累计净值直接作为总回报。",
    "risk": "缺少同一考察区间的已核验总回报序列。",
    "efficiency": "缺少同币种、同区间的已核验无风险收益资料。",
    "stability": "缺少完整月末观察值及已去重的同类比较样本。",
    "management": "经理任职记录已收录，但团队历史覆盖及策略归类仍需核验。",
    "holdings": "缺少同一报告期的完整持仓、发行人和行业分类。",
    "liquidity": "缺少完整资产变现资料及精确的最大单一持有人比例。",
    "cost": "管理费、托管费以外的全部适用费用及计费方式尚未核验。",
}


def empty_evidence(row: dict, family: str, as_of: date) -> dict:
    dimensions = [
        {
            "key": k,
            "name": name,
            "grade": None,
            "gradeLabel": None,
            "explanation": MISSING[k],
            "metrics": [],
            "sources": [],
        }
        for k, name in DIMENSIONS.items()
    ]
    if not row.get("manager_count"):
        dimensions[4]["explanation"] = "缺少已核验的完整团队任职历史及策略归类。"
    # 只展示已存的业务事实，不能把已存在的两个费率说成完整费用维度。
    for field, name in (("management_fee", "管理费年费率"), ("custodian_fee", "托管费年费率")):
        if row.get(field) is not None:
            dimensions[-1]["metrics"].append(
                {
                    "name": name,
                    "value": format(Decimal(str(row[field])).normalize(), "f"),
                    "unit": "%",
                    "period": "当前已收录资料，适用日期待核验",
                }
            )
    return {
        "summary": "必要资料尚未核验完整，暂未形成综合评级。",
        "comparison": {
            "category": CATEGORY_LABELS[family],
            "currency": None,
            "productCount": 0,
            "scope": "本站已收录基金；产品关系及细分类别待核验",
            "period": "考察区间待核验",
        },
        "dimensions": dimensions,
        "strengths": [],
        "weaknesses": [],
        "dataDates": {
            "nav": str(row["last_nav_date"]) if row.get("last_nav_date") else None,
            "holdings": None,
            "manager": None,
            "fees": None,
        },
        "limitations": [
            "产品与份额关系、币种及细分类别尚未完成来源核验。",
            "八个维度全部满足资料要求后才能评级；暂未评级不等于偏弱。",
        ],
    }


def validate_sources(session: Session, bundle: dict, frozen: datetime) -> None:
    """每一包均需可追溯官方引用、真实取得时间和来源授权；仅日期精度不伪造发布时刻。"""
    refs = bundle["sources"]
    if not refs or len(refs) > 100:
        raise ValueError("RATING_SOURCE_REQUIRED")
    source_codes = {r["source_code"] for r in refs}
    allowed = (
        session.execute(
            text(
                "SELECT source_code FROM source_registry WHERE enabled AND "
                "authorization_verified_at IS NOT NULL AND source_code=ANY(:codes)"
            ),
            {"codes": list(source_codes)},
        )
        .scalars()
        .all()
    )
    if source_codes != set(allowed):
        raise ValueError("RATING_SOURCE_NOT_AUTHORIZED")
    for r in refs:
        url = urlsplit(r["url"])
        if url.scheme != "https" or not url.netloc or url.username or url.password or not r["title"]:
            raise ValueError("RATING_SOURCE_URL")
        if len(r["content_hash"]) != 64 or any(c not in "0123456789abcdef" for c in r["content_hash"]):
            raise ValueError("RATING_SOURCE_DIGEST")
        acquired = datetime.fromisoformat(r["acquired_at"])
        if acquired.tzinfo is None or acquired > frozen or date.fromisoformat(r["published_on"]) > acquired.date():
            raise ValueError("RATING_SOURCE_TIME")
        if not r["revision"]:
            raise ValueError("RATING_SOURCE_REVISION")


def build_metrics(bundle: dict, as_of: date, frozen: datetime) -> tuple[dict, dict, datetime]:
    """按基金类别计算规范化原件。完整性由核验字段及原始数值共同约束。

    其他类别的专用原件尚无真实覆盖，保持未开放；不接受直接填写分项分数来绕过公式。
    """
    family = bundle["family"]
    if family not in CATEGORY_LABELS or family == "OTHER":
        raise ValueError("RATING_SPECIALIZED_INPUTS_UNVERIFIED")
    if bundle["as_of_date"] != str(as_of) or (not family.startswith("QDII") and bundle["currency"] != "CNY"):
        raise ValueError("RATING_INPUT_SCOPE")
    for field in (
        "events_complete",
        "calendar_verified",
        "full_holdings",
        "manager_history_complete",
        "fees_complete",
        "strategy_verified",
        "latest_report_verified",
        "retail_eligible",
    ):
        if bundle.get(field) is not True:
            raise ValueError("RATING_REQUIRED_VERIFICATION_" + field.upper())
    history = bundle["history"]
    if history["currency"] != bundle["currency"]:
        raise ValueError("RATING_CURRENCY_CONFLICT")
    if family != "MONEY" and history["risk_free_currency"] != bundle["currency"]:
        raise ValueError("RATING_RISK_FREE_CURRENCY")
    dates = [date.fromisoformat(d) for d in history["dates"]]
    if dates != [date.fromisoformat(d) for d in history["valuation_calendar"]] or dates[-1] != as_of:
        raise ValueError("RATING_CALENDAR_GAP")
    # 整个截止月份的来源估值日历必须齐全；不能因月末净值缺失偷换为更早的某一天。
    month_calendar = [date.fromisoformat(d) for d in history["as_of_month_calendar"]]
    if (
        not month_calendar
        or month_calendar != sorted(set(month_calendar))
        or month_calendar[-1] != as_of
        or any((d.year, d.month) != (as_of.year, as_of.month) for d in month_calendar)
    ):
        raise ValueError("RATING_MONTH_END_CALENDAR")
    if not family.startswith("QDII") and family != "MONEY" and history["annual_days"] != 252:
        raise ValueError("RATING_EQUITY_ANNUALIZATION")
    index = (
        distribution_index(history)
        if family == "MONEY"
        else total_return(history["unit_nav"], history["cash_dividend_per_share"], history["split_factor"])
    )
    metrics = history_metrics(
        dates,
        index,
        history.get("risk_free_interval_return", []) if family == "MONEY" else history["risk_free_interval_return"],
        [date.fromisoformat(d) for d in history["month_ends"]],
        history["annual_days"],
        require_sharpe=family != "MONEY",
    )
    h = bundle["holdings"]
    report = date.fromisoformat(h["report_date"])
    expected_unit = "YUAN" if bundle["currency"] == "CNY" else "CURRENCY_UNIT"
    if (
        not 0 <= (as_of - report).days <= 210
        or h["currency"] != bundle["currency"]
        or h["amount_unit"] != expected_unit
    ):
        raise ValueError("RATING_HOLDINGS_DATE_UNIT")
    extra = bundle.get("specialized", {})
    if family in {"ACTIVE_EQUITY", "QDII_EQUITY"}:
        metrics.update(
            holdings_metrics(h["positions"], h["net_assets"], h["cash"], h["payable_5d"], h["largest_holder"])
        )
        if family == "QDII_EQUITY":
            metrics["overseas_liquid_5d"] = metrics["liquid_5d"]
    else:
        metrics.update(specialized_holdings(family, h, extra))
    if family in {"INDEX", "QDII_COMMODITY"}:
        benchmark = extra["benchmark"]
        if benchmark["dates"] != history["dates"] or benchmark["currency"] != bundle["currency"]:
            raise ValueError("RATING_BENCHMARK_CALENDAR_CURRENCY")
        if benchmark["total_return_verified"] is not True or not benchmark["target_index"]:
            raise ValueError("RATING_BENCHMARK_DEFINITION")
        if benchmark["target_index"] != bundle["classification"]["peer"]["target_index"]:
            raise ValueError("RATING_BENCHMARK_TARGET")
        metrics.update(tracking(index, benchmark["index"], history["annual_days"]))
    if family in {"FOF", "MIXED_BOND"}:
        metrics["allocation_deviation"] = distance(extra["actual_allocation"], extra["target_allocation"])
    if family == "MONEY":
        stress = metrics["credit_loss_stress"] + metrics["duration_stress"]
        if stress <= 0:
            raise ValueError("RATING_ZERO_STRESS_DENOMINATOR")
        metrics["distribution_return_1y"] = metrics["return_1y"]
        metrics["distribution_stress_ratio"] = metrics["return_1y"] / stress
    if family.startswith("QDII") and any(
        extra.get(k) is not True
        for k in ("fx_basis_verified", "timezone_verified", "overseas_calendar_verified", "redemption_limits_verified")
    ):
        raise ValueError("RATING_OVERSEAS_BASIS_UNVERIFIED")
    team = bundle["manager"]
    fees = bundle["fees"]
    for block in (team, fees):
        checked = datetime.fromisoformat(block["checked_at"])
        if checked.tzinfo is None or not timedelta(0) <= frozen - checked <= timedelta(days=31):
            raise ValueError("RATING_VERIFICATION_EXPIRED")
        if date.fromisoformat(block["effective_date"]) > as_of:
            raise ValueError("RATING_FUTURE_EFFECTIVE_DATE")
    # 从完整团队生效区间重建变更日；同日多次人事变化只计一次。
    timeline = team["teams"]
    if not timeline or date.fromisoformat(timeline[0]["effective_date"]) > dates[0]:
        raise ValueError("RATING_MANAGER_HISTORY_GAP")
    events = {}
    for event in timeline:
        d = date.fromisoformat(event["effective_date"])
        if not event["members"] or d in events or d > as_of:
            raise ValueError("RATING_MANAGER_TIMELINE")
        events[d] = tuple(sorted(set(event["members"])))
    changes = []
    previous = None
    for d, members in sorted(events.items()):
        if previous is not None and members != previous:
            changes.append(d)
        previous = members
    last = max(changes) if changes else min(events)
    metrics["team_months"] = min(Decimal(36), Decimal((as_of - last).days) / Decimal("30.4375"))
    metrics["team_changes"] = sum(d > dates[0] for d in changes)
    if fees["redemption_holding_days"] != 365:
        raise ValueError("RATING_FEE_SCENARIO")
    if fees["trading_form"] == "OTC":
        metrics["cost_1y"] = standard_cost(
            *(fees[k] for k in ("management", "custody", "sales", "subscription", "redemption")),
            mode=fees["subscription_mode"],
        )
    elif fees["trading_form"] == "EXCHANGE":
        metrics["cost_1y"] = sum(
            ratio(fees[k])
            for k in (
                "management",
                "custody",
                "sales",
                "roundtrip_commission",
                "roundtrip_spread",
                "other_trading_cost",
            )
        )
    else:
        raise ValueError("RATING_FEE_TRADING_FORM")
    if family == "FOF":
        metrics["lookthrough_cost_1y"] = metrics["cost_1y"] + metrics["underlying_cost"]
    # 有效期是下一次更新月前五个真实交易日结束，日历不足时阻断，不能硬填固定日期。
    future = [date.fromisoformat(d) for d in bundle["next_update_calendar"]]
    next_month = (as_of.replace(day=28) + timedelta(days=4)).replace(day=1)
    update_month = (next_month.replace(day=28) + timedelta(days=4)).replace(day=1)
    if (
        future != sorted(set(future))
        or len(future) < 6
        or any(d.year != update_month.year or d.month != update_month.month for d in future)
    ):
        raise ValueError("RATING_EXPIRY_CALENDAR")
    from zoneinfo import ZoneInfo

    valid_until = datetime.combine(future[5], datetime.min.time(), ZoneInfo("Asia/Shanghai"))
    # 资料过期先于月度宽限时，采用更早边界，防止长期保留失效依据。
    valid_until = min(
        valid_until,
        datetime.combine(report + timedelta(days=211), datetime.min.time(), UTC),
        datetime.fromisoformat(team["checked_at"]) + timedelta(days=31),
        datetime.fromisoformat(fees["checked_at"]) + timedelta(days=31),
    )
    if valid_until <= frozen:
        raise ValueError("RATING_INPUT_ALREADY_EXPIRED")
    facts = {
        "nav": str(as_of),
        "holdings": str(report),
        "manager": team["checked_at"][:10],
        "fees": fees["checked_at"][:10],
    }
    return metrics, facts, valid_until


def prepare(session: Session, rows: list[dict], as_of: date, frozen: datetime) -> dict[str, list[dict]]:
    """一次读取完整候选集与证据；每类冻结前记录准入/排除理由，不由关注列表决定范围。"""
    codes = [r["fund_code"] for r in rows]
    evidence = session.scalars(
        select(RatingEvidence)
        .where(
            RatingEvidence.fund_code.in_(codes),
            RatingEvidence.as_of_date <= as_of,
            RatingEvidence.as_of_date >= as_of.replace(day=1),
            ~RatingEvidence.revoked,
            RatingEvidence.acquired_at <= frozen,
        )
        .order_by(RatingEvidence.as_of_date.desc(), RatingEvidence.acquired_at.desc(), RatingEvidence.evidence_hash)
    ).all()
    by_code = {}
    for e in evidence:
        by_code.setdefault(e.fund_code, e)
    classes = {
        c.classification_id: c
        for c in session.scalars(
            select(RatingClassification).where(RatingClassification.fund_code.in_(codes), ~RatingClassification.revoked)
        )
    }
    groups = defaultdict(list)
    for row in rows:
        family = provisional_family(row)
        member = {
            "fund_code": row["fund_code"],
            "fund_name": row["fund_name"],
            "product_id": None,
            "representative": False,
            "admitted": False,
            "metrics": {},
            "evidence_hash": None,
            "reasons": ["CLASSIFICATION_UNVERIFIED"],
            "family": family,
            "public": empty_evidence(row, family, as_of),
            "catalog": row,
        }
        category = digest({"unverified": family, "type": row.get("invest_type")})
        e = by_code.get(row["fund_code"])
        c = classes.get(e.classification_id) if e else None
        # 月末可能不是估值日；分类应覆盖该份原件的实际估值日，而非自然月最后一天。
        if c and c.effective_from <= e.as_of_date and (c.effective_to is None or c.effective_to >= e.as_of_date):
            category = c.category_code
            member.update(
                product_id=c.product_id,
                evidence_hash=e.evidence_hash,
                family=c.family,
                classification_id=c.classification_id,
                bundle=e.payload,
                reasons=[],
            )
            if digest(e.payload) != e.evidence_hash:
                raise ValueError("RATING_INPUT_HASH_MISMATCH")
            try:
                validate_sources(session, e.payload, frozen)
                metrics, facts, expiry = build_metrics(e.payload, e.as_of_date, frozen)
                member.update(
                    metrics=metrics,
                    admitted=True,
                    valid_until=expiry,
                    facts=facts,
                    found_date=e.payload["found_date"],
                    valuation_date=str(e.as_of_date),
                )
            except (KeyError, ValueError) as error:
                member["reasons"] = [str(error) if isinstance(error, ValueError) else "REQUIRED_INPUT_MISSING"]
                member["public"]["limitations"] = ["必要资料未通过完整性、日期或来源核验，暂未评级。"]
        groups[category].append(member)
    for members in groups.values():
        cutoffs = {m["valuation_date"] for m in members if m["admitted"]}
        if len(cutoffs) > 1:
            # 同类不能混合不同截止日，也不按可取得的短窗口迁就个别份额。
            for m in members:
                m.update(admitted=False, reasons=["CATEGORY_CUTOFF_CONFLICT"])
        products = defaultdict(list)
        for m in members:
            if m["admitted"]:
                products[m["product_id"]].append(m)
        for shares in products.values():
            # 成立时间、代码稳定排序，不以事后收益或费用择优。
            min(shares, key=lambda m: (m["found_date"], m["fund_code"]))["representative"] = True
    return dict(groups)
