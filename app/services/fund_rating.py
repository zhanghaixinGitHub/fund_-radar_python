"""八维评级任务和公共查询。写入仅由管理员手动任务或维护命令触发。"""

import json
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from time import monotonic

from sqlalchemy import select, text
from sqlalchemy.orm import Session, load_only

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import get_engine
from app.models.fund_rating import RatingBatch, RatingCurrent, RatingMember, RatingMethodology, RatingResult
from app.repositories.fund_rating import (
    RatingConflict,
    inputs_valid,
    load_validation_rows,
    persist_input,
    publish,
    summary,
    verify_batch,
)
from app.schemas.fund_rating import RatingDetail, RatingPage
from app.services.fund_rating_inputs import canonical, catalog, digest, empty_evidence, prepare
from app.services.fund_rating_metrics import stability
from app.services.fund_rating_rules import DIMENSIONS, GRADES, calculate, candidate, decimal, grade

logger = get_logger(__name__)
METRIC_LABELS = {
    "return_1y": ("近一年总回报", "%"),
    "return_3y": ("近三年年化总回报", "%"),
    "drawdown": ("最大回撤幅度", "%"),
    "downside": ("下行波动", "%"),
    "sharpe": ("夏普比率", ""),
    "stability": ("滚动一年窗口同类胜出比例", "%"),
    "team_months": ("现团队连续任职", "月"),
    "team_changes": ("团队变更", "次"),
    "issuer_hhi": ("发行人集中度", ""),
    "industry_hhi": ("行业集中度", ""),
    "liquid_5d": ("标准五日变现覆盖率", "%"),
    "largest_holder": ("最大单一持有人比例", "%"),
    "cost_1y": ("一万元持有一年标准成本率", "%"),
    "credit_loss_stress": ("信用压力损失比例", "%"),
    "duration_stress": ("利率上行情景损失比例", "%"),
    "lower_credit_ratio": ("较低信用资产比例", "%"),
    "bond_liquid_5d": ("债券五日变现覆盖率", "%"),
    "allocation_deviation": ("资产配置偏离", "%"),
    "equity_stress": ("权益下行情景损失比例", "%"),
    "conversion_premium": ("加权转股溢价率", "%"),
    "convertible_liquid_5d": ("可转债五日变现覆盖率", "%"),
    "tracking_error": ("年化跟踪误差", "%"),
    "tracking_bias": ("年化跟踪偏离幅度", "%"),
    "replication_deviation": ("指数复制偏离", "%"),
    "distribution_return_1y": ("近一年分配再投资收益", "%"),
    "distribution_stress_ratio": ("分配收益与压力损失之比", ""),
    "distribution_stability": ("分配收益同类胜出比例", "%"),
    "maturity_liquid_5d": ("五日到期及可赎回覆盖率", "%"),
    "lookthrough_hhi": ("穿透发行人集中度", ""),
    "overlap_ratio": ("底层资产重叠比例", "%"),
    "underlying_redeemable_5d": ("底层五日可赎回覆盖率", "%"),
    "lookthrough_cost_1y": ("扣除减免的穿透一年成本率", "%"),
    "counterparty_hhi": ("交易对手集中度", ""),
    "roll_cost": ("展期成本率", "%"),
    "overseas_liquid_5d": ("境外资产五日变现覆盖率", "%"),
}
EXPLANATIONS = {
    "return": "使用分红再投资总回报，与相同投资类型的基金产品比较。",
    "risk": "观察历史回撤及负收益波动，不表示未来风险上限。",
    "efficiency": "使用相同无风险收益口径，比较承担波动所获得的历史回报。",
    "stability": "观察25个月末滚动一年窗口，并列计半次；窗口相互重叠。",
    "management": "仅比较团队管理连续性，不等同于经理能力。",
    "holdings": "根据完整披露持仓比较发行人与行业集中程度，非实时持仓。",
    "liquidity": "按披露资产估算五日变现情景，不承诺实际赎回时间。",
    "cost": "采用统一标准费用场景，未使用个人折扣；不从历史收益重复扣费。",
}


def _public(member: dict, computed: dict | None, config: dict, product_count: int, as_of: date) -> dict:
    from app.services.fund_rating_reviewed_quality import BASES as REVIEWED_BASES

    if member.get("bundle", {}).get("basis") in REVIEWED_BASES:
        from app.services.fund_rating_reviewed_quality import public_evidence

        return public_evidence(member, computed)
    from app.services.fund_rating_public_policy import PUBLIC_BASES

    if config.get("basis") in PUBLIC_BASES:
        from app.services.fund_rating_public_view import public_evidence

        return public_evidence(member, computed, as_of)
    sources = [
        {"title": r["title"], "url": r["url"], "publishedOn": r["published_on"]} for r in member["bundle"]["sources"]
    ]
    dimensions = []
    period = f"{member['bundle']['history']['dates'][0]} 至 {as_of}"
    explanations = dict(EXPLANATIONS)
    family = member["family"]
    if family in {"INDEX", "QDII_COMMODITY"}:
        explanations["management"] = "比较同目标指数的跟踪误差与偏离，不要求主动跑赢指数。"
    if family in {"BOND", "SHORT_BOND", "MIXED_BOND", "QDII_BOND", "MONEY"}:
        explanations["risk"] = "使用披露信用与久期资料测算压力损失，低波动不等于低风险。"
        explanations["holdings"] = "比较完整债权资产的发行人集中程度与较低信用资产比例。"
    if family == "MONEY":
        explanations["return"] = "使用实际收益分配再投资，不使用近似恒定净值计算收益。"
        explanations["efficiency"] = "比较实际分配收益与同一信用、利率压力情景损失，不使用普通夏普比率。"
    if family in {"FOF", "MIXED_BOND"}:
        explanations["management"] = "根据完整目标配置与实际配置比较执行偏离，不推断经理个人能力。"
    if family == "FOF":
        explanations["holdings"] = "穿透底层基金识别发行人集中和重复持仓，不以底层基金数量代替分散程度。"
        explanations["cost"] = "合计外层与底层成本并扣除已证实的减免，避免同一费用重复计入。"
    if family == "INDEX":
        explanations["holdings"] = "比较完整持仓与目标指数权重的复制偏离，不惩罚指数本身的主题集中。"
    if family == "CONVERTIBLE":
        explanations["risk"] = "结合历史回撤与转股价值的权益下行情景，区分股性与债性风险。"
        explanations["holdings"] = "比较发行人集中程度与转股溢价，使用可转债专用比较类别。"
    if family == "QDII_COMMODITY":
        explanations["holdings"] = "比较交易对手集中程度与真实展期成本，不套用股票行业集中度。"
    for key, name in DIMENSIONS.items():
        g = grade(computed["dimensions"][key]) if computed else None
        metrics = []
        for metric, _, _ in config["formula"][key]:
            # 未达同类门槛时仍展示本基金已核验事实；尚无法比较的指标留空，不虚构分值。
            if metric not in member["metrics"]:
                continue
            label, unit = METRIC_LABELS[metric]
            value = decimal(member["metrics"][metric]) * (100 if unit == "%" else 1)
            metric_period = member["facts"]["holdings"] if key in {"holdings", "liquidity"} else period
            if key == "cost":
                metric_period = f"费用核对 {member['facts']['fees']}；标准持有365天"
            if metric in {"return_1y", "distribution_return_1y"}:
                metric_period = f"截至 {as_of} 的近一年"
            metrics.append(
                {
                    "name": label,
                    "value": f"{value:.4f}",
                    "unit": unit,
                    "period": metric_period,
                }
            )
        dimensions.append(
            {
                "key": key,
                "name": name,
                "grade": g,
                "gradeLabel": GRADES[g] if g else None,
                "explanation": explanations[key],
                "metrics": metrics,
                "sources": sources,
            }
        )

    def points(good):
        return [
            {
                "dimension": d["key"],
                "text": f"{d['name']}：{d['metrics'][0]['name']} "
                f"{d['metrics'][0]['value']}{d['metrics'][0]['unit']}，在本次同类比较中{d['gradeLabel']}。",
            }
            for d in dimensions
            if (d["grade"] in {"GOOD", "EXCELLENT"} if good else d["grade"] == "WEAK")
        ][:3]

    liquidity_note = "持仓为定期披露资料；五日变现假设每日参与过去20个交易日成交额中位数的10%。"
    if family == "MONEY":
        liquidity_note = "五日流动性使用已核验到期及可赎回金额，未知限制不能视为可变现。"
    elif family == "FOF":
        liquidity_note = "五日流动性受底层基金赎回安排和已知限制约束，不能视为到账承诺。"
    return {
        "summary": (
            f"在本次同类比较范围内，八维综合评价为{GRADES[computed['grade']]}。"
            if computed
            else "本基金原件已核验；同类完整资料或分类规则验证尚未满足要求，暂未形成综合评级。"
        ),
        "comparison": {
            "category": member["bundle"]["category_label"],
            "currency": member["bundle"].get("currency", "CNY"),
            "productCount": product_count,
            "scope": "本站已核验并去重的同类基金产品",
            "period": period,
        },
        "dimensions": dimensions,
        "strengths": points(True),
        "weaknesses": points(False),
        "dataDates": member["facts"],
        "limitations": [
            "历史评价不代表未来收益；不同类型的同名等级不能直接比较。",
            liquidity_note,
            "费用情景为一万元持有365天、无涨跌，不代表实际账户费用。",
        ],
    }


def update_ratings(
    fund_code: str | None = None,
    *,
    engine=None,
    root: Path | None = None,
    as_of: date | None = None,
    progress_reporter=None,
    due_only: bool = False,
    ready_codes: set[str] | None = None,
    event_review: bool = False,
) -> dict:
    """有界市场目录任务；单份额仅选所属完整类别，任何 GET 都不会调用此函数。"""
    engine = engine or get_engine()
    root = root or Path(get_settings().fund_insight_directory) / "ratings"
    started = monotonic()
    frozen = datetime.now(UTC)
    # 最近结束自然月；已核验资料包必须给出对应月末的有效估值日。
    month_end = frozen.astimezone(__import__("zoneinfo").ZoneInfo("Asia/Shanghai")).date().replace(day=1) - timedelta(
        days=1
    )
    if event_review and (not fund_code or as_of is None or due_only):
        raise ValueError("RATING_EVENT_REVIEW_SCOPE_REQUIRED")
    as_of = as_of or month_end
    latest_allowed = (
        frozen.astimezone(__import__("zoneinfo").ZoneInfo("Asia/Shanghai")).date() if event_review else month_end
    )
    if as_of > latest_allowed:
        raise ValueError("RATING_FUTURE_AS_OF")
    with engine.connect() as coordinator:
        if not coordinator.scalar(text("SELECT pg_try_advisory_lock(hashtextextended('fund-rating-task',0))")):
            raise RatingConflict("RATING_TASK_BUSY")
        coordinator.commit()
        try:
            with Session(engine.execution_options(isolation_level="REPEATABLE READ")) as session, session.begin():
                session.execute(text("SET LOCAL statement_timeout='15s'"))
                rows = catalog(session)
                from app.repositories.fund_rating import reviewed_current_bases
                from app.services.fund_rating_reviewed_snapshot import BASIS as SNAPSHOT_BASIS

                pinned = reviewed_current_bases(session, [r["fund_code"] for r in rows])
                snapshot_codes = sorted(code for code, basis in pinned.items() if basis == SNAPSHOT_BASIS)
                # 这些产品已按用户批准的部分/区间契约发布。旧采集器没有新契约的完整
                # 原始输入，不能把新结果降回旧公式；新期需要重新核验并追加资料清单。
                rows = [r for r in rows if r["fund_code"] not in snapshot_codes]
                groups = prepare(session, rows, as_of, frozen)
            if fund_code:
                if fund_code in snapshot_codes:
                    return {"target": 1, "rated": 0, "notRated": 0, "categories": [], "failures": [],
                            "grades": {}, "reasons": {}, "alreadyCurrent": [],
                            "deferred": [fund_code], "reviewedSnapshotUpdatesRequired": [fund_code]}
                groups = {k: v for k, v in groups.items() if any(m["fund_code"] == fund_code for m in v)}
                if not groups:
                    raise LookupError("RATING_FUND_NOT_FOUND")
                if event_review and not any(
                    m["admitted"] for members in groups.values() for m in members if m["fund_code"] == fund_code
                ):
                    raise ValueError("RATING_EVENT_REVIEW_DATA_INCOMPLETE")
            report = {
                "target": sum(map(len, groups.values())),
                "rated": 0,
                "notRated": 0,
                "categories": [],
                "failures": [],
                "grades": {},
                "reasons": {},
                "deferred": snapshot_codes if not fund_code else [],
                "reviewedSnapshotUpdatesRequired": snapshot_codes if not fund_code else [],
                "alreadyCurrent": [],
            }
            reasons, grades = Counter(), Counter()
            for category, members in groups.items():
                # 一键同步按完整比较类别发布。缺资料的基金保持原记录，不能以旧包重算
                # 或把本次采集失败写成新的未评级；管理员独立重评仍沿用原显式入口。
                admitted = {m["fund_code"] for m in members if m["admitted"]}
                if due_only:
                    with Session(engine) as check:
                        latest = check.scalars(
                            select(RatingBatch)
                            .join(RatingCurrent, RatingCurrent.batch_id == RatingBatch.batch_id)
                            .where(RatingBatch.category_code == category, RatingBatch.status == "PUBLISHED")
                        ).first()
                        completed_codes = (
                            set(
                                check.scalars(
                                    select(RatingResult.fund_code).where(
                                        RatingResult.batch_id == latest.batch_id, RatingResult.status == "RATED"
                                    )
                                )
                            )
                            if latest is not None
                            else set()
                        )
                    # 事项复核可发生在月中；既不能用旧月评覆盖较新的事项结论，也不能
                    # 让月中复核替代月底的完整月评。旧快照无cadence时按既有月评处理。
                    if latest is not None and (
                        latest.as_of_date > as_of
                        or (
                            latest.as_of_date.year == as_of.year
                            and latest.as_of_date.month == as_of.month
                            and latest.input_snapshot.get("cadence") != "EVENT_REVIEW"
                            and bool(completed_codes)
                            and admitted.issubset(completed_codes)
                        )
                    ):
                        report["alreadyCurrent"].append(category)
                        continue
                if ready_codes is not None and (not admitted or not admitted.issubset(ready_codes)):
                    report["deferred"].append(category)
                    continue
                if progress_reporter:
                    progress_reporter(
                        report["rated"] + report["notRated"],
                        report["target"],
                        None,
                        f"正在核验{members[0]['public']['comparison']['category']}完整比较类别",
                    )
                if monotonic() - started > 180:
                    raise TimeoutError("RATING_TASK_TIME_LIMIT")
                try:
                    dates = {m["valuation_date"] for m in members if m["admitted"]}
                    category_as_of = date.fromisoformat(next(iter(dates))) if dates else as_of
                    result = _category(
                        engine,
                        root,
                        category,
                        members,
                        category_as_of,
                        frozen,
                        evaluation_kind="EVENT_REVIEW" if event_review else "MONTHLY" if due_only else None,
                    )
                    report["categories"].append(result)
                    report["rated"] += result["rated"]
                    report["notRated"] += len(members) - result["rated"]
                    reasons.update(result["reasons"])
                    grades.update(result["grades"])
                except Exception:
                    logger.exception("fund_rating.calculate >>> category failed, category=%s", category)
                    report["failures"].append(category)
            report.update(reasons=dict(reasons), grades=dict(grades), elapsedSeconds=round(monotonic() - started, 3))
            return report
        finally:
            coordinator.execute(text("SELECT pg_advisory_unlock(hashtextextended('fund-rating-task',0))"))
            coordinator.commit()


def _category(engine, root, category, members, as_of, frozen, *, evaluation_kind=None) -> dict:
    family = members[0]["family"]
    if any(m["family"] != family for m in members):
        raise ValueError("RATING_MIXED_CATEGORY")
    from app.services.fund_rating_public_policy import PUBLIC_BASES, calculate_public, policy

    bases = {m.get("bundle", {}).get("basis") for m in members if m.get("bundle", {}).get("basis") in PUBLIC_BASES}
    public_basis = bool(bases)
    if len(bases) > 1 or (
        public_basis and any(m["admitted"] and m.get("bundle", {}).get("basis") not in bases for m in members)
    ):
        raise ValueError("RATING_MIXED_METHODS")
    config = policy(next(iter(bases))) if public_basis else candidate(family) if family != "OTHER" else None
    method_id = digest(config) if config else None
    admitted = [m for m in members if m["admitted"]]
    product_count = sum(m["representative"] for m in admitted)
    result_rows = []
    with Session(engine) as session, session.begin():
        session.execute(text("SET LOCAL statement_timeout='15s'"))
        session.execute(text("SET LOCAL lock_timeout='3s'"))
        from app.repositories.fund_rating import lock_category

        lock_category(session, category)
        method = session.get(RatingMethodology, method_id) if method_id else None
        if config and not method:
            method = RatingMethodology(
                methodology_id=method_id, family=family, config=config, active=False, created_at=frozen
            )
            session.add(method)
            session.flush()
        validated_categories = (
            {r["category"] for r in (method.validation or {}).get("report", {}).get("categories", []) if r["eligible"]}
            if method
            else set()
        )
        ready = method and method.active and category in validated_categories
        computed = {}
        if ready and product_count >= config["minimum_products"]:
            if public_basis:
                computed = {m["fund_code"]: calculate_public(m["metrics"], config=config) for m in admitted}
            else:
                stability(admitted)
                computed = {r["fund_code"]: r for r in calculate(admitted, config)}
        for m in members:
            if m["fund_code"] not in computed and not m["reasons"]:
                m["reasons"] = ["METHOD_NOT_VALIDATED" if not ready else "INSUFFICIENT_PRODUCTS"]
            if m.get("preparation"):
                from app.services.fund_rating_materials import archive_materials

                m["preparation"]["raw_archive"] = archive_materials(m["preparation"], root, frozen)
        # 冻结实际数值、来源版本、方法、目录与样本；同一输入重复任务复用批次。
        snapshot = json.loads(
            canonical(
                {
                    "category": category,
                    "as_of": as_of,
                    "method": config,
                    "method_active": bool(ready),
                    "members": members,
                }
            )
        )
        if evaluation_kind:
            snapshot["cadence"] = evaluation_kind
        input_hash, path = persist_input(root, snapshot)
        samples = sorted(m["fund_code"] for m in members if m["representative"])
        batch_id = digest(
            {
                "category": category,
                "as_of": str(as_of),
                "method": method_id,
                "input": input_hash,
                "samples": digest(samples),
            }
        )
        existing = session.get(RatingBatch, batch_id)
        if existing:
            result_rows = verify_batch(session, existing)
            if existing.withdrawn or not inputs_valid(session, existing):
                raise ValueError("RATING_EXISTING_WITHDRAWN")
            current = session.get(RatingCurrent, category)
            if not current or current.batch_id != batch_id:
                publish(session, existing)
        else:
            batch = RatingBatch(
                batch_id=batch_id,
                category_code=category,
                methodology_id=method_id,
                as_of_date=as_of,
                frozen_at=frozen,
                input_hash=input_hash,
                input_path=path,
                sample_hash=digest(samples),
                input_snapshot=snapshot,
                member_count=len(members),
                product_count=product_count,
                status="COMPLETE",
                completed_at=datetime.now(UTC),
                withdrawn=False,
            )
            session.add(batch)
            session.flush()
            for m in members:
                c = computed.get(m["fund_code"])
                session.add(
                    RatingMember(
                        batch_id=batch_id,
                        fund_code=m["fund_code"],
                        product_id=m["product_id"],
                        representative=m["representative"],
                        admitted=m["admitted"],
                        reasons=m["reasons"],
                        evidence_hash=m["evidence_hash"],
                    )
                )
                r = RatingResult(
                    batch_id=batch_id,
                    fund_code=m["fund_code"],
                    status="RATED" if c else "NOT_RATED" if family == "ACTIVE_EQUITY" else "UNSUPPORTED",
                    raw_score=c["raw_score"] if c else None,
                    score=c["score"] if c else None,
                    grade=c["grade"] if c else None,
                    dimension_values=c["dimensions"] if c else None,
                    public_evidence=_public(m, c, config, product_count, as_of) if m["admitted"] else m["public"],
                    valid_until=m.get("valid_until") if c else None,
                )
                result_rows.append(r)
                session.add(r)
            session.flush()
            publish(session, batch)
        counts = Counter(r.grade for r in result_rows if r.status == "RATED")
    # 提交后独立读回，确认不是只在 ORM 内看见待提交对象。
    with Session(engine) as check:
        verify_batch(check, check.get(RatingBatch, batch_id))
    logger.info(
        "fund_rating.calculate >>> complete, category=%s, as_of=%s, products=%s, rated=%s, members=%s",
        category,
        as_of,
        product_count,
        sum(counts.values()),
        len(members),
    )
    return {
        "category": category,
        "family": family,
        "batch": batch_id,
        "products": product_count,
        "rated": sum(counts.values()),
        "grades": dict(counts),
        "reasons": dict(Counter(m["reasons"][0] for m in members if m["reasons"])),
    }


def read(codes: list[str], *, rating_ref: str | None = None, detail: bool = False, engine=None):
    """一次 REPEATABLE READ 快照解析当前指针；不采集、不计算、不写库、不读个人数据。"""
    if not 1 <= len(codes) <= 100 or any(len(c) != 6 or not c.isascii() or not c.isdigit() for c in codes):
        raise ValueError("RATING_CODES_INVALID")
    codes = list(dict.fromkeys(codes))
    engine = engine or get_engine()
    now = datetime.now(UTC)
    with Session(engine.execution_options(isolation_level="REPEATABLE READ")) as session, session.begin():
        session.execute(text("SET TRANSACTION READ ONLY"))
        session.execute(text("SET LOCAL statement_timeout='5s'"))
        known = set(
            session.execute(
                text("SELECT fund_code FROM fund_share_class WHERE fund_code=ANY(:codes)"), {"codes": codes}
            ).scalars()
        )
        # 先定位批次，每份完整输入快照只读取一次。直接联表会在同批每只基金上重复
        # 传输、解析同一大份 JSON；列表摘要也不需要加载完整的八维展示正文。
        query = select(RatingBatch).where(
            RatingBatch.batch_id.in_(select(RatingResult.batch_id).where(RatingResult.fund_code.in_(codes))),
            RatingBatch.status == "PUBLISHED",
        )
        if rating_ref:
            query = query.where(RatingBatch.batch_id == rating_ref)
        else:
            query = query.where(RatingBatch.batch_id.in_(select(RatingCurrent.batch_id)))
        batches = session.scalars(query.order_by(RatingBatch.as_of_date.desc(), RatingBatch.completed_at.desc())).all()
        context = load_validation_rows(session, batches)
        method_ids = {batch.methodology_id for batch in batches if batch.methodology_id}
        methods = (
            {
                method.methodology_id: method
                for method in session.scalars(
                    select(RatingMethodology)
                    .where(RatingMethodology.methodology_id.in_(method_ids))
                    .options(
                        load_only(
                            RatingMethodology.methodology_id, RatingMethodology.active, RatingMethodology.validation
                        )
                    )
                )
            }
            if method_ids
            else {}
        )
        by_code = {}
        validity = {}
        for batch in batches:
            if batch.batch_id not in validity:
                complete_results = verify_batch(session, batch, context=context)
                valid = not batch.withdrawn and inputs_valid(session, batch, context=context)
                method = methods.get(batch.methodology_id)
                # 一批可能同时含缺资料行和已评级行；不能以本次查询遇到的第一行决定规则是否有效。
                if (
                    any(r.status in {"RATED", "PARTIAL"} for r in complete_results)
                    or batch.input_snapshot.get("method", {}).get("basis") == "PUBLIC_REVIEWED_QUALITY_SNAPSHOT_V3"
                ):
                    allowed = (
                        {
                            r["category"]
                            for r in (method.validation or {}).get("report", {}).get("categories", [])
                            if r["eligible"]
                        }
                        if method
                        else set()
                    )
                    if not method or not method.active or batch.category_code not in allowed:
                        valid = False
                validity[batch.batch_id] = valid
            for result in context.results.get(batch.batch_id, []):
                if result.fund_code in known:
                    by_code.setdefault(result.fund_code, (result, batch))
        items = []
        for code in codes:
            if code not in known:
                if detail:
                    raise LookupError("RATING_FUND_NOT_FOUND")
                items.append({"fundCode": code, "status": "NOT_FOUND", "message": "基金不存在"})
                continue
            row = by_code.get(code)
            if rating_ref and not row:
                raise LookupError("RATING_REFERENCE_NOT_FOUND")
            if row:
                result, batch = row
                item = summary(result, batch, invalid=not validity[batch.batch_id], now=now)
                if not rating_ref and item["status"] in {"RATED", "PARTIAL"}:
                    from app.services.fund_market_review import rating_review_required

                    if rating_review_required(code):
                        # 复用已有资料待更新状态，不篡改历史分数；新事项没有审阅前不继续显示有效等级。
                        item.update(
                            status="STALE", grade=None, gradeLabel=None, message="新增事项待复核，暂不展示当前等级"
                        )
                if detail:
                    item.update(result.public_evidence)
                    if item["status"] != "RATED":
                        # 历史评级明确另列，当前八维不继续显示旧等级暗示仍然有效。
                        if result.status == "RATED" and validity[batch.batch_id]:
                            item["previousRating"] = {
                                **summary(result, batch, invalid=False, now=batch.completed_at),
                                "message": "上次评级（已过有效期）",
                            }
                        item["summary"] = (
                            "评级依据需要更新，当前不展示综合等级。"
                            if item["status"] in {"STALE", "WITHDRAWN"}
                            else item["summary"]
                        )
                        item["strengths"], item["weaknesses"] = [], []
                        item["dimensions"] = [{**d, "grade": None, "gradeLabel": None} for d in item["dimensions"]]
                    # 数字分与补充事实严格来自当前已核验批次；不重新计算、不发布研究分。
                    from app.services.fund_rating_presentation import enrich_published_detail

                    item = enrich_published_detail(item, result, batch.input_snapshot)
            else:
                item = {"fundCode": code, "status": "NOT_RATED", "message": "暂未评级"}
                if detail:
                    item.update(empty_evidence({"fund_code": code}, "OTHER", now.date()))
            items.append(item)
        return RatingDetail.model_validate(items[0]).model_dump() if detail else RatingPage(items=items).model_dump()
