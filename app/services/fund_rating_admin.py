"""评级资料核验导入、试算与回退；仅本机维护命令使用，不接受浏览器文件路径。"""

from copy import deepcopy
from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models.fund_rating import RatingClassification, RatingEvidence, RatingMethodology
from app.services.fund_rating_inputs import build_metrics, catalog, digest, prepare, validate_sources
from app.services.fund_rating_metrics import stability
from app.services.fund_rating_rules import DIMENSIONS, calculate, candidate


def ingest(session: Session, bundle: dict) -> str:
    """导入实际规范化原件；分类必须有官方产品关系来源，缺少核验不生成分类或证据。"""
    now = datetime.now(UTC)
    code = bundle["fund_code"]
    if len(code) != 6 or not code.isascii() or not code.isdigit():
        raise ValueError("RATING_CODE_INVALID")
    if not session.scalar(text("SELECT 1 FROM fund_share_class WHERE fund_code=:code"), {"code": code}):
        raise ValueError("RATING_UNKNOWN_FUND")
    validate_sources(session, bundle, now)
    as_of = date.fromisoformat(bundle["as_of_date"])
    classification = bundle["classification"]
    if not classification["relationship_verified"] or not classification["source_refs"]:
        raise ValueError("RATING_PRODUCT_RELATION_UNVERIFIED")
    # 比较键明确覆盖策略、仓位、地区、币种、购买资格和交易形式；不能仅用页面大类。
    peer = classification["peer"]
    required = {
        "asset",
        "strategy",
        "region",
        "currency",
        "equity_band",
        "theme",
        "duration_credit",
        "eligibility",
        "trading_form",
        "target_index",
    }
    if set(peer) != required or any(not isinstance(v, str) or not v.strip() for v in peer.values()):
        raise ValueError("RATING_PEER_DEFINITION_INCOMPLETE")
    if peer["currency"] != bundle["currency"] or classification["family"] != bundle["family"]:
        raise ValueError("RATING_CLASSIFICATION_CONFLICT")
    if date.fromisoformat(classification["effective_from"]) > as_of:
        raise ValueError("RATING_CLASSIFICATION_FUTURE")
    if not classification["product_id"] or not bundle["category_label"]:
        raise ValueError("RATING_CLASSIFICATION_IDENTITY")
    source_hashes = {r["content_hash"] for r in bundle["sources"]}
    if not set(classification["source_refs"]) <= source_hashes:
        raise ValueError("RATING_CLASSIFICATION_SOURCE_MISSING")
    # 先独立复算所有必要指标；不会接受外部直接填写分数。
    build_metrics(bundle, as_of, now)
    cid = digest({"fund_code": code, "classification": classification})
    existing = session.get(RatingClassification, cid)
    if not existing:
        session.add(
            RatingClassification(
                classification_id=cid,
                fund_code=code,
                product_id=classification["product_id"],
                category_code=digest(peer),
                family=bundle["family"],
                currency=bundle["currency"],
                eligibility=peer["eligibility"],
                effective_from=date.fromisoformat(classification["effective_from"]),
                effective_to=date.fromisoformat(classification["effective_to"])
                if classification.get("effective_to")
                else None,
                source_refs=classification["source_refs"],
                verified_at=now,
                revoked=False,
            )
        )
        session.flush()
    elif existing.revoked or existing.fund_code != code:
        raise ValueError("RATING_CLASSIFICATION_REVOKED")
    key = digest(bundle)
    if not session.get(RatingEvidence, key):
        # 同日资料更正不覆盖旧包；已发布结果会通过撤回字段即时失效。
        for old in session.scalars(
            select(RatingEvidence).where(
                RatingEvidence.fund_code == code, RatingEvidence.as_of_date == as_of, ~RatingEvidence.revoked
            )
        ):
            old.revoked = True
        session.add(
            RatingEvidence(
                evidence_hash=key,
                fund_code=code,
                classification_id=cid,
                as_of_date=as_of,
                payload=bundle,
                acquired_at=now,
                revoked=False,
            )
        )
    return key


def trial(session: Session, as_of: date) -> dict:
    """真实资料试算，不发布；保留覆盖、四级自然分布及逐维权重扰动敏感性。

    权重逐项上下调整20%，其他项按比例调整，仅用于稳定性诊断，不修改正式配置。
    原始试算必须保留；样本不足不生成虚构分布。全部维度依然参与每次试算。
    """
    now = datetime.now(UTC)
    rows = catalog(session)
    groups = prepare(session, rows, as_of, now)
    report = {"as_of": str(as_of), "created_at": now.isoformat(), "target": len(rows), "categories": []}
    for category, members in groups.items():
        family = members[0]["family"]
        admitted = [m for m in members if m["admitted"]]
        products = sum(m["representative"] for m in admitted)
        entry = {
            "category": category,
            "family": family,
            "shares": len(members),
            "products": products,
            "qualified_shares": len(admitted),
            "eligible": False,
            "distribution": {},
            "input_hash": digest(members),
            "blockers": sorted({r for m in members for r in m["reasons"]}),
            "sensitivity": [],
            "dimension_correlations": [],
        }
        if family != "OTHER" and products >= 30:
            config = candidate(family)
            stability(admitted)
            baseline = calculate(admitted, config)
            from collections import Counter

            entry["distribution"] = dict(Counter(r["grade"] for r in baseline))
            for dimension in DIMENSIONS:
                for factor in (Decimal("0.8"), Decimal("1.2")):
                    changed = deepcopy(config)
                    weight = Decimal(config["weights"][dimension])
                    for key in DIMENSIONS:
                        changed["weights"][key] = (
                            weight * factor
                            if key == dimension
                            else (Decimal(config["weights"][key]) * (100 - weight * factor) / (100 - weight))
                        )
                    # Decimal 除法最后一位余数归于一个未受扰动维度，不丢失八维。
                    adjust = next(k for k in DIMENSIONS if k != dimension)
                    changed["weights"][adjust] += Decimal(100) - sum(changed["weights"].values())
                    calculated = calculate(admitted, changed)
                    entry["sensitivity"].append(
                        {
                            "dimension": dimension,
                            "factor": str(factor),
                            "changed_grades": sum(
                                a["grade"] != b["grade"] for a, b in zip(baseline, calculated, strict=True)
                            ),
                        }
                    )
            from statistics import correlation

            keys = list(DIMENSIONS)
            for i, a in enumerate(keys):
                for b in keys[i + 1 :]:
                    xs = [float(r["dimensions"][a]) for r in baseline]
                    ys = [float(r["dimensions"][b]) for r in baseline]
                    entry["dimension_correlations"].append(
                        {
                            "a": a,
                            "b": b,
                            "correlation": correlation(xs, ys) if len(set(xs)) > 1 and len(set(ys)) > 1 else None,
                        }
                    )
            entry.update(eligible=True, methodology_id=digest(config))
        report["categories"].append(entry)
    return report


def activate(session: Session, family: str, report: dict, review: str) -> None:
    """锁定候选规则须有当前库可复验的真实试算和审阅记录；不接受测试样本代替真实覆盖。"""
    if not review.strip():
        raise ValueError("RATING_REVIEW_REQUIRED")
    current = trial(session, date.fromisoformat(report["as_of"]))
    actual = {r["category"]: r for r in current["categories"] if r["family"] == family and r["eligible"]}
    provided = {r["category"]: r for r in report["categories"] if r["family"] == family and r["eligible"]}
    if (
        not actual
        or set(actual) != set(provided)
        or any(actual[k]["input_hash"] != provided[k]["input_hash"] for k in actual)
    ):
        raise ValueError("RATING_REAL_TRIAL_REQUIRED")
    config = candidate(family)
    key = digest(config)
    method = session.get(RatingMethodology, key)
    if not method:
        method = RatingMethodology(methodology_id=key, family=family, config=config, created_at=datetime.now(UTC))
        session.add(method)
    method.validation = {"report": current, "review": review, "validated_at": datetime.now(UTC).isoformat()}
    method.active = True
