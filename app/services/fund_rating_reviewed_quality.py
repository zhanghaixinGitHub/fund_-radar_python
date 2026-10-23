"""经真实资料及用户审阅启用的八维质量规则，不伪称经验校准或独立产品验证。

连续项读取目标和完整参照观测，状态项读取有原件支撑的明确事实档位；不接收
维度分或总分。资料包不可变、原件逐次核验，发布复用现有事务、类别锁和审计。
"""

import hashlib
import re
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlsplit

from app.core.config import get_settings
from app.services.fund_rating_rules import DIMENSIONS, GRADES, grade, stored_score

ACTIVE_BASIS = "PUBLIC_REVIEWED_ACTIVE_QUALITY_V2"
PASSIVE_BASIS = "PUBLIC_REVIEWED_PASSIVE_QUALITY_V2"
BASES = frozenset((ACTIVE_BASIS, PASSIVE_BASIS))
CRITICAL = ("risk", "efficiency", "management", "liquidity")
STATES = {
    "SEVERE_FAILURE": 0,
    "MATERIAL_UNRESOLVED": 25,
    "LOCAL_UNCERTAINTY": 50,
    "BASIC_ROLE_EVIDENCED": 50,
    "REMEDIATION_DISCLOSED": 50,
    "NORMAL_PUBLIC_EXECUTION": 75,
    "CONTINUITY_EVIDENCED": 75,
    "NORMAL_REVIEWED": 75,
    "CROSS_PERIOD_STRONG": 100,
}


def policy(basis: str) -> dict:
    """固定原八维和内部预算；被动基金评价复制偏离，不惩罚合同主题集中。"""
    if basis not in BASES:
        raise ValueError("RATING_QUALITY_BASIS")
    passive = basis == PASSIVE_BASIS
    terms = {
        "return": [("return_1y", 30), ("return_3y", 70)],
        "risk": [("extra_drawdown", 70), ("extra_downside", 30)],
        "efficiency": [("tracking_error", 100)] if passive else [("allocation", 60), ("mandate", 40)],
        "stability": [("monthly_success", 30), ("annual_success", 70)],
        "management": [("experience", 40), ("continuity", 20), ("governance", 40)],
        "holdings": [("issuer_replication", 50), ("industry_replication", 50)]
        if passive
        else [("issuer_concentration", 35), ("industry_concentration", 35), ("asset_quality", 30)],
        "liquidity": [("market_capacity", 40), ("restricted", 20), ("known_payment", 25), ("holders", 15)],
        "cost": [("cost_365", 100)],
    }
    return {
        "basis": basis,
        "family": "INDEX" if passive else "ACTIVE_EQUITY",
        "revision": "reviewed-eight-dimensions-20261020",
        "weights": dict(
            zip(
                DIMENSIONS,
                (10, 10, 25, 10, 10, 10, 10, 15) if passive else (15, 20, 10, 10, 15, 10, 10, 10),
                strict=True,
            )
        ),
        # 配置会写入 JSONB 与输入快照，使用 JSON 原生数组保证读回后仍严格相等。
        "terms": {key: [[name, weight] for name, weight in values] for key, values in terms.items()},
        "minimum_products": 1,
        "critical_dimensions": list(CRITICAL),
        "grade_boundaries": [50, 70, 90],
        "weak_limit": 50,
        "excellent_minimum": 70,
        "acceptance": "REAL_SOURCE_ARITHMETIC_AND_EXPLICIT_USER_REVIEW",
        "independent_product_validation": False,
        "local_only": True,
        "verification_age_days": 31,
        "holdings_age_days": 210,
    }


def _fraction(value) -> Fraction:
    if value is None or isinstance(value, bool):
        raise ValueError("RATING_QUALITY_VALUE_MISSING")
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("RATING_QUALITY_VALUE_INVALID") from error


def _decimal(value: Fraction) -> Decimal:
    with localcontext() as context:
        context.prec = 70
        return Decimal(value.numerator) / Decimal(value.denominator)


def _node(node: dict, *, aliases: set[str]) -> Fraction:
    """闭合的业务观测类型；未知状态、缺观察和不完整参照均拒绝，不能默认为正常。"""
    kind = node.get("kind")
    if kind == "rank":
        peers = node["peers"]
        expected = node["expected_products"]
        identities = [p["product_id"] for p in peers]
        if (
            not peers
            or len(peers) > 5000
            or len(set(identities)) != len(peers)
            or sorted(identities) != sorted(expected)
            or set(identities) & aliases
        ):
            raise ValueError("RATING_QUALITY_PEER_CLOSURE")
        target = _fraction(node["target"])
        values = [_fraction(p["value"]) for p in peers]
        if node["direction"] == "higher":
            better = sum(v < target for v in values)
        elif node["direction"] == "lower":
            better = sum(v > target for v in values)
        else:
            raise ValueError("RATING_QUALITY_DIRECTION")
        return 100 * (Fraction(better) + Fraction(sum(v == target for v in values), 2)) / len(values)
    if kind == "windows":
        rows = node["rows"]
        if len(rows) != node["expected_windows"] or node["expected_windows"] not in {25, 36}:
            raise ValueError("RATING_QUALITY_WINDOW_CLOSURE")
        if len({r["end"] for r in rows}) != len(rows):
            raise ValueError("RATING_QUALITY_DUPLICATE_WINDOW")
        differences = [_fraction(r["target"]) - _fraction(r["reference"]) for r in rows]
        return (
            100
            * (Fraction(sum(d > 0 for d in differences)) + Fraction(sum(d == 0 for d in differences), 2))
            / len(rows)
        )
    if kind == "state":
        if not node.get("reason") or node.get("state") not in STATES:
            raise ValueError("RATING_QUALITY_STATE_EVIDENCE")
        return Fraction(STATES[node["state"]])
    if kind == "asset_quality":
        rows = node["positions"]
        if not rows or len({r["security"] for r in rows}) != len(rows):
            raise ValueError("RATING_QUALITY_ASSET_CLOSURE")
        amounts = [_fraction(r["amount"]) for r in rows]
        if any(a <= 0 for a in amounts) or sum(amounts) != _fraction(node["stock_total"]):
            raise ValueError("RATING_QUALITY_ASSET_RECONCILIATION")
        if any(r.get("state") not in STATES or not r.get("reason") for r in rows):
            raise ValueError("RATING_QUALITY_ASSET_STATE")
        return sum(a * STATES[r["state"]] for a, r in zip(amounts, rows, strict=True)) / sum(amounts)
    if kind == "restricted":
        amount, nav = _fraction(node["restricted_amount"]), _fraction(node["net_assets"])
        if amount < 0 or nav <= 0 or node.get("scope_closed") is not True:
            raise ValueError("RATING_QUALITY_RESTRICTED_SCOPE")
        return 100 * max(Fraction(0), 1 - amount / nav)
    if kind == "payment":
        cash, liabilities = _fraction(node["cash_principal"]), _fraction(node["all_book_liabilities"])
        if cash < 0 or liabilities < 0 or not node.get("reason"):
            raise ValueError("RATING_QUALITY_PAYMENT_FACTS")
        if cash >= liabilities:
            return Fraction(75)
        if node.get("state") not in {"SEVERE_FAILURE", "MATERIAL_UNRESOLVED", "LOCAL_UNCERTAINTY"}:
            raise ValueError("RATING_QUALITY_PAYMENT_GAP_UNRESOLVED")
        return Fraction(STATES[node["state"]])
    raise ValueError("RATING_QUALITY_NODE_UNSUPPORTED")


def calculate(metrics: dict, *, config: dict) -> dict:
    """从八维业务观测重算；等级限制单列，不再从总分扣减一次。"""
    if config != policy(config["basis"]):
        raise ValueError("RATING_QUALITY_POLICY_CHANGED")
    nodes = metrics["observations"]
    required = {name for terms in config["terms"].values() for name, _ in terms}
    if set(nodes) != required or metrics.get("material_event_pending") is not False:
        raise ValueError("RATING_QUALITY_REQUIRED_OBSERVATIONS")
    kinds = {
        "allocation": "state",
        "mandate": "state",
        "experience": "state",
        "continuity": "state",
        "governance": "state",
        "holders": "state",
        "monthly_success": "windows",
        "annual_success": "windows",
        "asset_quality": "asset_quality",
        "restricted": "restricted",
        "known_payment": "payment",
    }
    for name, node in nodes.items():
        if node.get("kind") != kinds.get(name, "rank"):
            raise ValueError("RATING_QUALITY_OBSERVATION_TYPE:" + name)
        if node["kind"] == "rank" and node.get("direction") != ("higher" if name.startswith("return_") else "lower"):
            raise ValueError("RATING_QUALITY_DIRECTION:" + name)
        if name in {"monthly_success", "annual_success"} and node["expected_windows"] != (
            36 if name == "monthly_success" else 25
        ):
            raise ValueError("RATING_QUALITY_WINDOW_TYPE")
    aliases = set(metrics["target_product_aliases"])
    if not aliases:
        raise ValueError("RATING_QUALITY_PRODUCT_IDENTITY")
    dimensions = {
        key: sum(_node(nodes[name], aliases=aliases) * weight / 100 for name, weight in terms)
        for key, terms in config["terms"].items()
    }
    raw = sum(dimensions[k] * config["weights"][k] / 100 for k in DIMENSIONS)
    score = stored_score(_decimal(raw))
    base = grade(score)
    final, caps = capped_grade(dimensions, base)
    return {
        "raw_score": str(_decimal(raw)),
        "score": str(score),
        "grade": final,
        "raw_grade": base,
        "grade_cap_reasons": caps,
        "dimensions": {k: str(_decimal(v)) for k, v in dimensions.items()},
    }


def capped_grade(dimensions: dict, base: str) -> tuple[str, list[str]]:
    values = {k: _fraction(dimensions[k]) for k in CRITICAL}
    weak = [k for k, v in values.items() if v < 50]
    if weak and base in {"GOOD", "EXCELLENT"}:
        return "AVERAGE", ["KEY_DIMENSION_WEAK:" + k for k in weak]
    below = [k for k, v in values.items() if v < 70]
    if below and base == "EXCELLENT":
        return "GOOD", ["EXCELLENT_FLOOR:" + k for k in below]
    return base, []


def validate_sources(bundle: dict, frozen: datetime, *, root: Path | None = None) -> None:
    """核验归档真实字节、日期、路径和节点来源；公开原件与研究派生文件明确分开。"""
    if get_settings().app_environment != "local" or bundle.get("source_scope") != "PUBLIC_DISCLOSURE_LOCAL_ANALYSIS":
        raise ValueError("RATING_QUALITY_LOCAL_SCOPE")
    root = (root or Path(get_settings().fund_insight_directory) / "ratings").resolve()
    directory = (root / "sources").resolve()
    refs = bundle["raw_archive"]
    if not 1 <= len(refs) <= 6000 or len({r["sha256"] for r in refs}) != len(refs):
        raise ValueError("RATING_QUALITY_SOURCE_CLOSURE")
    asof = date.fromisoformat(bundle["as_of_date"])
    hashes, originals = set(), set()
    paths = {}
    for source in refs:
        path = (root / source["file"]).resolve()
        received = datetime.fromisoformat(source["acquired_at"])
        if received.tzinfo is None or received > frozen:
            raise ValueError("RATING_QUALITY_SOURCE_FUTURE")
        if not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size > 50_000_000:
            raise ValueError("RATING_QUALITY_SOURCE_PATH")
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("RATING_QUALITY_SOURCE_HASH")
        if source["kind"] == "ORIGINAL":
            url = urlsplit(source["url"])
            from app.services.fund_rating_public_market import HOSTS

            allowed = HOSTS | {
                "www.dbfund.com.cn",
                "dbfund.com.cn",
                "static.cninfo.com.cn",
                "www.cninfo.com.cn",
                "www.bse.cn",
                "www.bseinfo.net",
                "www.sse.com.cn",
                "static.sse.com.cn",
                "www.szse.cn",
                "investor.szse.cn",
                "www.csrc.gov.cn",
                "sp.capco.org.cn",
                "dataclouds.cninfo.com.cn",
                "www.cnindex.com.cn",
                "api.tushare.pro",
                "np-cnotice-stock.eastmoney.com",
                "epaper.stcn.com",
            }
            provider = (
                source.get("source_code") == "TUSHARE_PRO_FUND" and source["url"] == get_settings().tushare_api_url
            )
            if (url.scheme != "https" or url.hostname not in allowed) and not provider or url.username or url.password:
                raise ValueError("RATING_QUALITY_SOURCE_URL")
            if source.get("published_on") and date.fromisoformat(source["published_on"]) > asof:
                raise ValueError("RATING_QUALITY_SOURCE_AFTER_CUTOFF")
            originals.add(source["sha256"])
        elif source["kind"] != "DERIVED":
            raise ValueError("RATING_QUALITY_SOURCE_KIND")
        hashes.add(source["sha256"])
        paths[source["sha256"]] = path
    if not originals:
        raise ValueError("RATING_QUALITY_ORIGINALS_REQUIRED")
    for node in bundle["quality_metrics"]["observations"].values():
        if not node.get("source_refs") or not set(node["source_refs"]) <= hashes:
            raise ValueError("RATING_QUALITY_NODE_SOURCE")
    identity = bundle["classification"]
    if not identity["relationship_verified"] or not set(identity["source_refs"]) <= originals:
        raise ValueError("RATING_QUALITY_IDENTITY_ORIGINAL")
    bindings = identity.get("identity_bindings")
    if not bindings or not any(bundle["fund_code"] in b.get("required_text", []) for b in bindings):
        raise ValueError("RATING_QUALITY_IDENTITY_BINDINGS")
    for binding in bindings:
        if binding["source_ref"] not in originals or not binding.get("required_text"):
            raise ValueError("RATING_QUALITY_IDENTITY_BINDINGS")
        path = paths[binding["source_ref"]]
        if path.suffix.lower() == ".pdf":
            import pypdfium2 as pdfium

            with pdfium.PdfDocument(path) as document:
                body = "".join(document[i].get_textpage().get_text_range() for i in range(min(12, len(document))))
        else:
            body = path.read_text(encoding="utf-8")
        body = re.sub(r"\s+", "", body)
        if any(re.sub(r"\s+", "", text) not in body for text in binding["required_text"]):
            raise ValueError("RATING_QUALITY_IDENTITY_ORIGINAL_MISMATCH")
    # 页面说明采用既有业务白名单，不接收额外内部字段或文件路径。
    from app.schemas.fund_rating import RatingComparison, RatingDates, RatingDimension, RatingPoint

    view = bundle["quality_public"]
    RatingComparison.model_validate(view["comparison"])
    RatingDates.model_validate(view["dataDates"])
    if [d["key"] for d in view["dimensions"]] != list(DIMENSIONS):
        raise ValueError("RATING_QUALITY_PUBLIC_EIGHT_DIMENSIONS")
    for item in view["dimensions"]:
        RatingDimension.model_validate(item)
    for item in view["strengths"] + view["weaknesses"]:
        RatingPoint.model_validate(item)


def build_metrics(bundle: dict, asof: date, frozen: datetime) -> tuple[dict, dict, datetime]:
    """观察截止日、资料有效期与取得时间分开，不能用今天替换原评价日期。"""
    config = policy(bundle["basis"])
    if str(asof) != bundle["as_of_date"] or bundle["family"] != config["family"]:
        raise ValueError("RATING_QUALITY_SCOPE")
    metrics = bundle["quality_metrics"]
    if bundle["classification"]["product_id"] not in metrics["target_product_aliases"]:
        raise ValueError("RATING_QUALITY_PRODUCT_IDENTITY")
    calculate(metrics, config=config)
    holding = date.fromisoformat(bundle["quality_public"]["dataDates"]["holdings"])
    acquired = datetime.fromisoformat(bundle["verified_at"])
    if acquired.tzinfo is None or acquired > frozen or holding > asof:
        raise ValueError("RATING_QUALITY_VERIFICATION_TIME")
    expiry = min(
        datetime.combine(holding, datetime.min.time(), UTC) + timedelta(days=210), acquired + timedelta(days=31)
    )
    if expiry <= frozen:
        raise ValueError("RATING_QUALITY_EVIDENCE_EXPIRED")
    return metrics, bundle["quality_public"]["dataDates"], expiry


def ingest(session, bundle: dict) -> str:
    """追加新口径证据；不撤回旧同日证据，保留原批次和历史回退能力。"""
    from sqlalchemy import or_, select, text

    from app.models.fund_rating import RatingClassification, RatingEvidence
    from app.services.fund_rating_inputs import digest

    now = datetime.now(UTC)
    code = bundle["fund_code"]
    if not session.scalar(text("SELECT 1 FROM fund_share_class WHERE fund_code=:c"), {"c": code}):
        raise ValueError("RATING_UNKNOWN_FUND")
    validate_sources(bundle, now)
    build_metrics(bundle, date.fromisoformat(bundle["as_of_date"]), now)
    c = bundle["classification"]
    category = digest({"basis": bundle["basis"], "currency": "CNY"})
    cid = digest({"code": code, "classification": c})
    asof = date.fromisoformat(bundle["as_of_date"])
    existing = session.scalars(
        select(RatingClassification).where(
            RatingClassification.fund_code == code,
            RatingClassification.revoked.is_(False),
            RatingClassification.effective_from <= asof,
            or_(RatingClassification.effective_to.is_(None), RatingClassification.effective_to >= asof),
        )
    ).all()
    if existing:
        # 方法切换不等于法律产品转型。复用已核准身份，不撤销旧分类或破坏旧批次。
        if len(existing) != 1 or existing[0].product_id not in bundle["quality_metrics"]["target_product_aliases"]:
            raise ValueError("RATING_QUALITY_EXISTING_IDENTITY_MISMATCH")
        previous = existing[0]
        if previous.family != bundle["family"] or previous.currency != "CNY":
            raise ValueError("RATING_QUALITY_EXISTING_CLASSIFICATION_MISMATCH")
        cid = previous.classification_id
    if not session.get(RatingClassification, cid):
        session.add(
            RatingClassification(
                classification_id=cid,
                fund_code=code,
                product_id=c["product_id"],
                category_code=category,
                family=bundle["family"],
                currency="CNY",
                eligibility="PUBLIC_LOCAL_ANALYSIS",
                effective_from=date.fromisoformat(c["effective_from"]),
                source_refs=c["source_refs"],
                verified_at=now,
                revoked=False,
            )
        )
        session.flush()
    key = digest(bundle)
    if not session.get(RatingEvidence, key):
        session.add(
            RatingEvidence(
                evidence_hash=key,
                fund_code=code,
                classification_id=cid,
                as_of_date=date.fromisoformat(bundle["as_of_date"]),
                payload=bundle,
                acquired_at=now,
                revoked=False,
            )
        )
    return key


def public_evidence(member: dict, computed: dict | None) -> dict:
    from copy import deepcopy

    value = deepcopy(member["bundle"]["quality_public"])
    for dimension in value["dimensions"]:
        g = grade(computed["dimensions"][dimension["key"]]) if computed else None
        dimension.update(grade=g, gradeLabel=GRADES[g] if g else None)
    return value


def validate_category(members: list[dict]) -> dict:
    """真实案例复算准入，明确不是产品互斥、时间隔离的经验校准证明。"""
    from app.services.fund_rating_inputs import digest

    admitted = [m for m in members if m["admitted"]]
    bases = {m["bundle"].get("basis") for m in admitted}
    if not admitted or len(bases) != 1 or not bases <= BASES:
        return {"eligible": False, "basis": ACTIVE_BASIS}
    basis = next(iter(bases))
    config = policy(basis)
    outcomes = {}
    for member in admitted:
        result = calculate(member["metrics"], config=config)
        # 对已冻结原始子项观测逐项做另一条Decimal加权，检查保存精度与总和。
        with localcontext() as context:
            context.prec = 60
            total = sum(Decimal(result["dimensions"][k]) * config["weights"][k] / 100 for k in DIMENSIONS)
        if abs(total - Decimal(result["score"])) > Decimal("1e-11"):
            raise ValueError("RATING_QUALITY_ARITHMETIC_MISMATCH")
        outcomes[member["fund_code"]] = {
            "result": result,
            "source_arithmetic": "PASS",
            "independent_product_validation": False,
        }
    return {
        "basis": basis,
        "eligible": True,
        "methodology_id": digest(config),
        "real_cases": outcomes,
        "distribution": dict(Counter(v["result"]["grade"] for v in outcomes.values())),
        "independent_product_validation": False,
        "validation_scope": "真实资料、八维算术和用户审阅启用；不声称跨产品独立验证通过",
    }
