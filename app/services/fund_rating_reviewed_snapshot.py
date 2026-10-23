"""将用户批准的已有研究结果追加到正式批次，保留区间、缺项及覆盖率。

这是本机受控导入入口，不接收浏览器提交的分数。原研究原件按哈希归档，
完整八维仍为空时绝不生成总分或等级；货币M3可核的三维由正式纯计算器重算。
未完成的独立方法验收如实封存，不伪造旧校准通过记录。
"""

import hashlib
import json
import re
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, localcontext
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.fund import FundMaster, FundShareClass
from app.models.fund_rating import (
    RatingBatch,
    RatingClassification,
    RatingEvidence,
    RatingMember,
    RatingMethodology,
    RatingResult,
)
from app.services.fund_rating_inputs import digest

BASIS = "PUBLIC_REVIEWED_QUALITY_SNAPSHOT_V3"
KEYS = {
    "return": "return",
    "risk": "risk",
    "execution": "efficiency",
    "stability": "stability",
    "management": "management",
    "holdings": "holdings",
    "liquidity": "liquidity",
    "fee": "cost",
}
NAMES = dict(
    zip(
        KEYS,
        ["收益能力", "风险控制", "目标执行", "业绩稳定性", "管理与策略", "持仓结构", "规模与流动性", "费用"],
        strict=True,
    )
)
PUBLIC_EXPLANATIONS = {
    "return": "使用本份额实际费后收益评价；缺少完整可比收益或适配参照时保留范围。",
    "risk": "按基金类型评价已核风险观测；未明确的信用、期限等暴露仍保留缺项。",
    "execution": "核对已公开的合同约束和实际配置；尚未核准的资产资格、历史偏离与调整记录继续列为缺项。",
    "stability": "观察跨月份和滚动年度的持续表现；比较资料不完整时保留范围。",
    "management": "根据已核的经理职责、交接连续性和治理事项评价。",
    "holdings": "核对持仓集中、资产质量及适用的底层穿透；未识别机构或信用依据不足的部分保留范围。",
    "liquidity": "根据已公开的资产可变现性、现金负债及持有人结构评价；逐笔支取和到账条件未明确的部分仍为缺项。",
    "fee": "按该份额公开费用和统一持有场景评价；底层费用减免等未明确部分保留缺项。",
}


def archive_root() -> Path:
    return (Path(get_settings().fund_insight_directory) / "ratings").resolve()


def _number(value) -> Decimal:
    if value is None or isinstance(value, bool):
        raise ValueError("RATING_SNAPSHOT_NUMBER_MISSING")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("RATING_SNAPSHOT_NUMBER_INVALID")
    return result


def _read(ref: dict, root: Path) -> dict | None:
    """仅允许归档sources内的实际文件；PDF只校验字节，不执行内容。"""
    path = (root / ref["file"]).resolve()
    if not path.is_relative_to(root / "sources") or not path.is_file() or path.stat().st_size > 25_000_000:
        raise ValueError("RATING_SNAPSHOT_ARCHIVE_PATH")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != ref["sha256"]:
        raise ValueError("RATING_SNAPSHOT_ARCHIVE_HASH")
    return json.loads(content.decode("utf-8-sig")) if ref.get("kind") == "json" else None


def _case(document: dict, code: str, format_name: str) -> dict:
    if format_name == "nine_cases":
        matches = [r for r in document["all_cases"] if r["code"] == code]
        if len(matches) != 1:
            raise ValueError("RATING_SNAPSHOT_CASE_IDENTITY")
        return matches[0]
    if document.get("code") != code:
        raise ValueError("RATING_SNAPSHOT_CASE_IDENTITY")
    return document


def validate_sources(bundle: dict, frozen: datetime, *, root: Path | None = None) -> None:
    """研究取得日可晚于截止日；报告本身原公开日必须早于评价截止日。"""
    if get_settings().app_environment != "local" or bundle.get("basis") != BASIS:
        raise ValueError("RATING_SNAPSHOT_LOCAL_SCOPE")
    root = (root or archive_root()).resolve()
    document = _read(bundle["research_archive"], root)
    if _case(document, bundle["fund_code"], bundle["research_format"]) != bundle["reviewed_document"]:
        raise ValueError("RATING_SNAPSHOT_DOCUMENT_MISMATCH")
    identity = _read(bundle["identity_archive"], root)
    if not identity.get("identity_verified") or identity["code"] != bundle["fund_code"]:
        raise ValueError("RATING_SNAPSHOT_ORIGINAL_IDENTITY")
    if identity["published_on"] > bundle["as_of_date"]:
        raise ValueError("RATING_SNAPSHOT_ORIGINAL_FUTURE")
    _read(bundle["original_report_archive"], root)
    if identity["source"]["sha256"] != bundle["original_report_archive"]["sha256"]:
        raise ValueError("RATING_SNAPSHOT_REPORT_IDENTITY_HASH")
    plain = re.sub(r"\s+", "", "".join(identity["pages"]))
    if (
        bundle["fund_code"] not in plain
        or bundle["identity"]["manager_name"] not in plain
        or re.sub(r"\s+", "", bundle["identity"]["legal_name"]) not in re.sub(r"\s+", "", identity["title"])
    ):
        raise ValueError("RATING_SNAPSHOT_LEGAL_IDENTITY_MISMATCH")
    if bundle.get("money_input_archive") and _read(bundle["money_input_archive"], root) != bundle.get("money_inputs"):
        raise ValueError("RATING_SNAPSHOT_MONEY_INPUT_HASH")
    for ref in bundle.get("extra_archives", []):
        _read(ref, root)
    verified = datetime.fromisoformat(bundle["verified_at"])
    if verified.tzinfo is None or verified > frozen or verified + timedelta(days=31) <= frozen:
        raise ValueError("RATING_SNAPSHOT_VERIFICATION_EXPIRED")
    if bundle["holding_date"] > bundle["as_of_date"] or date.fromisoformat(bundle["as_of_date"]) > frozen.date():
        raise ValueError("RATING_SNAPSHOT_DATE_FUTURE")


def calculate(bundle: dict) -> dict:
    """从封存的逐维研究观测复算原权重合成；不把0..100域当已评分区间。"""
    with localcontext() as context:
        context.prec = 80
        doc = bundle["reviewed_document"]
        code = bundle["fund_code"]
        if doc.get("code") != code:
            raise ValueError("RATING_SNAPSHOT_CASE_IDENTITY")
        if doc.get("as_of", doc.get("cutoff")) != bundle["as_of_date"]:
            raise ValueError("RATING_SNAPSHOT_AS_OF_MISMATCH")
        weights = doc["outer_weights_percent"]
        if set(weights) != set(KEYS) or sum(_number(v) for v in weights.values()) != 100:
            raise ValueError("RATING_SNAPSHOT_EIGHT_WEIGHTS")
        if set(doc["dimensions"]) != set(KEYS):
            raise ValueError("RATING_SNAPSHOT_EIGHT_DIMENSIONS")
        dimensions = {}
        # JSONB不保留对象键顺序；始终按唯一八维顺序复算和生成展示数组。
        for key in KEYS:
            value = doc["dimensions"][key]
            if isinstance(value, dict):
                lo, hi = _number(value["lower"]), _number(value["upper"])
                point = value.get("candidate")
                reason = value.get("reason", "")
            elif value is None:
                lo, hi, point, reason = Decimal(0), Decimal(100), None, doc.get("specific_blocker", "必要资料尚未核准")
            else:
                lo = hi = _number(value)
                point = str(value)
                reason = "沿用已核原件及原方法的确定候选"
            if not 0 <= lo <= hi <= 100 or (point is not None and not lo == hi == _number(point)):
                raise ValueError("RATING_SNAPSHOT_INTERVAL_INVALID")
            dimensions[key] = {
                "lower": str(lo),
                "upper": str(hi),
                "candidate": str(point) if point is not None else None,
                "reason": reason or "按公开披露数据计算；未公开的底层信息仍保留缺项",
            }
        if bundle.get("money_inputs"):
            from app.services.fund_rating_money_public import execution, holdings, liquidity

            raw = bundle["money_inputs"]
            values = {
                "execution": execution(raw["numeric_execution_observations"], raw["category_observations"]),
                "holdings": holdings(raw["holdings"]),
                "liquidity": liquidity(raw["liquidity"]),
            }
            for key, value in values.items():
                for bound in ["lower", "upper"]:
                    if abs(_number(value[bound]) - _number(dimensions[key][bound])) > Decimal("1e-60"):
                        raise ValueError("RATING_SNAPSHOT_MONEY_RECALCULATION")
        points = [k for k, v in dimensions.items() if v["candidate"] is not None]
        ranges = [
            k
            for k, v in dimensions.items()
            if v["candidate"] is None and (Decimal(v["lower"]) > 0 or Decimal(v["upper"]) < 100)
        ]
        point_weight = sum(_number(weights[k]) for k in points)
        range_weight = sum(_number(weights[k]) for k in ranges)
        known = sum(_number(weights[k]) * _number(dimensions[k]["candidate"]) / 100 for k in points)
        lower = sum(_number(weights[k]) * _number(dimensions[k]["lower"]) / 100 for k in KEYS)
        upper = sum(_number(weights[k]) * _number(dimensions[k]["upper"]) / 100 for k in KEYS)
        return {
            "dimensions": dimensions,
            "weights": weights,
            "point_weight": str(point_weight),
            "interval_weight": str(range_weight),
            "missing_weight": str(100 - point_weight - range_weight),
            "point_dimensions": len(points),
            "interval_dimensions": len(ranges),
            "partial_score": str(known * 100 / point_weight) if point_weight else None,
            "lower": str(lower),
            "upper": str(upper),
            "has_full_informative_interval": point_weight + range_weight == 100,
            "status": "PARTIAL" if point_weight > 0 else "NOT_RATED",
            "score": None,
            "grade": None,
        }


def _display(value) -> str:
    return f"{_number(value):.2f}"


def public_evidence(bundle: dict, calculated: dict) -> dict:
    """只传基金业务结果；文件路径、方法号、审批与调度细节不进入页面。"""
    coverage = {
        "pointWeight": calculated["point_weight"],
        "intervalWeight": calculated["interval_weight"],
        "missingWeight": calculated["missing_weight"],
        "pointDimensions": calculated["point_dimensions"],
        "intervalDimensions": calculated["interval_dimensions"],
    }
    all_interval = calculated["has_full_informative_interval"]
    source = {
        "title": bundle["report_title"],
        "url": bundle["report_url"],
        "publishedOn": bundle["report_published_on"],
    }
    dimensions = []
    for key, value in calculated["dimensions"].items():
        meaningful = value["candidate"] is None and (Decimal(value["lower"]) > 0 or Decimal(value["upper"]) < 100)
        dimensions.append(
            {
                "key": KEYS[key],
                "name": NAMES[key],
                "grade": None,
                "gradeLabel": None,
                "score": _display(value["candidate"]) if value["candidate"] is not None else None,
                "scoreRange": {"lower": _display(value["lower"]), "upper": _display(value["upper"])}
                if meaningful
                else None,
                "explanation": bundle.get("public_explanations", {}).get(key, PUBLIC_EXPLANATIONS[key]),
                "metrics": [],
                "sources": [source],
            }
        )
    point_count = calculated["point_dimensions"]
    summary = (
        f"八个维度均已有分数或有效范围；其中{point_count}个维度得分确定，覆盖原权重{calculated['point_weight']}%。"
        "尚未给出单点总分和等级。"
        if all_interval
        else f"已有{point_count}个维度可确定评分，覆盖原权重{calculated['point_weight']}%；"
        "其余资料或方法依据仍不足，暂不提供完整总分和等级。"
    )
    return {
        "score": None,
        "scoreRange": {"lower": _display(calculated["lower"]), "upper": _display(calculated["upper"])}
        if all_interval
        else None,
        "partialScore": _display(calculated["partial_score"]) if calculated["partial_score"] is not None else None,
        "scoreCoverage": coverage,
        "summary": summary,
        "dimensions": dimensions,
        "comparison": {
            "category": bundle["public_name"],
            "currency": "CNY",
            "productCount": 1,
            "scope": "按基金类型使用适用公开资料；部分评分不作为不同覆盖率基金的综合排名。",
            "period": f"截至{bundle['as_of_date']}",
        },
        "strengths": [],
        "weaknesses": [],
        "dataDates": {
            "nav": bundle["as_of_date"],
            "holdings": bundle["holding_date"],
            "manager": bundle["as_of_date"],
            "fees": bundle["as_of_date"],
        },
        "limitations": [
            "部分评分仅对已确定维度折算；分数范围保留尚不确定部分，未取中点。",
            "评价公开披露的信息；不代表全部底层数据已齐全、逐日合规或未来赎回保证。",
            "部分专用方法和完整独立验证仍待完成；具体缺项见各维度说明。",
        ],
        "previousRating": None,
    }


def verify_result(result: RatingResult, batch: RatingBatch) -> None:
    config = batch.input_snapshot["method"]
    members = batch.input_snapshot["members"]
    member = next(m for m in members if m["fund_code"] == result.fund_code)
    bundle = member["bundle"]
    if config.get("approved_artifacts", {}).get(result.fund_code) != bundle["research_archive"]["sha256"]:
        raise ValueError("RATING_SNAPSHOT_REVIEW_SCOPE")
    if digest(bundle) != member["evidence_hash"]:
        raise ValueError("RATING_SNAPSHOT_BUNDLE_HASH")
    calculated = calculate(bundle)
    expected = {KEYS[k]: v["candidate"] for k, v in calculated["dimensions"].items()}
    if (
        result.status != calculated["status"]
        or result.score is not None
        or result.grade is not None
        or result.dimension_values != expected
        or result.public_evidence != public_evidence(bundle, calculated)
    ):
        raise ValueError("RATING_SNAPSHOT_RESULT_MISMATCH")


def publish_reviewed(session: Session, dossier: dict, reason: str, *, root: Path | None = None) -> dict:
    """同一事务追加目录、来源、方法、批次和当前指针；失败整体回滚。"""
    from app.repositories.fund_rating import lock_category, persist_input, publish

    if not reason.strip() or dossier.get("basis") != BASIS or not dossier.get("entries"):
        raise ValueError("RATING_SNAPSHOT_REVIEW_REQUIRED")
    root = (root or archive_root()).resolve()
    now = datetime.now(UTC)
    out = []
    codes = [e["fund_code"] for e in dossier["entries"]]
    if len(codes) != len(set(codes)) or len(codes) > 30:
        raise ValueError("RATING_SNAPSHOT_REVIEW_SCOPE")
    config = {
        "basis": BASIS,
        "revision": "reviewed-partial-20261023",
        "minimum_products": 1,
        "local_only": True,
        "independent_product_validation": False,
        "acceptance": "FROZEN_RESEARCH_SOURCE_ARITHMETIC_AND_EXPLICIT_USER_REVIEW",
        "approved_artifacts": {e["fund_code"]: e["research_archive"]["sha256"] for e in dossier["entries"]},
    }
    method_id = digest(config)
    if session.get(RatingMethodology, method_id) is None:
        session.add(
            RatingMethodology(
                methodology_id=method_id,
                family="MULTITYPE_REVIEWED",
                config=config,
                active=False,
                validation=None,
                created_at=now,
            )
        )
        session.flush()
    categories = []
    for entry in dossier["entries"]:
        code = entry["fund_code"]
        if not isinstance(code, str) or len(code) != 6 or not code.isascii() or not code.isdigit():
            raise ValueError("RATING_SNAPSHOT_CODE")
        bundle = deepcopy(entry)
        bundle["basis"] = BASIS
        doc = _read(entry["research_archive"], root)
        bundle["reviewed_document"] = _case(doc, code, entry["research_format"])
        if entry.get("money_input_archive"):
            bundle["money_inputs"] = _read(entry["money_input_archive"], root)
        validate_sources(bundle, now, root=root)
        computed = calculate(bundle)
        identity = entry["identity"]
        master_id = uuid5(NAMESPACE_URL, identity["manager_name"] + "/" + identity["legal_name"])
        category = digest({"basis": BASIS, "product_id": str(master_id), "share": code})
        categories.append({"category": category, "eligible": True})
        lock_category(session, category)
        master = session.scalar(
            select(FundMaster).where(
                FundMaster.manager_name == identity["manager_name"], FundMaster.fund_name == identity["legal_name"]
            )
        )
        if master is None:
            master = FundMaster(
                fund_master_id=master_id,
                fund_name=identity["legal_name"],
                manager_name=identity["manager_name"],
                fund_type=entry["family"],
                status="ACTIVE",
            )
            session.add(master)
            session.flush()
        share = session.get(FundShareClass, code)
        if share is None:
            share = FundShareClass(
                fund_code=code,
                fund_master_id=master.fund_master_id,
                share_class=entry["share_class"],
                fund_name=entry["public_name"],
                fund_type=entry["family"],
                status="ACTIVE",
                source_code="PUBLIC_DISCLOSURE",
                source_fund_code=code,
            )
            session.add(share)
            session.flush()
        elif share.fund_master_id != master.fund_master_id:
            raise ValueError("RATING_SNAPSHOT_EXISTING_PRODUCT_MISMATCH")
        cid = digest({"basis": BASIS, "code": code, "product": str(master.fund_master_id)})
        if session.get(RatingClassification, cid) is None:
            session.add(
                RatingClassification(
                    classification_id=cid,
                    fund_code=code,
                    product_id=str(master.fund_master_id),
                    category_code=category,
                    family=entry["family"],
                    currency="CNY",
                    eligibility="PUBLIC_LOCAL_ANALYSIS",
                    effective_from=date.fromisoformat(entry["as_of_date"]),
                    source_refs=[entry["original_report_archive"]["sha256"]],
                    verified_at=now,
                    revoked=False,
                )
            )
            session.flush()
        key = digest(bundle)
        if session.get(RatingEvidence, key) is None:
            session.add(
                RatingEvidence(
                    evidence_hash=key,
                    fund_code=code,
                    classification_id=cid,
                    as_of_date=date.fromisoformat(entry["as_of_date"]),
                    payload=bundle,
                    acquired_at=now,
                    revoked=False,
                )
            )
            session.flush()
        member = {"fund_code": code, "bundle": bundle, "evidence_hash": key}
        snapshot = {"method": config, "members": [member], "as_of_date": entry["as_of_date"]}
        batch_id, path = persist_input(root, snapshot)
        expiry = min(
            datetime.fromisoformat(entry["verified_at"]) + timedelta(days=31),
            datetime.combine(date.fromisoformat(entry["holding_date"]), datetime.min.time(), UTC) + timedelta(days=210),
        )
        if expiry <= now:
            raise ValueError("RATING_SNAPSHOT_EXPIRED")
        batch = session.get(RatingBatch, batch_id)
        if batch is None:
            batch = RatingBatch(
                batch_id=batch_id,
                category_code=category,
                methodology_id=method_id,
                as_of_date=date.fromisoformat(entry["as_of_date"]),
                frozen_at=now,
                input_hash=batch_id,
                sample_hash=digest([code]),
                input_path=path,
                input_snapshot=snapshot,
                member_count=1,
                product_count=1,
                status="COMPLETE",
                completed_at=now,
                withdrawn=False,
            )
            session.add(batch)
            session.flush()
            session.add(
                RatingMember(
                    batch_id=batch_id,
                    fund_code=code,
                    product_id=str(master.fund_master_id),
                    representative=True,
                    admitted=True,
                    reasons=[],
                    evidence_hash=key,
                )
            )
            session.add(
                RatingResult(
                    batch_id=batch_id,
                    fund_code=code,
                    status=computed["status"],
                    raw_score=computed["partial_score"],
                    score=None,
                    grade=None,
                    dimension_values={KEYS[k]: v["candidate"] for k, v in computed["dimensions"].items()},
                    public_evidence=public_evidence(bundle, computed),
                    valid_until=expiry,
                )
            )
            session.flush()
        out.append(
            {
                "fundCode": code,
                "batch": batch_id,
                "status": computed["status"],
                "partialScore": computed["partial_score"],
                "pointCoverage": computed["point_weight"],
                "completeScore": None,
                "grade": None,
            }
        )
    method = session.get(RatingMethodology, method_id)
    if method is None:
        # 方法必须先于引用它的批次存在；调用端先建受控空方法记录，再完成真实类别列表。
        raise ValueError("RATING_SNAPSHOT_METHOD_REQUIRED")
    method.validation = {
        "review": reason,
        "validated_at": now.isoformat(),
        "report": {"categories": categories},
        "independent_product_validation": False,
    }
    method.active = True
    session.flush()
    for row in out:
        publish(session, session.get(RatingBatch, row["batch"]))
    return {"methodology_id": method_id, "results": out, "independent_product_validation": False}
