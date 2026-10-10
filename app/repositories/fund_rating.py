"""评级快照持久化和一致性读取；发布、撤回、回退均由数据库锁保护。"""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models.fund_rating import (
    RatingAudit,
    RatingBatch,
    RatingClassification,
    RatingCurrent,
    RatingEvidence,
    RatingMember,
    RatingMethodology,
    RatingResult,
)
from app.services.fund_rating_inputs import digest
from app.services.fund_rating_rules import DIMENSIONS, GRADES, grade


class RatingConflict(RuntimeError):
    pass


def lock_category(session: Session, category: str) -> None:
    if not session.scalar(
        text("SELECT pg_try_advisory_xact_lock(hashtextextended(:key,0))"), {"key": "fund-rating:" + category}
    ):
        raise RatingConflict("RATING_CATEGORY_BUSY")


def verify_batch(session: Session, batch: RatingBatch) -> list[RatingResult]:
    """哈希、数量、八维及映射核验；损坏抛故障，绝不转换为业务资料不足。"""
    if digest(batch.input_snapshot) != batch.input_hash:
        raise ValueError("RATING_BATCH_HASH_MISMATCH")
    members = session.scalars(select(RatingMember).where(RatingMember.batch_id == batch.batch_id)).all()
    results = session.scalars(select(RatingResult).where(RatingResult.batch_id == batch.batch_id)).all()
    if len(members) != batch.member_count or {m.fund_code for m in members} != {r.fund_code for r in results}:
        raise ValueError("RATING_BATCH_INCOMPLETE")
    samples = sorted(m.fund_code for m in members if m.representative)
    if len(samples) != batch.product_count or digest(samples) != batch.sample_hash:
        raise ValueError("RATING_SAMPLE_HASH_MISMATCH")
    for result in results:
        if result.status == "RATED":
            if not result.dimension_values or set(result.dimension_values) != set(DIMENSIONS):
                raise ValueError("RATING_EIGHT_DIMENSIONS_REQUIRED")
            if result.grade != grade(result.score) or not result.valid_until:
                raise ValueError("RATING_RESULT_INVALID")
            for value in result.dimension_values.values():
                grade(value)
    return results


def inputs_valid(session: Session, batch: RatingBatch) -> bool:
    members = session.scalars(
        select(RatingMember).where(RatingMember.batch_id == batch.batch_id, RatingMember.evidence_hash.is_not(None))
    ).all()
    keys = [m.evidence_hash for m in members]
    if not keys:
        return True
    rows = session.execute(
        select(RatingEvidence, RatingClassification)
        .join(RatingClassification, RatingEvidence.classification_id == RatingClassification.classification_id)
        .where(RatingEvidence.evidence_hash.in_(keys))
    ).all()
    if len(rows) != len(set(keys)):
        return False
    if not all(not e.revoked and not c.revoked and digest(e.payload) == e.evidence_hash for e, c in rows):
        return False
    codes = {ref["source_code"] for e, _ in rows for ref in e.payload["sources"]}
    enabled = set(
        session.execute(
            text(
                "SELECT source_code FROM source_registry WHERE enabled AND "
                "authorization_verified_at IS NOT NULL AND source_code=ANY(:codes)"
            ),
            {"codes": list(codes)},
        ).scalars()
    )
    return codes == enabled


def publish(session: Session, batch: RatingBatch, *, rollback_reason: str | None = None) -> None:
    """完整结果先入库再切指针；正常更新禁止日期倒退，回退只允许仍有效且未撤回的批次。"""
    lock_category(session, batch.category_code)
    results = verify_batch(session, batch)
    if batch.withdrawn or not inputs_valid(session, batch):
        raise ValueError("RATING_BATCH_WITHDRAWN")
    if any(r.status == "RATED" for r in results):
        method = session.get(RatingMethodology, batch.methodology_id)
        if not method or not method.active or not method.validation or digest(method.config) != method.methodology_id:
            raise ValueError("RATING_METHOD_NOT_VALIDATED")
        allowed = {r["category"] for r in method.validation.get("report", {}).get("categories", []) if r["eligible"]}
        if batch.category_code not in allowed:
            raise ValueError("RATING_CATEGORY_NOT_VALIDATED")
    current = session.get(RatingCurrent, batch.category_code, with_for_update=True)
    now = datetime.now(UTC)
    if current:
        old = session.get(RatingBatch, current.batch_id)
        if old.as_of_date > batch.as_of_date and not rollback_reason:
            raise RatingConflict("RATING_OLDER_BATCH")
    if rollback_reason and any(r.status == "RATED" and r.valid_until <= now for r in results):
        raise ValueError("RATING_ROLLBACK_EXPIRED")
    if not current:
        current = RatingCurrent(category_code=batch.category_code, batch_id=batch.batch_id, updated_at=now)
        session.add(current)
    current.batch_id, current.updated_at = batch.batch_id, now
    batch.status = "PUBLISHED"
    session.add(
        RatingAudit(
            audit_id=uuid4().hex,
            category_code=batch.category_code,
            batch_id=batch.batch_id,
            action="ROLLBACK" if rollback_reason else "PUBLISH",
            reason=rollback_reason or "完整类别快照校验通过",
            created_at=now,
        )
    )
    session.flush()


def persist_input(root: Path, snapshot: dict) -> tuple[str, str]:
    from app.services.fund_exposure_common import read, save

    key = digest(snapshot)
    relative = f"inputs/{key}.json"
    path = root / relative
    try:
        save(path, snapshot)
    except FileExistsError:
        if digest(read(path)) != key:
            raise ValueError("RATING_INPUT_FILE_CORRUPT") from None
    return key, relative


def withdraw(session: Session, batch_id: str, reason: str) -> None:
    if not reason.strip():
        raise ValueError("RATING_WITHDRAW_REASON_REQUIRED")
    batch = session.get(RatingBatch, batch_id)
    if not batch:
        raise LookupError("RATING_BATCH_NOT_FOUND")
    lock_category(session, batch.category_code)
    batch.withdrawn = True
    session.add(
        RatingAudit(
            audit_id=uuid4().hex,
            category_code=batch.category_code,
            batch_id=batch_id,
            action="WITHDRAW",
            reason=reason,
            created_at=datetime.now(UTC),
        )
    )


def summary(result: RatingResult, batch: RatingBatch, *, invalid: bool, now: datetime) -> dict:
    status = result.status
    if invalid:
        status = "WITHDRAWN"
    elif status == "RATED" and result.valid_until <= now:
        status = "STALE"
    rated = status == "RATED"
    return {
        "fundCode": result.fund_code,
        "status": status,
        "grade": result.grade if rated else None,
        "gradeLabel": GRADES[result.grade] if rated else None,
        "asOfDate": str(batch.as_of_date),
        "validUntil": result.valid_until.isoformat() if result.valid_until else None,
        "ratingRef": batch.batch_id,
        "message": None if rated else "评级待更新" if status in {"STALE", "WITHDRAWN"} else "暂未评级",
    }
