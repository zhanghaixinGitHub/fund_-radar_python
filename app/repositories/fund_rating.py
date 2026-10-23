"""评级快照持久化和一致性读取；发布、撤回、回退均由数据库锁保护。"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.orm import Session, load_only

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


def reviewed_current_bases(session: Session, codes: list[str]) -> dict[str, str]:
    """已受控启用的基金保持方法口径；旧采集器或缺新资料不能触发自动降回旧规则。"""
    from app.services.fund_rating_public_policy import REVIEWED_BASES, SNAPSHOT_BASIS

    rows = session.execute(
        select(RatingResult.fund_code, RatingMethodology.config)
        .select_from(RatingCurrent)
        .join(RatingBatch, RatingBatch.batch_id == RatingCurrent.batch_id)
        .join(RatingResult, RatingResult.batch_id == RatingBatch.batch_id)
        .join(RatingMethodology, RatingMethodology.methodology_id == RatingBatch.methodology_id)
        .where(RatingResult.fund_code.in_(codes))
    )
    result = {}
    for code, config in rows:
        basis = config.get("basis")
        if basis in REVIEWED_BASES or basis == SNAPSHOT_BASIS:
            if code in result and result[code] != basis:
                raise ValueError("RATING_REVIEWED_CURRENT_CONFLICT")
            result[code] = basis
    return result


@dataclass
class RatingValidationRows:
    """单次只读事务共用的核验数据；不跨请求缓存，撤回及授权变化仍在本次快照内检查。"""

    members: dict[str, list[RatingMember]]
    results: dict[str, list[RatingResult]]
    evidence: dict[str, tuple[RatingEvidence, RatingClassification]]
    enabled_sources: set[str]


def load_validation_rows(session: Session, batches: list[RatingBatch]) -> RatingValidationRows:
    """按本次涉及的批次一次取齐核验行，避免每个类别重复查询及加载八维展示正文。

    完整批次成员、得分映射、输入哈希和来源资格仍全部核验。评级正文只在详情真正
    使用时由 ORM 读取；列表不把完整披露依据搬进内存，也不会缓存核验结论。
    """
    context = RatingValidationRows({}, {}, {}, set())
    batch_ids = [batch.batch_id for batch in batches]
    if not batch_ids:
        return context
    members = session.scalars(select(RatingMember).where(RatingMember.batch_id.in_(batch_ids))).all()
    for member in members:
        context.members.setdefault(member.batch_id, []).append(member)
    results = session.scalars(
        select(RatingResult)
        .where(RatingResult.batch_id.in_(batch_ids))
        .options(
            load_only(
                RatingResult.batch_id,
                RatingResult.fund_code,
                RatingResult.status,
                RatingResult.score,
                RatingResult.grade,
                RatingResult.dimension_values,
                RatingResult.valid_until,
            )
        )
    ).all()
    for result in results:
        context.results.setdefault(result.batch_id, []).append(result)
    hashes = {member.evidence_hash for member in members if member.evidence_hash is not None}
    if hashes:
        rows = session.execute(
            select(RatingEvidence, RatingClassification)
            .join(RatingClassification, RatingEvidence.classification_id == RatingClassification.classification_id)
            .where(RatingEvidence.evidence_hash.in_(hashes))
        ).all()
        context.evidence = {e.evidence_hash: (e, classification) for e, classification in rows}
    from app.services.fund_rating_public_policy import PUBLIC_BASES, REVIEWED_BASES

    codes = {
        ref["source_code"]
        for evidence, _ in context.evidence.values()
        if evidence.payload.get("basis") not in PUBLIC_BASES or evidence.payload.get("basis") in REVIEWED_BASES
        for ref in evidence.payload["sources"]
    }
    if codes:
        context.enabled_sources = set(
            session.execute(
                text(
                    "SELECT source_code FROM source_registry WHERE enabled AND "
                    "authorization_verified_at IS NOT NULL AND source_code=ANY(:codes)"
                ),
                {"codes": list(codes)},
            ).scalars()
        )
    return context


def verify_batch(
    session: Session, batch: RatingBatch, *, context: RatingValidationRows | None = None
) -> list[RatingResult]:
    """哈希、数量、八维及映射核验；损坏抛故障，绝不转换为业务资料不足。"""
    if digest(batch.input_snapshot) != batch.input_hash:
        raise ValueError("RATING_BATCH_HASH_MISMATCH")
    members = (
        context.members.get(batch.batch_id, [])
        if context is not None
        else session.scalars(select(RatingMember).where(RatingMember.batch_id == batch.batch_id)).all()
    )
    results = (
        context.results.get(batch.batch_id, [])
        if context is not None
        else session.scalars(select(RatingResult).where(RatingResult.batch_id == batch.batch_id)).all()
    )
    if len(members) != batch.member_count or {m.fund_code for m in members} != {r.fund_code for r in results}:
        raise ValueError("RATING_BATCH_INCOMPLETE")
    samples = sorted(m.fund_code for m in members if m.representative)
    if len(samples) != batch.product_count or digest(samples) != batch.sample_hash:
        raise ValueError("RATING_SAMPLE_HASH_MISMATCH")
    for result in results:
        from app.services.fund_rating_reviewed_snapshot import BASIS as SNAPSHOT_BASIS
        from app.services.fund_rating_reviewed_snapshot import verify_result

        if (batch.input_snapshot.get("method") or {}).get("basis") == SNAPSHOT_BASIS:
            verify_result(result, batch)
        if result.status == "RATED":
            if not result.dimension_values or set(result.dimension_values) != set(DIMENSIONS):
                raise ValueError("RATING_EIGHT_DIMENSIONS_REQUIRED")
            from app.services.fund_rating_public_policy import REVIEWED_BASES

            config = batch.input_snapshot.get("method") or {}
            expected_grade = grade(result.score)
            if config.get("basis") in REVIEWED_BASES:
                from app.services.fund_rating_reviewed_quality import calculate

                frozen = next(m for m in batch.input_snapshot["members"] if m["fund_code"] == result.fund_code)
                if (
                    digest(frozen["bundle"]) != frozen["evidence_hash"]
                    or frozen["metrics"] != frozen["bundle"]["quality_metrics"]
                ):
                    raise ValueError("RATING_QUALITY_FROZEN_OBSERVATIONS_MISMATCH")
                rebuilt = calculate(frozen["metrics"], config=config)
                if str(result.score) != rebuilt["score"] or result.dimension_values != rebuilt["dimensions"]:
                    raise ValueError("RATING_QUALITY_RESULT_MISMATCH")
                expected_grade = rebuilt["grade"]
            if result.grade != expected_grade or not result.valid_until:
                raise ValueError("RATING_RESULT_INVALID")
            for value in result.dimension_values.values():
                grade(value)
    return results


def inputs_valid(session: Session, batch: RatingBatch, *, context: RatingValidationRows | None = None) -> bool:
    members = (
        context.members.get(batch.batch_id, [])
        if context is not None
        else session.scalars(
            select(RatingMember).where(RatingMember.batch_id == batch.batch_id, RatingMember.evidence_hash.is_not(None))
        ).all()
    )
    keys = [m.evidence_hash for m in members if m.evidence_hash is not None]
    if not keys:
        return True
    rows = (
        [context.evidence[key] for key in set(keys) if key in context.evidence]
        if context is not None
        else session.execute(
            select(RatingEvidence, RatingClassification)
            .join(RatingClassification, RatingEvidence.classification_id == RatingClassification.classification_id)
            .where(RatingEvidence.evidence_hash.in_(keys))
        ).all()
    )
    if len(rows) != len(set(keys)):
        return False
    if not all(not e.revoked and not c.revoked and digest(e.payload) == e.evidence_hash for e, c in rows):
        return False
    from app.services.fund_rating_public_inputs import validate_public_sources
    from app.services.fund_rating_public_policy import PUBLIC_BASES, REVIEWED_BASES

    legacy_rows = []
    for e, c in rows:
        if e.payload.get("basis") in PUBLIC_BASES:
            try:
                validate_public_sources(e.payload, datetime.now(UTC))
            except ValueError:
                return False
            if e.payload.get("basis") in REVIEWED_BASES:
                # 新口径仍依赖已授权供应商；撤销资格立即影响读取，不因属于公开资料分支绕过。
                legacy_rows.append((e, c))
        else:
            legacy_rows.append((e, c))
    codes = {ref["source_code"] for e, _ in legacy_rows for ref in e.payload["sources"]}
    if not codes:
        return True
    if context is not None:
        return codes <= context.enabled_sources
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
    codes = sorted(r.fund_code for r in results)
    for code in codes:
        if not session.scalar(
            text("SELECT pg_try_advisory_xact_lock(hashtextextended(:key,0))"), {"key": "rating-fund-method:" + code}
        ):
            raise RatingConflict("RATING_FUND_METHOD_BUSY")
    pinned = reviewed_current_bases(session, codes)
    basis = (batch.input_snapshot.get("method") or {}).get("basis")
    if not rollback_reason and any(value != basis for value in pinned.values()):
        raise ValueError("RATING_REVIEWED_INPUT_REQUIRED")
    if batch.withdrawn or not inputs_valid(session, batch):
        raise ValueError("RATING_BATCH_WITHDRAWN")
    if any(r.status in {"RATED", "PARTIAL"} for r in results) or basis == "PUBLIC_REVIEWED_QUALITY_SNAPSHOT_V3":
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
    if rollback_reason and any(r.status in {"RATED", "PARTIAL"} and r.valid_until <= now for r in results):
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


def rollback_switch(session: Session, from_batch_id: str, to_batch_id: str, reason: str) -> None:
    """跨口径回退：只撤销明确指定的当前指针，历史批次及原件均保留。

    新类别必须完全包含在恢复批次内，避免恢复单只基金时影响同批其他基金。
    指针检查、旧批次有效性核验及恢复处于调用者的同一个事务中。
    """
    if not reason.strip() or from_batch_id == to_batch_id:
        raise ValueError("RATING_SWITCH_ROLLBACK_ARGUMENTS")
    source, target = session.get(RatingBatch, from_batch_id), session.get(RatingBatch, to_batch_id)
    if source is None or target is None:
        raise LookupError("RATING_BATCH_NOT_FOUND")
    for category in sorted({source.category_code, target.category_code}):
        lock_category(session, category)
    pointer = session.get(RatingCurrent, source.category_code, with_for_update=True)
    if pointer is None or pointer.batch_id != from_batch_id:
        raise RatingConflict("RATING_SWITCH_CURRENT_CHANGED")
    source_codes = {r.fund_code for r in verify_batch(session, source)}
    target_codes = {r.fund_code for r in verify_batch(session, target)}
    if not source_codes or not source_codes <= target_codes:
        raise ValueError("RATING_SWITCH_ROLLBACK_SCOPE")
    if source.category_code != target.category_code:
        target_pointer = session.get(RatingCurrent, target.category_code, with_for_update=True)
        if target_pointer is not None and target_pointer.batch_id != to_batch_id:
            raise RatingConflict("RATING_SWITCH_ROLLBACK_TARGET_CHANGED")
        session.delete(pointer)
        session.add(
            RatingAudit(
                audit_id=uuid4().hex,
                category_code=source.category_code,
                batch_id=from_batch_id,
                action="ROLLBACK_SWITCH",
                reason=f"恢复批次{to_batch_id}：{reason}",
                created_at=datetime.now(UTC),
            )
        )
    publish(session, target, rollback_reason=reason)


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
    elif (
        (status in {"RATED", "PARTIAL"} or result.public_evidence.get("scoreCoverage") is not None)
        and result.valid_until and result.valid_until <= now
    ):
        status = "STALE"
    rated = status == "RATED"
    output = {
        "fundCode": result.fund_code,
        "status": status,
        "grade": result.grade if rated else None,
        "gradeLabel": GRADES[result.grade] if rated else None,
        "asOfDate": str(batch.as_of_date),
        "validUntil": result.valid_until.isoformat() if result.valid_until else None,
        "ratingRef": batch.batch_id,
        "message": (
            None if rated else "评级待更新" if status in {"STALE", "WITHDRAWN"}
            else "部分维度已评分" if status == "PARTIAL" else "暂未评级"
        ),
    }
    if status == "PARTIAL":
        for key in ("scoreRange", "partialScore", "scoreCoverage"):
            output[key] = result.public_evidence.get(key)
    return output
