"""本地观察日志的元数据聚合；没有查询源值/标签正文，也不把日志作为正式准入授权。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.models.cash_source_observation import CashSourceObservation as Observation
from app.schemas.cash_source_observation import CashObservationCheckRequest


def observation_counts(
    session: Session,
    *,
    source_id: UUID,
    request: CashObservationCheckRequest,
    checked_at: datetime,
    cutoff_end: datetime,
):
    active = Observation.expires_at > checked_at
    return (
        session.execute(
            select(
                func.count().filter(active).label("active_record_count"),
                func.count().filter(active, Observation.observed_at < cutoff_end).label("recorded_before_cutoff_count"),
                func.count().filter(~active).label("expired_record_count"),
                func.min(Observation.observed_at).filter(active).label("first_observed_at"),
            ).where(
                Observation.source_id == source_id,
                Observation.fund_code == request.fund_code,
                Observation.event_date.between(request.start_date, request.end_date),
            )
        )
        .mappings()
        .one()
    )


def expired_observation_ids(session: Session, *, limit: int, for_update: bool = False) -> tuple[int, ...]:
    """仅选择已到期的本表记录，单批最多1000条；执行清理时跳过其他事务锁住的行。"""
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("expired observation batch limit must be 1..1000")
    query = (
        select(Observation.observation_id)
        .where(Observation.expires_at <= func.clock_timestamp())
        .order_by(Observation.expires_at, Observation.observation_id)
        .limit(limit)
    )
    if for_update:
        query = query.with_for_update(skip_locked=True)
    return tuple(session.scalars(query))


def purge_expired_observations(session: Session, *, limit: int = 1000) -> int:
    """仅由显式维护命令调用；不清源表/样本/模型，数据库触发器再次拒绝删除未到期记录。"""
    ids = expired_observation_ids(session, limit=limit, for_update=True)
    if not ids:
        return 0
    deleted = session.scalars(
        delete(Observation)
        .where(Observation.observation_id.in_(ids), Observation.expires_at <= func.clock_timestamp())
        .returning(Observation.observation_id)
    ).all()
    return len(deleted)
