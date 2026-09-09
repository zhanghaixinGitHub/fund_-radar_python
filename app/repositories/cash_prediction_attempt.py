"""回执仅按主键或唯一重试凭证等值读取；不扫描净值、样本或旧预测表。"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.cash_prediction_attempt import CashPredictionAttemptRecord


def find_attempt(
    session: Session, *, attempt_id: UUID | None = None, request_key: UUID | None = None
) -> CashPredictionAttemptRecord | None:
    if (attempt_id is None) == (request_key is None):
        raise ValueError("exactly one attempt lookup key is required")
    clause = (
        CashPredictionAttemptRecord.attempt_id == attempt_id
        if attempt_id is not None
        else CashPredictionAttemptRecord.request_key == request_key
    )
    return session.scalar(select(CashPredictionAttemptRecord).where(clause))
