"""只操作独立现金研究表；事务由服务层持有，仓储不提交、不覆盖。"""

from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from app.models.cash_reinvestment import CashSampleBatch, CashSampleLabel, CashSampleRecord


def find_batch(session: Session, *, batch_id=None, request_key=None):
    condition = (
        CashSampleBatch.batch_id == batch_id if batch_id is not None else CashSampleBatch.request_key == request_key
    )
    return session.scalar(select(CashSampleBatch).where(condition))


def read_rows(session: Session, batch_id):
    return session.execute(
        select(CashSampleRecord, CashSampleLabel)
        .outerjoin(CashSampleLabel, CashSampleRecord.sample_id == CashSampleLabel.sample_id)
        .where(CashSampleRecord.batch_id == batch_id)
        .order_by(CashSampleRecord.cutoff_date)
        .limit(32)  # 多读一条侦测损坏，不能无界导出。
    ).all()


def insert_rows(session: Session, batch, samples, labels):
    session.add(batch)
    session.flush()
    if samples:
        session.execute(insert(CashSampleRecord).execution_options(render_nulls=True), samples)
    if labels:
        session.execute(insert(CashSampleLabel), labels)
