"""有界分页读明确批次；模型报告独立保存，不访问旧样本或原净值。"""

from collections import defaultdict

from sqlalchemy import select

from app.models.cash_reinvestment import CashResearchRun, CashSampleBatch, CashSampleLabel, CashSampleRecord
from app.services.historical_nav_storage import HistoricalNavStorageError


def iter_cash_batches(session, batch_ids):
    for offset in range(0, len(batch_ids), 8):
        page = batch_ids[offset : offset + 8]
        batches = session.scalars(
            select(CashSampleBatch)
            .where(CashSampleBatch.batch_id.in_(page))
            .order_by(CashSampleBatch.batch_id)
            .limit(8)
        ).all()
        if len(batches) != len(page):
            raise HistoricalNavStorageError("CASH_BATCH_NOT_FOUND", "所选批次有不存在或非现金版本的编号。", 404)
        rows = session.execute(
            select(CashSampleRecord, CashSampleLabel)
            .outerjoin(CashSampleLabel, CashSampleRecord.sample_id == CashSampleLabel.sample_id)
            .where(CashSampleRecord.batch_id.in_(page))
            .order_by(CashSampleRecord.batch_id, CashSampleRecord.cutoff_date)
            .limit(8 * 31 + 1)
        ).all()
        if len(rows) > 8 * 31:
            raise HistoricalNavStorageError("CASH_BATCH_CORRUPTED", "批次题目数超出上限。", 503)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[0].batch_id].append(row)
        for batch in batches:
            yield batch, grouped[batch.batch_id]


def find_research(session, *, request_key=None, run_id=None):
    column, value = (CashResearchRun.request_key, request_key) if request_key else (CashResearchRun.run_id, run_id)
    return session.scalar(select(CashResearchRun).where(column == value))
