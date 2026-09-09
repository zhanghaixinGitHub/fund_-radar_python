"""有界分页读明确批次；模型报告独立保存，不访问旧样本或原净值。"""

from collections import defaultdict
from datetime import date

from sqlalchemy import select

from app.models.cash_reinvestment import CashResearchRun, CashSampleBatch, CashSampleLabel, CashSampleRecord
from app.services.historical_nav_storage import HistoricalNavStorageError


def _check_page_dates(session, batches):
    """读取任何X/y正文前，先有界核对批次、题目和标签日期；绝不靠载入答案后才拒绝2025。"""
    lower, upper = date(2022, 1, 1), date(2024, 12, 31)
    by_id = {batch.batch_id: batch for batch in batches}
    if any(batch.end_date > upper for batch in batches):
        raise HistoricalNavStorageError("TEST_PERIOD_PROTECTED", "所选批次进入2025，未读取样本或答案正文。", 409)
    if any(
        not lower <= batch.start_date <= batch.end_date or (batch.end_date - batch.start_date).days >= 31
        for batch in batches
    ):
        raise HistoricalNavStorageError("CASH_BATCH_CORRUPTED", "批次日期范围不符合现金研究协议。", 503)
    dates = session.execute(
        select(
            CashSampleRecord.batch_id,
            CashSampleRecord.cutoff_date,
            CashSampleLabel.sample_id,
            CashSampleLabel.label_payload["label_end_date"].as_string(),
            CashSampleLabel.label_payload["label_available_at"].as_string(),
        )
        .outerjoin(CashSampleLabel, CashSampleRecord.sample_id == CashSampleLabel.sample_id)
        .where(CashSampleRecord.batch_id.in_(tuple(by_id)))
        .order_by(CashSampleRecord.batch_id, CashSampleRecord.cutoff_date)
        .limit(8 * 31 + 1)
    ).all()
    if len(dates) > 8 * 31:
        raise HistoricalNavStorageError("CASH_BATCH_CORRUPTED", "批次题目数超出上限。", 503)
    for batch_id, cutoff, label_id, end_text, available_text in dates:
        batch = by_id[batch_id]
        if cutoff > upper:
            raise HistoricalNavStorageError("TEST_PERIOD_PROTECTED", "题目进入2025，未读取输入或答案正文。", 409)
        if not batch.start_date <= cutoff <= batch.end_date:
            raise HistoricalNavStorageError("CASH_BATCH_CORRUPTED", "题目日期不在所属批次范围。", 503)
        if label_id is None:
            continue
        try:
            end, available = date.fromisoformat(end_text), date.fromisoformat(available_text)
            if (end.isoformat(), available.isoformat()) != (end_text, available_text) or not cutoff < end <= available:
                raise ValueError("label dates inconsistent")
        except (TypeError, ValueError) as error:
            raise HistoricalNavStorageError(
                "CASH_BATCH_CORRUPTED", "标签日期无法核验，未读取答案正文。", 503
            ) from error
        if available > upper:
            raise HistoricalNavStorageError("TEST_PERIOD_PROTECTED", "标签进入2025，未读取答案正文。", 409)


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
        _check_page_dates(session, batches)
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
