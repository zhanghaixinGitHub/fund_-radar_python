"""按显式批次清单分组读取三表快照；不读原始净值、不更改数据。"""

from collections import defaultdict
from collections.abc import Iterator
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.historical_nav_sample import (
    HistoricalNavSampleBatch,
    HistoricalNavSampleLabel,
    HistoricalNavSampleRecord,
)

# 每组至多32个批次、992份合法样本；页数与基金历史长度无关。
BATCH_READ_PAGE_SIZE = 32


def iter_batch_rows(
    session: Session, batch_ids: tuple[UUID, ...]
) -> Iterator[tuple[HistoricalNavSampleBatch, list[tuple[HistoricalNavSampleRecord, HistoricalNavSampleLabel | None]]]]:
    """先有界取封面，再每组一次关联读明细；任一缺失批次由服务整体拒绝。"""
    batches = list(
        session.scalars(
            select(HistoricalNavSampleBatch)
            .where(HistoricalNavSampleBatch.batch_id.in_(batch_ids))
            .order_by(HistoricalNavSampleBatch.batch_id)
            .limit(len(batch_ids))
        )
    )
    if len(batches) != len(batch_ids):
        raise LookupError("one or more selected batches do not exist")
    for start in range(0, len(batches), BATCH_READ_PAGE_SIZE):
        page = batches[start : start + BATCH_READ_PAGE_SIZE]
        grouped = defaultdict(list)
        rows = session.execute(
            select(HistoricalNavSampleRecord, HistoricalNavSampleLabel)
            .outerjoin(
                HistoricalNavSampleLabel, HistoricalNavSampleRecord.sample_id == HistoricalNavSampleLabel.sample_id
            )
            .where(HistoricalNavSampleRecord.batch_id.in_([b.batch_id for b in page]))
            .order_by(HistoricalNavSampleRecord.batch_id, HistoricalNavSampleRecord.as_of_date)
            .limit(len(page) * 31 + 1)
        ).tuples()
        for sample, label in rows:
            grouped[sample.batch_id].append((sample, label))
        for batch in page:
            yield batch, grouped[batch.batch_id]
