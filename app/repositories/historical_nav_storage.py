"""学习样本的有界读写；只插入新批次，不更新旧批次，不负责提交事务。"""

from uuid import UUID

from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from app.models.historical_nav_sample import (
    HistoricalNavSampleBatch,
    HistoricalNavSampleLabel,
    HistoricalNavSampleRecord,
)


def find_batch_by_request_key(session: Session, request_key: UUID) -> HistoricalNavSampleBatch | None:
    """按唯一索引定位重试；在再次读取净值之前调用，保证来源更新不改变已保存结果。"""
    return session.scalar(select(HistoricalNavSampleBatch).where(HistoricalNavSampleBatch.request_key == request_key))


def get_batch(session: Session, batch_id: UUID) -> HistoricalNavSampleBatch | None:
    """按主键取封面；不做全历史扫描。"""
    return session.get(HistoricalNavSampleBatch, batch_id)


def read_sample_rows(
    session: Session,
    batch_id: UUID,
) -> list[tuple[HistoricalNavSampleRecord, HistoricalNavSampleLabel | None]]:
    """一条LEFT JOIN读取题目及可选答案；多取第32条用于发现非法超量，不产生N+1查询。"""
    return list(
        session.execute(
            select(HistoricalNavSampleRecord, HistoricalNavSampleLabel)
            .outerjoin(
                HistoricalNavSampleLabel, HistoricalNavSampleRecord.sample_id == HistoricalNavSampleLabel.sample_id
            )
            .where(HistoricalNavSampleRecord.batch_id == batch_id)
            .order_by(HistoricalNavSampleRecord.as_of_date)
            .limit(32)
        ).tuples()
    )


def insert_batch_rows(
    session: Session,
    batch: HistoricalNavSampleBatch,
    sample_rows: list[dict[str, object]],
    label_rows: list[dict[str, object]],
) -> None:
    """按外键顺序写三张表；任一步失败由调用者的事务整体回滚。

    先flush封面，数据库唯一约束仲裁并发的request_key；随后批量插入题目和答案。
    空范围只保存0条统计的封面；不伪造题目和答案，也不执行空的批量INSERT。
    """
    session.add(batch)
    session.flush()
    if sample_rows:
        # 显式写入None，避免不同空值组合把连续样本拆成逐行INSERT。
        session.execute(insert(HistoricalNavSampleRecord).execution_options(render_nulls=True), sample_rows)
    if label_rows:
        session.execute(insert(HistoricalNavSampleLabel), label_rows)
