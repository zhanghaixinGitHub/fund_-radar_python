"""免费场内基金与市场参考数据的幂等写入、水位和查询。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.benchmark import BenchmarkSeries
from app.models.fund import FundShareClass, SourceRegistry, SourceSyncCursor
from app.models.market_reference import (
    FundExchangeDaily,
    IndexWeightSnapshot,
    MarketIndexCatalog,
    MarketIndexClassification,
)
from app.repositories.fund_sync import TUSHARE_SOURCE_CODE, WriteStats


class SourceCapabilityError(ValueError):
    """来源未启用或未登记所需接口能力时拒绝外部调用。"""


@dataclass(frozen=True)
class FundExchangeTarget:
    """已有精确来源交易代码的场内基金同步目标。"""

    fund_code: str
    source_fund_code: str


@dataclass(frozen=True)
class FundExchangeDailyUpsert:
    """一条已规范化、哈希化的场内基金日线记录。"""

    fund_code: str
    trade_date: date
    open_price: Decimal | None
    high_price: Decimal | None
    low_price: Decimal | None
    close_price: Decimal
    previous_close_price: Decimal | None
    change_value: Decimal | None
    change_percent: Decimal | None
    volume: Decimal | None
    amount: Decimal | None
    content_hash: str


@dataclass(frozen=True)
class MarketIndexCatalogUpsert:
    """一条来源指数目录记录。"""

    index_code: str
    display_name: str
    market: str | None
    publisher: str | None
    category: str | None
    base_date: date | None
    list_date: date | None
    expiry_date: date | None
    row_hash: str


@dataclass(frozen=True)
class MarketIndexClassificationUpsert:
    """一条来源指数分类记录。"""

    classification_code: str
    classification_name: str
    parent_classification_code: str | None
    hierarchy_level: int | None
    source_name: str | None
    row_hash: str


@dataclass(frozen=True)
class IndexWeightSnapshotUpsert:
    """一条来源指数成分权重快照。"""

    index_code: str
    weight_date: date
    constituent_code: str
    weight: Decimal
    row_hash: str


def require_tushare_source_capabilities(session: Session, api_names: tuple[str, ...]) -> SourceRegistry:
    """确认 Tushare 来源已启用且所有指定接口均已做过最小验权。"""
    source = session.scalar(select(SourceRegistry).where(SourceRegistry.source_code == TUSHARE_SOURCE_CODE))
    if source is None or not source.enabled:
        raise SourceCapabilityError("TUSHARE_SOURCE_DISABLED")
    authorized_names = set(source.authorized_api_names)
    missing_names = tuple(api_name for api_name in api_names if api_name not in authorized_names)
    if missing_names:
        raise SourceCapabilityError(f"TUSHARE_API_NOT_AUTHORIZED:{','.join(missing_names)}")
    return source


def list_active_exchange_targets(session: Session) -> tuple[FundExchangeTarget, ...]:
    """返回已有明确 .SH/.SZ 来源代码的启用基金，不猜测后缀。"""
    rows = session.execute(
        select(FundShareClass.fund_code, FundShareClass.source_fund_code)
        .where(
            FundShareClass.source_code == TUSHARE_SOURCE_CODE,
            FundShareClass.status == "ACTIVE",
            FundShareClass.source_fund_code.is_not(None),
        )
        .order_by(FundShareClass.source_fund_code.asc())
    ).all()
    return tuple(
        FundExchangeTarget(fund_code=fund_code, source_fund_code=source_fund_code)
        for fund_code, source_fund_code in rows
        if source_fund_code.endswith((".SH", ".SZ"))
    )


def list_reference_benchmark_codes(session: Session, *, source_id: UUID) -> tuple[str, ...]:
    """返回已登记的市场参考指数；DRAFT 可采集但不可用于模型激活。"""
    return tuple(
        session.scalars(
            select(BenchmarkSeries.benchmark_code)
            .where(
                BenchmarkSeries.source_id == source_id,
                BenchmarkSeries.status.in_(("DRAFT", "ACTIVE")),
            )
            .order_by(BenchmarkSeries.benchmark_code.asc())
        ).all()
    )


def upsert_fund_exchange_daily_batch(
    session: Session, *, source_id: UUID, records: tuple[FundExchangeDailyUpsert, ...]
) -> WriteStats:
    """按基金、交易日和来源幂等写入场内基金行情。"""
    if not records:
        return WriteStats()
    fund_codes = tuple({record.fund_code for record in records})
    dates = tuple({record.trade_date for record in records})
    existing_by_key = {
        (row.fund_code, row.trade_date): row
        for row in session.scalars(
            select(FundExchangeDaily).where(
                FundExchangeDaily.source_id == source_id,
                FundExchangeDaily.fund_code.in_(fund_codes),
                FundExchangeDaily.trade_date.in_(dates),
            )
        ).all()
    }
    created_count = 0
    updated_count = 0
    skipped_count = 0
    for record in records:
        key = (record.fund_code, record.trade_date)
        existing = existing_by_key.get(key)
        if existing is None:
            session.add(
                FundExchangeDaily(
                    fund_code=record.fund_code,
                    trade_date=record.trade_date,
                    source_id=source_id,
                    open_price=record.open_price,
                    high_price=record.high_price,
                    low_price=record.low_price,
                    close_price=record.close_price,
                    previous_close_price=record.previous_close_price,
                    change_value=record.change_value,
                    change_percent=record.change_percent,
                    volume=record.volume,
                    amount=record.amount,
                    content_hash=record.content_hash,
                )
            )
            created_count += 1
            continue
        if existing.content_hash == record.content_hash:
            skipped_count += 1
            continue
        for field_name in (
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "previous_close_price",
            "change_value",
            "change_percent",
            "volume",
            "amount",
            "content_hash",
        ):
            setattr(existing, field_name, getattr(record, field_name))
        updated_count += 1
    return WriteStats(created_count=created_count, updated_count=updated_count, skipped_count=skipped_count)


def upsert_market_index_catalog_batch(
    session: Session, *, source_id: UUID, records: tuple[MarketIndexCatalogUpsert, ...]
) -> WriteStats:
    """按来源和指数代码幂等写入目录，不自动创建模型基准。"""
    if not records:
        return WriteStats()
    codes = tuple({record.index_code for record in records})
    existing_by_code = {
        row.index_code: row
        for row in session.scalars(
            select(MarketIndexCatalog).where(
                MarketIndexCatalog.source_id == source_id,
                MarketIndexCatalog.index_code.in_(codes),
            )
        ).all()
    }
    created_count = 0
    updated_count = 0
    skipped_count = 0
    for record in records:
        existing = existing_by_code.get(record.index_code)
        if existing is None:
            session.add(MarketIndexCatalog(source_id=source_id, **record.__dict__))
            created_count += 1
            continue
        if existing.row_hash == record.row_hash:
            skipped_count += 1
            continue
        for field_name in record.__dataclass_fields__:
            setattr(existing, field_name, getattr(record, field_name))
        updated_count += 1
    return WriteStats(created_count=created_count, updated_count=updated_count, skipped_count=skipped_count)


def upsert_market_index_classifications_batch(
    session: Session, *, source_id: UUID, records: tuple[MarketIndexClassificationUpsert, ...]
) -> WriteStats:
    """按来源和分类编码幂等写入指数分类层级。"""
    if not records:
        return WriteStats()
    codes = tuple({record.classification_code for record in records})
    existing_by_code = {
        row.classification_code: row
        for row in session.scalars(
            select(MarketIndexClassification).where(
                MarketIndexClassification.source_id == source_id,
                MarketIndexClassification.classification_code.in_(codes),
            )
        ).all()
    }
    created_count = 0
    updated_count = 0
    skipped_count = 0
    for record in records:
        existing = existing_by_code.get(record.classification_code)
        if existing is None:
            session.add(MarketIndexClassification(source_id=source_id, **record.__dict__))
            created_count += 1
            continue
        if existing.row_hash == record.row_hash:
            skipped_count += 1
            continue
        for field_name in record.__dataclass_fields__:
            setattr(existing, field_name, getattr(record, field_name))
        updated_count += 1
    return WriteStats(created_count=created_count, updated_count=updated_count, skipped_count=skipped_count)


def upsert_index_weight_snapshots_batch(
    session: Session, *, source_id: UUID, records: tuple[IndexWeightSnapshotUpsert, ...]
) -> WriteStats:
    """按来源、指数、日期和成分股幂等写入权重快照。"""
    if not records:
        return WriteStats()
    index_codes = tuple({record.index_code for record in records})
    dates = tuple({record.weight_date for record in records})
    existing_by_key = {
        (row.index_code, row.weight_date, row.constituent_code): row
        for row in session.scalars(
            select(IndexWeightSnapshot).where(
                IndexWeightSnapshot.source_id == source_id,
                IndexWeightSnapshot.index_code.in_(index_codes),
                IndexWeightSnapshot.weight_date.in_(dates),
            )
        ).all()
    }
    created_count = 0
    updated_count = 0
    skipped_count = 0
    for record in records:
        key = (record.index_code, record.weight_date, record.constituent_code)
        existing = existing_by_key.get(key)
        if existing is None:
            session.add(IndexWeightSnapshot(source_id=source_id, **record.__dict__))
            created_count += 1
            continue
        if existing.row_hash == record.row_hash:
            skipped_count += 1
            continue
        existing.weight = record.weight
        existing.row_hash = record.row_hash
        updated_count += 1
    return WriteStats(created_count=created_count, updated_count=updated_count, skipped_count=skipped_count)


def mark_cursor_success(
    session: Session,
    *,
    source_id: UUID,
    dataset_code: str,
    entity_key: str,
    last_successful_data_date: date | None,
    last_sync_run_id: UUID,
) -> None:
    """仅在实体数据已成功写入后推进水位并清除该实体失败摘要。"""
    cursor = session.get(
        SourceSyncCursor,
        {"source_id": source_id, "dataset_code": dataset_code, "entity_key": entity_key},
    )
    if cursor is None:
        cursor = SourceSyncCursor(source_id=source_id, dataset_code=dataset_code, entity_key=entity_key)
        session.add(cursor)
    cursor.last_successful_data_date = last_successful_data_date
    cursor.last_sync_run_id = last_sync_run_id
    cursor.consecutive_failure_count = 0
    cursor.last_error_summary = None
    cursor.updated_at = datetime.now(UTC)


def mark_cursor_failure(
    session: Session, *, source_id: UUID, dataset_code: str, entity_key: str, error_summary: str
) -> None:
    """记录实体失败而不改变最后成功水位。"""
    cursor = session.get(
        SourceSyncCursor,
        {"source_id": source_id, "dataset_code": dataset_code, "entity_key": entity_key},
    )
    if cursor is None:
        cursor = SourceSyncCursor(source_id=source_id, dataset_code=dataset_code, entity_key=entity_key)
        session.add(cursor)
    cursor.consecutive_failure_count = (cursor.consecutive_failure_count or 0) + 1
    cursor.last_error_summary = error_summary[:512]
    cursor.updated_at = datetime.now(UTC)
