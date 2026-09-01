"""M3-G1 股票型基金特征快照构建；不产生预测、回测或用户提醒。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.session import get_engine
from app.repositories.feature_snapshot import (
    FeatureNavPoint,
    FeatureSnapshotUpsert,
    FeatureSnapshotWriteStats,
    StockFeatureInput,
    get_enabled_feature_source,
    list_stock_feature_inputs,
    upsert_feature_snapshots,
)
from app.repositories.fund_sync import TUSHARE_SOURCE_CODE

logger = get_logger(__name__)

STOCK_FEATURE_VERSION = "M3_STOCK_FEATURE_V1"
STOCK_FUND_TYPE = "STOCK"
MIN_NAV_OBSERVATIONS = 252
FEATURE_LOOKBACK_OBSERVATIONS = 60
_COMPLETENESS_QUANTUM = Decimal("0.0001")
_METRIC_QUANTUM = Decimal("0.00000001")
_STOCK_FEATURE_SNAPSHOT_LOCK_KEY = 7_089_123_007

FeatureProgressReporter = Callable[[int, int, str | None, str], None]


class FeatureSnapshotBuildInProgressError(RuntimeError):
    """另一个进程已在构建同一类特征快照时拒绝重复写入。"""


@dataclass(frozen=True)
class StockFeatureBuildSummary:
    """一次受控特征构建的可审计摘要。"""

    status: str
    source_code: str
    source_sync_run_id: UUID | None
    attempted_fund_count: int
    scorable_count: int
    data_insufficient_count: int
    no_nav_count: int
    created_count: int
    updated_count: int
    skipped_count: int

    def to_payload(self) -> dict[str, str | int | None]:
        """返回可由 Celery JSON 序列化的安全摘要。"""
        return {
            "status": self.status,
            "source_code": self.source_code,
            "source_sync_run_id": str(self.source_sync_run_id) if self.source_sync_run_id else None,
            "attempted_fund_count": self.attempted_fund_count,
            "scorable_count": self.scorable_count,
            "data_insufficient_count": self.data_insufficient_count,
            "no_nav_count": self.no_nav_count,
            "created_count": self.created_count,
            "updated_count": self.updated_count,
            "skipped_count": self.skipped_count,
        }


class StockFeatureSnapshotService:
    """使用已登记来源构建股票型基金的只读输入特征并幂等落库。"""

    def build(self, *, progress_reporter: FeatureProgressReporter | None = None) -> StockFeatureBuildSummary:
        """构建试点特征快照；来源未就绪时拒绝写库，不访问外部系统。"""
        engine = get_engine()
        with _stock_feature_snapshot_lock(engine):
            return self._build_with_engine(engine, progress_reporter=progress_reporter)

    def _build_with_engine(
        self, engine: Engine, *, progress_reporter: FeatureProgressReporter | None = None
    ) -> StockFeatureBuildSummary:
        """在已持有跨进程互斥锁时完成读取、计算与幂等写入。"""
        with Session(engine) as session:
            source = get_enabled_feature_source(session, TUSHARE_SOURCE_CODE)
            if source is None:
                logger.warning(
                    "stock_feature_snapshot.build >>> feature build blocked because source is absent or disabled, "
                    "source_code=%s",
                    TUSHARE_SOURCE_CODE,
                )
                return StockFeatureBuildSummary(
                    status="SOURCE_NOT_READY",
                    source_code=TUSHARE_SOURCE_CODE,
                    source_sync_run_id=None,
                    attempted_fund_count=0,
                    scorable_count=0,
                    data_insufficient_count=0,
                    no_nav_count=0,
                    created_count=0,
                    updated_count=0,
                    skipped_count=0,
                )

            inputs = list_stock_feature_inputs(
                session,
                source_code=source.source_code,
                source_sync_run_id=source.source_sync_run_id,
                source_sync_finished_at=source.source_sync_finished_at,
                history_limit=MIN_NAV_OBSERVATIONS,
            )
            snapshots_list: list[FeatureSnapshotUpsert] = []
            for current, item in enumerate(inputs, start=1):
                snapshot = build_stock_feature_snapshot(item)
                if snapshot is not None:
                    snapshots_list.append(snapshot)
                if progress_reporter is not None:
                    progress_reporter(
                        current,
                        len(inputs),
                        item.fund_code,
                        f"正在生成 {item.fund_code} 的特征快照",
                    )
            snapshots = tuple(snapshots_list)
            no_nav_count = len(inputs) - len(snapshots)
            scorable_count = sum(snapshot.eligibility_status == "SCORABLE" for snapshot in snapshots)
            data_insufficient_count = sum(
                snapshot.eligibility_status == "DATA_INSUFFICIENT" for snapshot in snapshots
            )
            try:
                write_stats = upsert_feature_snapshots(session, records=snapshots)
                session.commit()
            except Exception:
                session.rollback()
                logger.exception(
                    "stock_feature_snapshot.build >>> feature build failed, source_code=%s, attempted_fund_count=%s",
                    source.source_code,
                    len(inputs),
                )
                raise

        summary = _to_summary(
            source_code=source.source_code,
            source_sync_run_id=source.source_sync_run_id,
            attempted_fund_count=len(inputs),
            scorable_count=scorable_count,
            data_insufficient_count=data_insufficient_count,
            no_nav_count=no_nav_count,
            write_stats=write_stats,
        )
        logger.info(
            "stock_feature_snapshot.build >>> feature build completed, source_code=%s, attempted=%s, scorable=%s, "
            "data_insufficient=%s, no_nav=%s, created=%s, updated=%s, skipped=%s",
            summary.source_code,
            summary.attempted_fund_count,
            summary.scorable_count,
            summary.data_insufficient_count,
            summary.no_nav_count,
            summary.created_count,
            summary.updated_count,
            summary.skipped_count,
        )
        return summary


@contextmanager
def _stock_feature_snapshot_lock(engine: Engine) -> Iterator[None]:
    """串行化计划任务和手动重试，避免同一快照产生并发写入竞争。"""
    with engine.connect() as connection:
        acquired = bool(connection.scalar(select(func.pg_try_advisory_lock(_STOCK_FEATURE_SNAPSHOT_LOCK_KEY))))
        if not acquired:
            raise FeatureSnapshotBuildInProgressError("stock feature snapshot build is already running")
        try:
            yield
        finally:
            released = bool(connection.scalar(select(func.pg_advisory_unlock(_STOCK_FEATURE_SNAPSHOT_LOCK_KEY))))
            if not released:
                logger.error("stock_feature_snapshot._stock_feature_snapshot_lock >>> advisory lock release failed")


def build_stock_feature_snapshot(input_record: StockFeatureInput) -> FeatureSnapshotUpsert | None:
    """从单只基金的受控净值序列构建可复现特征；绝不生成方向、概率或信心结论。"""
    if input_record.fund_type != STOCK_FUND_TYPE:
        raise ValueError(f"unsupported pilot fund_type={input_record.fund_type}")
    if not input_record.nav_points:
        return None

    latest_nav_date = input_record.nav_points[-1].nav_date
    base_payload = {
        "schema_version": STOCK_FEATURE_VERSION,
        "source": {
            "source_code": input_record.source_code,
            "source_sync_run_id": str(input_record.source_sync_run_id),
            "source_sync_finished_at": input_record.source_sync_finished_at.isoformat(),
        },
        "input": {
            "as_of_date": latest_nav_date.isoformat(),
            "history_start_date": input_record.nav_points[0].nav_date.isoformat(),
            "history_end_date": latest_nav_date.isoformat(),
            "nav_observation_count": len(input_record.nav_points),
            "minimum_required_nav_observation_count": MIN_NAV_OBSERVATIONS,
        },
        "evidence_refs": [
            {
                "source_sync_run_id": str(input_record.source_sync_run_id),
                "data_as_of": latest_nav_date.isoformat(),
            }
        ],
    }
    if any(point.unit_nav <= 0 for point in input_record.nav_points):
        return _data_insufficient_snapshot(
            input_record,
            latest_nav_date,
            base_payload,
            reason="INVALID_UNIT_NAV",
            completeness=Decimal("0"),
        )
    if len(input_record.nav_points) < MIN_NAV_OBSERVATIONS:
        return _data_insufficient_snapshot(
            input_record,
            latest_nav_date,
            base_payload,
            reason=(
                "NAV_HISTORY_SHORTAGE: "
                f"observed={len(input_record.nav_points)}, required={MIN_NAV_OBSERVATIONS}"
            ),
            completeness=_completeness(len(input_record.nav_points)),
        )

    nav_values, nav_basis = _select_consistent_nav_values(input_record.nav_points)
    metrics = {
        "return_5d": _decimal_text(_period_return(nav_values, 5)),
        "return_20d": _decimal_text(_period_return(nav_values, 20)),
        "return_60d": _decimal_text(_period_return(nav_values, FEATURE_LOOKBACK_OBSERVATIONS)),
        "volatility_20d": _decimal_text(_volatility(nav_values[-21:])),
        "max_drawdown_60d": _decimal_text(_max_drawdown(nav_values[-FEATURE_LOOKBACK_OBSERVATIONS:])),
    }
    payload: dict[str, object] = {
        **base_payload,
        "source": {
            "source_code": input_record.source_code,
            "source_sync_run_id": str(input_record.source_sync_run_id),
            "source_sync_finished_at": input_record.source_sync_finished_at.isoformat(),
            "nav_value_basis": nav_basis,
        },
        "quality": {"status": "SCORABLE", "issues": []},
        "metrics": metrics,
    }
    return _snapshot_record(
        input_record=input_record,
        as_of_date=latest_nav_date,
        completeness=Decimal("1"),
        eligibility_status="SCORABLE",
        unavailable_reason=None,
        payload=payload,
    )


def _data_insufficient_snapshot(
    input_record: StockFeatureInput,
    as_of_date,
    base_payload: dict[str, object],
    *,
    reason: str,
    completeness: Decimal,
) -> FeatureSnapshotUpsert:
    payload: dict[str, object] = {
        **base_payload,
        "quality": {"status": "DATA_INSUFFICIENT", "issues": [reason]},
        "metrics": None,
    }
    return _snapshot_record(
        input_record=input_record,
        as_of_date=as_of_date,
        completeness=completeness,
        eligibility_status="DATA_INSUFFICIENT",
        unavailable_reason=reason,
        payload=payload,
    )


def _snapshot_record(
    *,
    input_record: StockFeatureInput,
    as_of_date,
    completeness: Decimal,
    eligibility_status: str,
    unavailable_reason: str | None,
    payload: dict[str, object],
) -> FeatureSnapshotUpsert:
    return FeatureSnapshotUpsert(
        fund_code=input_record.fund_code,
        as_of_date=as_of_date,
        fund_type=input_record.fund_type,
        feature_version=STOCK_FEATURE_VERSION,
        completeness=completeness.quantize(_COMPLETENESS_QUANTUM, rounding=ROUND_HALF_UP),
        eligibility_status=eligibility_status,
        unavailable_reason=unavailable_reason,
        feature_payload=payload,
        feature_hash=_feature_hash(
            {
                "fund_code": input_record.fund_code,
                "as_of_date": as_of_date.isoformat(),
                "fund_type": input_record.fund_type,
                "feature_version": STOCK_FEATURE_VERSION,
                "completeness": _decimal_text(completeness),
                "eligibility_status": eligibility_status,
                "unavailable_reason": unavailable_reason,
                "feature_payload": payload,
            }
        ),
    )


def _select_consistent_nav_values(nav_points: tuple[FeatureNavPoint, ...]) -> tuple[tuple[Decimal, ...], str]:
    """优先全量使用累计净值；任一缺失时统一回退单位净值，避免混合序列。"""
    if all(point.accumulated_nav is not None and point.accumulated_nav > 0 for point in nav_points):
        return (
            tuple(point.accumulated_nav for point in nav_points if point.accumulated_nav is not None),
            "ACCUMULATED_NAV",
        )
    return tuple(point.unit_nav for point in nav_points), "UNIT_NAV"


def _period_return(nav_values: tuple[Decimal, ...], interval_count: int) -> Decimal:
    if len(nav_values) <= interval_count:
        raise ValueError(f"insufficient observations for interval_count={interval_count}")
    base = nav_values[-(interval_count + 1)]
    if base <= 0:
        raise ValueError("nav base must be positive")
    return nav_values[-1] / base - Decimal("1")


def _volatility(nav_values: tuple[Decimal, ...]) -> Decimal:
    """计算最近二十个净值区间的总体标准差，不年化、不附会为风险结论。"""
    if len(nav_values) < 2:
        raise ValueError("at least two observations are required for volatility")
    returns = tuple(
        current / previous - Decimal("1")
        for previous, current in zip(nav_values[:-1], nav_values[1:], strict=True)
    )
    mean = sum(returns, Decimal("0")) / Decimal(len(returns))
    variance = sum(((value - mean) ** 2 for value in returns), Decimal("0")) / Decimal(len(returns))
    try:
        return variance.sqrt()
    except InvalidOperation as error:
        raise ValueError("volatility variance cannot be represented") from error


def _max_drawdown(nav_values: tuple[Decimal, ...]) -> Decimal:
    """计算固定窗口内历史最大回撤，结果为非正数。"""
    if not nav_values:
        raise ValueError("at least one observation is required for max drawdown")
    peak = nav_values[0]
    max_drawdown = Decimal("0")
    for value in nav_values:
        if value > peak:
            peak = value
        drawdown = value / peak - Decimal("1")
        if drawdown < max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def _completeness(observation_count: int) -> Decimal:
    return min(Decimal("1"), Decimal(observation_count) / Decimal(MIN_NAV_OBSERVATIONS))


def _decimal_text(value: Decimal) -> str:
    return format(value.quantize(_METRIC_QUANTUM, rounding=ROUND_HALF_UP), "f")


def _feature_hash(value: dict[str, object]) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _to_summary(
    *,
    source_code: str,
    source_sync_run_id: UUID,
    attempted_fund_count: int,
    scorable_count: int,
    data_insufficient_count: int,
    no_nav_count: int,
    write_stats: FeatureSnapshotWriteStats,
) -> StockFeatureBuildSummary:
    return StockFeatureBuildSummary(
        status="COMPLETED",
        source_code=source_code,
        source_sync_run_id=source_sync_run_id,
        attempted_fund_count=attempted_fund_count,
        scorable_count=scorable_count,
        data_insufficient_count=data_insufficient_count,
        no_nav_count=no_nav_count,
        created_count=write_stats.created_count,
        updated_count=write_stats.updated_count,
        skipped_count=write_stats.skipped_count,
    )
