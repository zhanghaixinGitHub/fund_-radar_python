"""受控业绩基准登记、手动导入和可用性校验。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.session import get_engine
from app.models.benchmark import BenchmarkSeries
from app.repositories.benchmark_series import (
    BenchmarkNavPoint,
    BenchmarkSeriesCoverage,
    find_benchmark_coverage,
    find_benchmark_series,
    find_source_registry,
    list_benchmark_coverages,
    list_benchmark_nav_points,
    upsert_benchmark_nav_points,
)

logger = get_logger(__name__)

STOCK_FUND_TYPE = "STOCK"
MIN_ACTIVATION_NAV_POINTS = 400
MAX_IMPORT_POINTS = 10_000
_BENCHMARK_CODE_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._-]{1,63}$")


class BenchmarkRegistryError(ValueError):
    """基准登记、导入或启用不满足治理条件时抛出。"""


@dataclass(frozen=True)
class BenchmarkPointInput:
    """API 层已解析的一条人工核验基准日收盘点。"""

    nav_date: date
    closing_value: Decimal
    source_published_at: datetime | None = None


def list_stock_benchmarks() -> tuple[BenchmarkSeriesCoverage, ...]:
    """返回股票型基准状态；读取不会修改来源、基准或模型状态。"""
    with Session(get_engine()) as session:
        return list_benchmark_coverages(session, fund_type=STOCK_FUND_TYPE)


def register_stock_benchmark(
    *,
    benchmark_code: str,
    display_name: str,
    source_code: str,
    license_reference: str,
) -> BenchmarkSeriesCoverage:
    """登记候选基准元数据；来源尚未启用时只能保持 DRAFT。"""
    code = _normalize_benchmark_code(benchmark_code)
    clean_name = _required_text(display_name, "display_name", 128)
    clean_source_code = _required_text(source_code, "source_code", 64)
    clean_license_reference = _required_text(license_reference, "license_reference", 512)
    with Session(get_engine()) as session:
        source = find_source_registry(session, clean_source_code)
        if source is None:
            raise BenchmarkRegistryError("BENCHMARK_SOURCE_NOT_REGISTERED")
        series = find_benchmark_series(session, code, lock=True)
        if series is None:
            series = BenchmarkSeries(
                benchmark_code=code,
                display_name=clean_name,
                fund_type=STOCK_FUND_TYPE,
                source_id=source.source_id,
                status="DRAFT",
                license_reference=clean_license_reference,
                row_hash=_stable_hash(
                    {
                        "benchmark_code": code,
                        "display_name": clean_name,
                        "fund_type": STOCK_FUND_TYPE,
                        "source_code": clean_source_code,
                        "license_reference": clean_license_reference,
                    }
                ),
            )
            session.add(series)
        else:
            if series.source_id != source.source_id:
                raise BenchmarkRegistryError("BENCHMARK_SOURCE_CANNOT_CHANGE")
            if series.status == "ACTIVE":
                raise BenchmarkRegistryError("ACTIVE_BENCHMARK_MUST_BE_SUSPENDED_BEFORE_METADATA_CHANGE")
            series.display_name = clean_name
            series.license_reference = clean_license_reference
            series.row_hash = _stable_hash(
                {
                    "benchmark_code": code,
                    "display_name": clean_name,
                    "fund_type": STOCK_FUND_TYPE,
                    "source_code": clean_source_code,
                    "license_reference": clean_license_reference,
                }
            )
        session.commit()
        coverage = find_benchmark_coverage(session, code)
    if coverage is None:
        raise RuntimeError("benchmark registration was not persisted")
    logger.info(
        "benchmark_registry.register_stock_benchmark >>> registered, benchmark_code=%s, source_code=%s, status=%s",
        coverage.benchmark_code,
        coverage.source_code,
        coverage.status,
    )
    return coverage


def import_stock_benchmark_points(
    *,
    benchmark_code: str,
    points: tuple[BenchmarkPointInput, ...],
) -> BenchmarkSeriesCoverage:
    """批量导入已授权人工核验的基准点；只允许 DRAFT/SUSPENDED 基准写入。"""
    code = _normalize_benchmark_code(benchmark_code)
    normalized_points = _normalize_points(points)
    with Session(get_engine()) as session:
        series = find_benchmark_series(session, code, lock=True)
        if series is None:
            raise BenchmarkRegistryError("BENCHMARK_NOT_REGISTERED")
        if series.fund_type != STOCK_FUND_TYPE:
            raise BenchmarkRegistryError("BENCHMARK_FUND_TYPE_NOT_SUPPORTED")
        if series.status == "ACTIVE":
            raise BenchmarkRegistryError("ACTIVE_BENCHMARK_MUST_BE_SUSPENDED_BEFORE_IMPORT")
        # 通过基准覆盖摘要读取来源状态，避免把来源对象暴露到接口层。
        coverage_before = find_benchmark_coverage(session, code)
        if coverage_before is None or not coverage_before.source_enabled:
            raise BenchmarkRegistryError("BENCHMARK_SOURCE_DISABLED")
        changed_count = upsert_benchmark_nav_points(
            session,
            benchmark_code=code,
            points=tuple(
                BenchmarkNavPoint(
                    nav_date=point.nav_date,
                    closing_value=point.closing_value,
                    source_published_at=point.source_published_at,
                    row_hash=_stable_hash(
                        {
                            "benchmark_code": code,
                            "nav_date": point.nav_date.isoformat(),
                            "closing_value": _decimal_text(point.closing_value),
                            "source_published_at": point.source_published_at.isoformat()
                            if point.source_published_at
                            else None,
                        }
                    ),
                )
                for point in normalized_points
            ),
        )
        session.commit()
        coverage = find_benchmark_coverage(session, code)
    if coverage is None:
        raise RuntimeError("benchmark import was not persisted")
    logger.info(
        "benchmark_registry.import_stock_benchmark_points >>> imported, benchmark_code=%s, input=%s, changed=%s",
        code,
        len(normalized_points),
        changed_count,
    )
    return coverage


def activate_stock_benchmark(benchmark_code: str) -> BenchmarkSeriesCoverage:
    """启用已登记基准；来源和最小历史覆盖不足时显式拒绝。"""
    code = _normalize_benchmark_code(benchmark_code)
    with Session(get_engine()) as session:
        series = find_benchmark_series(session, code, lock=True)
        if series is None:
            raise BenchmarkRegistryError("BENCHMARK_NOT_REGISTERED")
        coverage_before = find_benchmark_coverage(session, code)
        if coverage_before is None or not coverage_before.source_enabled:
            raise BenchmarkRegistryError("BENCHMARK_SOURCE_DISABLED")
        if coverage_before.point_count < MIN_ACTIVATION_NAV_POINTS:
            raise BenchmarkRegistryError(
                "BENCHMARK_HISTORY_SHORTAGE: "
                f"observed={coverage_before.point_count}, required={MIN_ACTIVATION_NAV_POINTS}"
            )
        if series.status == "ACTIVE":
            return coverage_before
        series.status = "ACTIVE"
        session.commit()
        coverage = find_benchmark_coverage(session, code)
    if coverage is None:
        raise RuntimeError("benchmark activation was not persisted")
    logger.info(
        "benchmark_registry.activate_stock_benchmark >>> activated, benchmark_code=%s, point_count=%s",
        coverage.benchmark_code,
        coverage.point_count,
    )
    return coverage


def suspend_stock_benchmark(benchmark_code: str) -> BenchmarkSeriesCoverage:
    """暂停基准，阻止其被新的回测使用；历史回测结果保留。"""
    code = _normalize_benchmark_code(benchmark_code)
    with Session(get_engine()) as session:
        series = find_benchmark_series(session, code, lock=True)
        if series is None:
            raise BenchmarkRegistryError("BENCHMARK_NOT_REGISTERED")
        if series.status != "ACTIVE":
            raise BenchmarkRegistryError("ONLY_ACTIVE_BENCHMARK_CAN_SUSPEND")
        series.status = "SUSPENDED"
        session.commit()
        coverage = find_benchmark_coverage(session, code)
    if coverage is None:
        raise RuntimeError("benchmark suspension was not persisted")
    logger.info("benchmark_registry.suspend_stock_benchmark >>> suspended, benchmark_code=%s", code)
    return coverage


def load_active_stock_benchmark_points(benchmark_code: str | None) -> tuple[BenchmarkNavPoint, ...]:
    """回测前加载可用基准；不满足授权、类别或状态时返回空序列。"""
    if benchmark_code is None:
        return ()
    code = _normalize_benchmark_code(benchmark_code)
    with Session(get_engine()) as session:
        coverage = find_benchmark_coverage(session, code)
        if (
            coverage is None
            or coverage.fund_type != STOCK_FUND_TYPE
            or coverage.status != "ACTIVE"
            or not coverage.source_enabled
        ):
            return ()
        return list_benchmark_nav_points(session, code)


def get_stock_benchmark_readiness(benchmark_code: str | None) -> str | None:
    """返回可安全写入回测失败摘要的基准不可用原因；可用时返回 None。"""
    if benchmark_code is None:
        return "BENCHMARK_NOT_CONFIGURED"
    code = _normalize_benchmark_code(benchmark_code)
    with Session(get_engine()) as session:
        coverage = find_benchmark_coverage(session, code)
    if coverage is None:
        return "BENCHMARK_NOT_REGISTERED"
    if coverage.fund_type != STOCK_FUND_TYPE:
        return "BENCHMARK_FUND_TYPE_NOT_SUPPORTED"
    if not coverage.source_enabled:
        return "BENCHMARK_SOURCE_DISABLED"
    if coverage.status != "ACTIVE":
        return "BENCHMARK_NOT_ACTIVE"
    if coverage.point_count < MIN_ACTIVATION_NAV_POINTS:
        return "BENCHMARK_HISTORY_SHORTAGE"
    return None


def _normalize_benchmark_code(benchmark_code: str) -> str:
    code = _required_text(benchmark_code, "benchmark_code", 64).upper()
    if not _BENCHMARK_CODE_PATTERN.fullmatch(code):
        raise BenchmarkRegistryError("BENCHMARK_CODE_INVALID")
    return code


def _required_text(value: str, field_name: str, max_length: int) -> str:
    clean_value = value.strip()
    if not clean_value or len(clean_value) > max_length:
        raise BenchmarkRegistryError(f"{field_name.upper()}_INVALID")
    return clean_value


def _normalize_points(points: tuple[BenchmarkPointInput, ...]) -> tuple[BenchmarkPointInput, ...]:
    if not points or len(points) > MAX_IMPORT_POINTS:
        raise BenchmarkRegistryError("BENCHMARK_IMPORT_SIZE_INVALID")
    normalized: dict[date, BenchmarkPointInput] = {}
    for point in points:
        if point.closing_value <= 0 or not point.closing_value.is_finite():
            raise BenchmarkRegistryError("BENCHMARK_CLOSING_VALUE_INVALID")
        existing = normalized.get(point.nav_date)
        if existing is not None and existing.closing_value != point.closing_value:
            raise BenchmarkRegistryError("BENCHMARK_IMPORT_DUPLICATE_DATE_CONFLICT")
        normalized[point.nav_date] = point
    return tuple(normalized[nav_date] for nav_date in sorted(normalized))


def _stable_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _decimal_text(value: Decimal) -> str:
    try:
        return format(value.normalize(), "f")
    except (AttributeError, InvalidOperation) as error:
        raise BenchmarkRegistryError("BENCHMARK_CLOSING_VALUE_INVALID") from error
