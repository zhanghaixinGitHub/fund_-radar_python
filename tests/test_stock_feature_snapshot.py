"""M3-G1 股票型基金特征快照的纯计算边界验证。"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID

from app.repositories.feature_snapshot import FeatureNavPoint, StockFeatureInput
from app.services.stock_feature_snapshot import MIN_NAV_OBSERVATIONS, build_stock_feature_snapshot


def _input_with_points(
    point_count: int,
    *,
    accumulated_nav_missing_at: int | None = None,
) -> StockFeatureInput:
    start_date = date(2025, 1, 1)
    points = tuple(
        FeatureNavPoint(
            nav_date=start_date + timedelta(days=index),
            unit_nav=Decimal("1") + Decimal(index) / Decimal("1000"),
            accumulated_nav=(
                None
                if accumulated_nav_missing_at == index
                else Decimal("1.1") + Decimal(index) / Decimal("1000")
            ),
        )
        for index in range(point_count)
    )
    return StockFeatureInput(
        fund_code="000001",
        fund_type="STOCK",
        source_code="TUSHARE_PRO_FUND",
        source_sync_run_id=UUID("00000000-0000-0000-0000-000000000001"),
        source_sync_finished_at=datetime(2026, 9, 1, 6, 0, tzinfo=UTC),
        nav_points=points,
    )


def test_stock_feature_snapshot_is_scorable_but_does_not_contain_directional_conclusion() -> None:
    """满足试点输入门槛时只产生特征，不产生方向、概率或置信度字段。"""
    snapshot = build_stock_feature_snapshot(_input_with_points(MIN_NAV_OBSERVATIONS))

    assert snapshot is not None
    assert snapshot.eligibility_status == "SCORABLE"
    assert snapshot.completeness == Decimal("1.0000")
    assert snapshot.feature_payload["source"] == {
        "source_code": "TUSHARE_PRO_FUND",
        "source_sync_run_id": "00000000-0000-0000-0000-000000000001",
        "source_sync_finished_at": "2026-09-01T06:00:00+00:00",
        "nav_value_basis": "ACCUMULATED_NAV",
    }
    metrics = snapshot.feature_payload["metrics"]
    assert isinstance(metrics, dict)
    assert set(metrics) == {"return_5d", "return_20d", "return_60d", "volatility_20d", "max_drawdown_60d"}
    assert "direction" not in snapshot.feature_payload
    assert "directional_probability" not in snapshot.feature_payload
    assert "confidence" not in snapshot.feature_payload


def test_stock_feature_snapshot_reports_short_history_without_metrics() -> None:
    """历史不足时必须显式标记数据不足，不能填充伪造指标。"""
    snapshot = build_stock_feature_snapshot(_input_with_points(MIN_NAV_OBSERVATIONS - 1))

    assert snapshot is not None
    assert snapshot.eligibility_status == "DATA_INSUFFICIENT"
    assert snapshot.feature_payload["metrics"] is None
    assert snapshot.unavailable_reason == "NAV_HISTORY_SHORTAGE: observed=251, required=252"
    assert snapshot.feature_payload["quality"] == {
        "status": "DATA_INSUFFICIENT",
        "issues": ["NAV_HISTORY_SHORTAGE: observed=251, required=252"],
    }


def test_stock_feature_snapshot_uses_a_single_nav_basis_when_accumulated_nav_is_incomplete() -> None:
    """累计净值不完整时全序列回退单位净值，避免把两个口径混合计算。"""
    snapshot = build_stock_feature_snapshot(
        _input_with_points(MIN_NAV_OBSERVATIONS, accumulated_nav_missing_at=100)
    )

    assert snapshot is not None
    assert snapshot.eligibility_status == "SCORABLE"
    assert snapshot.feature_payload["source"] == {
        "source_code": "TUSHARE_PRO_FUND",
        "source_sync_run_id": "00000000-0000-0000-0000-000000000001",
        "source_sync_finished_at": "2026-09-01T06:00:00+00:00",
        "nav_value_basis": "UNIT_NAV",
    }


def test_stock_feature_snapshot_is_deterministic_for_identical_input() -> None:
    """同一输入必须生成相同摘要，保证重跑时可以幂等跳过。"""
    input_record = _input_with_points(MIN_NAV_OBSERVATIONS)

    first = build_stock_feature_snapshot(input_record)
    second = build_stock_feature_snapshot(input_record)

    assert first is not None
    assert second is not None
    assert first.feature_hash == second.feature_hash
