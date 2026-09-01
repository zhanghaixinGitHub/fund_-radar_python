"""M3-04 基线评分与滚动回测的纯计算边界测试。"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from uuid import uuid4

from app.models.analysis import AnalysisModelRelease, FeatureSnapshot
from app.repositories.analysis_execution import BacktestNavPoint
from app.services.baseline_analysis import (
    RollingBacktestConfig,
    _build_backtest_observations,
    evaluate_stock_rolling_backtest,
    score_stock_feature_snapshot,
)


def _scorable_snapshot() -> FeatureSnapshot:
    """构造不依赖数据库的完整股票型特征快照。"""
    return FeatureSnapshot(
        feature_id=uuid4(),
        fund_code="000001.OF",
        as_of_date=date(2026, 8, 31),
        fund_type="STOCK",
        feature_version="M3_STOCK_FEATURE_V1",
        completeness=Decimal("1"),
        eligibility_status="SCORABLE",
        unavailable_reason=None,
        feature_payload={
            "metrics": {
                "return_20d": "0.03000000",
                "volatility_20d": "0.01000000",
                "max_drawdown_60d": "-0.04000000",
            }
        },
        feature_hash="a" * 64,
    )


def _active_release() -> AnalysisModelRelease:
    """构造与特征版本兼容的已发布模型。"""
    return AnalysisModelRelease(
        model_release_id=uuid4(),
        model_code="M3_STOCK_MOMENTUM_BASELINE",
        model_version="M3_STOCK_MOMENTUM_BASELINE_V1",
        feature_version="M3_STOCK_FEATURE_V1",
        fund_type="STOCK",
        backtest_run_id=uuid4(),
        release_status="ACTIVE",
        release_reason="测试发布",
        config_hash="b" * 64,
    )


def _nav_points(*, changed_future_value: Decimal | None = None) -> tuple[BacktestNavPoint, ...]:
    """生成三只基金的合成净值序列，避免测试依赖外部数据或数据库。"""
    records: list[BacktestNavPoint] = []
    start = date(2020, 1, 1)
    for fund_index, fund_code in enumerate(("000001.OF", "000002.OF", "000003.OF"), start=1):
        for index in range(520):
            value = Decimal("100") + Decimal(index) * Decimal("0.08") + Decimal(fund_index)
            if changed_future_value is not None and fund_code == "000001.OF" and index == 70:
                value = changed_future_value
            records.append(
                BacktestNavPoint(
                    fund_code=fund_code,
                    nav_date=start + timedelta(days=index),
                    unit_nav=value,
                    accumulated_nav=value,
                )
            )
    return tuple(records)


def test_scorable_feature_without_active_release_is_explicitly_model_rejected() -> None:
    """模型未发布时不得伪造方向、概率或置信度。"""
    result = score_stock_feature_snapshot(_scorable_snapshot(), None)

    assert result.score_status == "MODEL_REJECTED"
    assert result.direction is None
    assert result.directional_probability is None
    assert result.confidence is None
    assert result.model_release_id is None


def test_active_release_allows_explainable_score_with_release_lineage() -> None:
    """完整特征只在 ACTIVE 且特征版本兼容时产生带血缘的评分。"""
    release = _active_release()
    result = score_stock_feature_snapshot(_scorable_snapshot(), release)

    assert result.score_status == "SCORED"
    assert result.direction == "UP"
    assert result.directional_probability == Decimal("0.5600")
    assert result.confidence == Decimal("0.5600")
    assert result.model_release_id == release.model_release_id
    assert "不构成交易指令" in result.explanation


def test_data_insufficient_feature_never_has_directional_fields() -> None:
    """特征不足优先降级，即使模型已发布也不允许生成方向性字段。"""
    snapshot = _scorable_snapshot()
    snapshot.eligibility_status = "DATA_INSUFFICIENT"
    snapshot.unavailable_reason = "NAV_HISTORY_SHORTAGE"

    result = score_stock_feature_snapshot(snapshot, _active_release())

    assert result.score_status == "DATA_INSUFFICIENT"
    assert result.direction is None
    assert result.directional_probability is None
    assert result.confidence is None
    assert result.risk_level is None


def test_future_label_change_does_not_change_historical_signal_direction() -> None:
    """离线标签可以变化，但信号只由信号日及此前的 20 日输入决定。"""
    original = _build_backtest_observations(_nav_points(), 20)
    changed = _build_backtest_observations(_nav_points(changed_future_value=Decimal("9999")), 20)
    signal_date = date(2020, 1, 1) + timedelta(days=50)
    original_observation = next(
        item for item in original if item.fund_code == "000001.OF" and item.signal_date == signal_date
    )
    changed_observation = next(
        item for item in changed if item.fund_code == "000001.OF" and item.signal_date == signal_date
    )

    assert changed_observation.previous_return == original_observation.previous_return
    assert changed_observation.direction == original_observation.direction
    assert changed_observation.realized_return != original_observation.realized_return


def test_rolling_backtest_uses_strict_time_order_and_blocks_missing_benchmark() -> None:
    """训练、验证、测试边界严格递进；无已授权基准不能通过模型发布闸门。"""
    evaluation = evaluate_stock_rolling_backtest(
        _nav_points(),
        RollingBacktestConfig(
            min_train_dates=180,
            validation_dates=60,
            test_dates=60,
            rolling_step_dates=60,
            min_test_samples=10,
        ),
    )

    assert evaluation.train_end is not None
    assert evaluation.validation_end is not None
    assert evaluation.test_start is not None
    assert evaluation.test_end is not None
    assert evaluation.train_end < evaluation.validation_end < evaluation.test_start <= evaluation.test_end
    assert evaluation.metrics["sample_count"] >= 10
    assert evaluation.baselines["benchmark_status"] == "NOT_CONFIGURED"
    assert evaluation.publication_status == "INELIGIBLE"
    assert evaluation.failure_reason is not None
    assert "BENCHMARK_NOT_CONFIGURED" in evaluation.failure_reason
