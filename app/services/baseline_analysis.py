"""M3-04 股票型基金可解释基线评分、滚动回测和模型发布闸门。"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from statistics import fmean, pstdev
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.session import get_engine
from app.models.analysis import AnalysisModelRelease, BacktestRun, FeatureSnapshot
from app.repositories.analysis_execution import (
    BacktestNavPoint,
    ForecastResultUpsert,
    ForecastWriteStats,
    find_backtest_run,
    get_active_model_release,
    list_latest_feature_snapshots,
    list_stock_backtest_nav_points,
    upsert_forecast_results,
)
from app.repositories.benchmark_series import BenchmarkNavPoint
from app.repositories.fund_sync import TUSHARE_SOURCE_CODE
from app.services.benchmark_registry import (
    get_stock_benchmark_readiness,
    load_active_stock_benchmark_points,
)
from app.services.stock_feature_snapshot import STOCK_FEATURE_VERSION

logger = get_logger(__name__)

STOCK_FUND_TYPE = "STOCK"
STOCK_BASELINE_MODEL_CODE = "M3_STOCK_MOMENTUM_BASELINE"
STOCK_BASELINE_MODEL_VERSION = "M3_STOCK_MOMENTUM_BASELINE_V1"
STOCK_BASELINE_STRATEGY_VERSION = "M3_STOCK_MOMENTUM_ROLLING_V1"
DEFAULT_TARGET_HORIZON_TRADING_DAYS = 20
DEFAULT_FEE_RATE = Decimal("0.001500")
DEFAULT_MIN_TRAIN_DATES = 252
DEFAULT_VALIDATION_DATES = 60
DEFAULT_TEST_DATES = 60
DEFAULT_ROLLING_STEP_DATES = 60
DEFAULT_MIN_TEST_SAMPLES = 100
DEFAULT_MIN_HIT_RATE = Decimal("0.5000")
DEFAULT_MIN_BENCHMARK_COVERAGE = Decimal("0.9500")
_PROBABILITY_QUANTUM = Decimal("0.0001")
_DRAWDOWN_QUANTUM = Decimal("0.000001")
_ANALYSIS_SCORE_LOCK_KEY = 7_089_123_101
_ANALYSIS_BACKTEST_LOCK_KEY = 7_089_123_102


class AnalysisRunInProgressError(RuntimeError):
    """同类评分或回测已经由其他进程执行时拒绝并发运行。"""


class ModelReleaseTransitionError(ValueError):
    """模型发布状态不满足激活、暂停或退役条件时抛出。"""


@dataclass(frozen=True)
class BaselineScoringSummary:
    """一次最新特征评分的安全统计摘要。"""

    status: str
    fund_type: str
    model_version: str
    active_model_release_id: UUID | None
    attempted_count: int
    scored_count: int
    data_insufficient_count: int
    not_applicable_count: int
    model_rejected_count: int
    created_count: int
    updated_count: int
    skipped_count: int

    def to_payload(self) -> dict[str, str | int | None]:
        """返回适合 Celery 任务状态的非敏感 JSON 摘要。"""
        return {
            "status": self.status,
            "fund_type": self.fund_type,
            "model_version": self.model_version,
            "active_model_release_id": str(self.active_model_release_id) if self.active_model_release_id else None,
            "attempted_count": self.attempted_count,
            "scored_count": self.scored_count,
            "data_insufficient_count": self.data_insufficient_count,
            "not_applicable_count": self.not_applicable_count,
            "model_rejected_count": self.model_rejected_count,
            "created_count": self.created_count,
            "updated_count": self.updated_count,
            "skipped_count": self.skipped_count,
        }


@dataclass(frozen=True)
class RollingBacktestConfig:
    """单类别、单目标周期的可审计回测配置。"""

    fund_type: str = STOCK_FUND_TYPE
    source_code: str = TUSHARE_SOURCE_CODE
    feature_version: str = STOCK_FEATURE_VERSION
    model_code: str = STOCK_BASELINE_MODEL_CODE
    model_version: str = STOCK_BASELINE_MODEL_VERSION
    strategy_version: str = STOCK_BASELINE_STRATEGY_VERSION
    target_horizon_trading_days: int = DEFAULT_TARGET_HORIZON_TRADING_DAYS
    fee_rate: Decimal = DEFAULT_FEE_RATE
    min_train_dates: int = DEFAULT_MIN_TRAIN_DATES
    validation_dates: int = DEFAULT_VALIDATION_DATES
    test_dates: int = DEFAULT_TEST_DATES
    rolling_step_dates: int = DEFAULT_ROLLING_STEP_DATES
    min_test_samples: int = DEFAULT_MIN_TEST_SAMPLES
    min_hit_rate: Decimal = DEFAULT_MIN_HIT_RATE
    benchmark_id: str | None = None
    min_benchmark_coverage: Decimal = DEFAULT_MIN_BENCHMARK_COVERAGE

    def validate(self) -> None:
        """校验时间、费用和门槛，避免构造无法解释的回测。"""
        positive_fields = (
            self.target_horizon_trading_days,
            self.min_train_dates,
            self.validation_dates,
            self.test_dates,
            self.rolling_step_dates,
            self.min_test_samples,
        )
        if self.fund_type != STOCK_FUND_TYPE:
            raise ValueError(f"unsupported pilot fund_type={self.fund_type}")
        if any(value < 1 for value in positive_fields):
            raise ValueError("rolling backtest date and sample windows must be positive")
        if self.fee_rate < 0 or self.fee_rate >= 1:
            raise ValueError("fee_rate must be in [0, 1)")
        if self.min_hit_rate < 0 or self.min_hit_rate > 1:
            raise ValueError("min_hit_rate must be in [0, 1]")
        if self.min_benchmark_coverage <= 0 or self.min_benchmark_coverage > 1:
            raise ValueError("min_benchmark_coverage must be in (0, 1]")

    @property
    def config_hash(self) -> str:
        """配置哈希用于关联回测、候选发布和运行审计。"""
        return _stable_hash(
            {
                "fund_type": self.fund_type,
                "source_code": self.source_code,
                "feature_version": self.feature_version,
                "model_code": self.model_code,
                "model_version": self.model_version,
                "strategy_version": self.strategy_version,
                "target_horizon_trading_days": self.target_horizon_trading_days,
                "fee_rate": str(self.fee_rate),
                "min_train_dates": self.min_train_dates,
                "validation_dates": self.validation_dates,
                "test_dates": self.test_dates,
                "rolling_step_dates": self.rolling_step_dates,
                "min_test_samples": self.min_test_samples,
                "min_hit_rate": str(self.min_hit_rate),
                "benchmark_id": self.benchmark_id,
                "min_benchmark_coverage": str(self.min_benchmark_coverage),
            }
        )


@dataclass(frozen=True)
class BacktestObservation:
    """由历史净值生成的一条时间点预测与已知未来标签，仅用于离线评估。"""

    fund_code: str
    signal_date: date
    future_date: date
    signal_nav: Decimal
    future_nav: Decimal
    previous_return: Decimal
    realized_return: Decimal
    direction: str


@dataclass(frozen=True)
class RollingFold:
    """一次扩展训练窗口、验证窗口和独立测试窗口的日期边界。"""

    train_end: date
    validation_end: date
    test_start: date
    test_end: date


@dataclass(frozen=True)
class RollingBacktestEvaluation:
    """纯计算回测结果，持久化前不包含数据库状态。"""

    window_start: date
    window_end: date
    train_end: date | None
    validation_end: date | None
    test_start: date | None
    test_end: date | None
    metrics: dict[str, object]
    baselines: dict[str, object]
    publication_status: str
    failure_reason: str | None


@dataclass(frozen=True)
class BacktestRunSummary:
    """已落库的回测运行与候选发布状态摘要。"""

    run_id: UUID
    status: str
    publication_status: str
    failure_reason: str | None
    model_release_id: UUID | None
    model_release_status: str | None

    def to_payload(self) -> dict[str, str | None]:
        """返回可由任务层安全转发的运行摘要。"""
        return {
            "run_id": str(self.run_id),
            "status": self.status,
            "publication_status": self.publication_status,
            "failure_reason": self.failure_reason,
            "model_release_id": str(self.model_release_id) if self.model_release_id else None,
            "model_release_status": self.model_release_status,
        }


class BaselineAnalysisService:
    """执行 M3-04 的本地评分、回测和模型发布状态机。"""

    def score_latest_stock_features(self) -> BaselineScoringSummary:
        """只对最新快照评分；没有 ACTIVE 发布时显式落 MODEL_REJECTED。"""
        engine = get_engine()
        with _analysis_lock(engine, _ANALYSIS_SCORE_LOCK_KEY, "stock baseline scoring"):
            with Session(engine) as session:
                active_release = get_active_model_release(
                    session,
                    model_code=STOCK_BASELINE_MODEL_CODE,
                    fund_type=STOCK_FUND_TYPE,
                )
                snapshots = list_latest_feature_snapshots(
                    session,
                    fund_type=STOCK_FUND_TYPE,
                    feature_version=STOCK_FEATURE_VERSION,
                )
                records = tuple(score_stock_feature_snapshot(snapshot, active_release) for snapshot in snapshots)
                try:
                    write_stats = upsert_forecast_results(
                        session,
                        records=records,
                        scored_at=datetime.now(UTC),
                    )
                    session.commit()
                except Exception:
                    session.rollback()
                    logger.exception(
                        "baseline_analysis.score_latest_stock_features >>> score persistence failed, attempted=%s, "
                        "active_release_id=%s",
                        len(records),
                        active_release.model_release_id if active_release else None,
                    )
                    raise
        summary = _scoring_summary(records, active_release, write_stats)
        logger.info(
            "baseline_analysis.score_latest_stock_features >>> completed, attempted=%s, scored=%s, insufficient=%s, "
            "not_applicable=%s, rejected=%s, active_release_id=%s",
            summary.attempted_count,
            summary.scored_count,
            summary.data_insufficient_count,
            summary.not_applicable_count,
            summary.model_rejected_count,
            summary.active_model_release_id,
        )
        return summary

    def run_rolling_backtest(self, config: RollingBacktestConfig) -> BacktestRunSummary:
        """在已落库净值上运行防未来泄漏的滚动回测，并创建候选发布记录。"""
        config.validate()
        engine = get_engine()
        with _analysis_lock(engine, _ANALYSIS_BACKTEST_LOCK_KEY, "stock rolling backtest"):
            with Session(engine) as session:
                points = list_stock_backtest_nav_points(session, source_code=config.source_code)
            benchmark_readiness = get_stock_benchmark_readiness(config.benchmark_id)
            if benchmark_readiness is not None:
                evaluation = _failed_backtest_evaluation(points, benchmark_readiness, config)
            else:
                benchmark_points = load_active_stock_benchmark_points(config.benchmark_id)
                try:
                    evaluation = evaluate_stock_rolling_backtest(points, config, benchmark_points)
                except ValueError as error:
                    evaluation = _failed_backtest_evaluation(points, str(error), config)

            with Session(engine) as session:
                now = datetime.now(UTC)
                run = BacktestRun(
                    fund_type=config.fund_type,
                    strategy_version=config.strategy_version,
                    feature_version=config.feature_version,
                    model_version=config.model_version,
                    window_start=evaluation.window_start,
                    window_end=evaluation.window_end,
                    train_end=evaluation.train_end,
                    validation_end=evaluation.validation_end,
                    test_start=evaluation.test_start,
                    test_end=evaluation.test_end,
                    fee_rate=config.fee_rate,
                    status="COMPLETED",
                    publication_status=evaluation.publication_status,
                    metrics=evaluation.metrics,
                    baselines=evaluation.baselines,
                    failure_reason=evaluation.failure_reason,
                    started_at=now,
                    finished_at=now,
                )
                session.add(run)
                try:
                    session.flush()
                    release = _upsert_candidate_release(session, config=config, run=run, evaluation=evaluation, now=now)
                    session.commit()
                    summary = BacktestRunSummary(
                        run_id=run.run_id,
                        status=run.status,
                        publication_status=run.publication_status,
                        failure_reason=run.failure_reason,
                        model_release_id=release.model_release_id if release else None,
                        model_release_status=release.release_status if release else None,
                    )
                except Exception:
                    session.rollback()
                    logger.exception(
                        "baseline_analysis.run_rolling_backtest >>> persistence failed, fund_type=%s, model_version=%s",
                        config.fund_type,
                        config.model_version,
                    )
                    raise
        logger.info(
            "baseline_analysis.run_rolling_backtest >>> completed, run_id=%s, publication_status=%s, "
            "release_id=%s, release_status=%s",
            summary.run_id,
            summary.publication_status,
            summary.model_release_id,
            summary.model_release_status,
        )
        return summary

    def activate_model_release(self, model_release_id: UUID, *, reason: str) -> AnalysisModelRelease:
        """仅将关联 ELIGIBLE 回测的候选发布激活，并暂停同类别旧 ACTIVE 版本。"""
        return self._transition_model_release(model_release_id, target_status="ACTIVE", reason=reason)

    def suspend_model_release(self, model_release_id: UUID, *, reason: str) -> AnalysisModelRelease:
        """暂停发布，停止该版本后续新 SCORED 结果；历史记录保持只读。"""
        return self._transition_model_release(model_release_id, target_status="SUSPENDED", reason=reason)

    def retire_model_release(self, model_release_id: UUID, *, reason: str) -> AnalysisModelRelease:
        """退役非 ACTIVE 发布；ACTIVE 版本必须先暂停，避免隐式停发。"""
        return self._transition_model_release(model_release_id, target_status="RETIRED", reason=reason)

    def _transition_model_release(
        self,
        model_release_id: UUID,
        *,
        target_status: str,
        reason: str,
    ) -> AnalysisModelRelease:
        """在单事务内执行发布状态机，锁定记录避免并发激活竞争。"""
        clean_reason = reason.strip()
        if not clean_reason:
            raise ValueError("model release transition reason must not be blank")
        engine = get_engine()
        with Session(engine) as session:
            release = session.scalar(
                select(AnalysisModelRelease)
                .where(AnalysisModelRelease.model_release_id == model_release_id)
                .with_for_update()
            )
            if release is None:
                raise ModelReleaseTransitionError(f"model release not found: {model_release_id}")
            now = datetime.now(UTC)
            if target_status == "ACTIVE":
                _assert_release_can_activate(session, release)
                previous_active = session.scalars(
                    select(AnalysisModelRelease)
                    .where(
                        AnalysisModelRelease.model_code == release.model_code,
                        AnalysisModelRelease.fund_type == release.fund_type,
                        AnalysisModelRelease.release_status == "ACTIVE",
                        AnalysisModelRelease.model_release_id != release.model_release_id,
                    )
                    .with_for_update()
                ).all()
                for previous in previous_active:
                    previous.release_status = "SUSPENDED"
                    previous.suspended_at = now
                    previous.release_reason = "已被新的已审核模型版本替代。"
                release.release_status = "ACTIVE"
                release.effective_at = now
                release.suspended_at = None
                release.release_reason = clean_reason
            elif target_status == "SUSPENDED":
                if release.release_status not in {"ACTIVE", "ELIGIBLE"}:
                    raise ModelReleaseTransitionError(
                        f"cannot suspend model release in status={release.release_status}"
                    )
                release.release_status = "SUSPENDED"
                release.suspended_at = now
                release.release_reason = clean_reason
            elif target_status == "RETIRED":
                if release.release_status == "ACTIVE":
                    raise ModelReleaseTransitionError("active model release must be suspended before retirement")
                if release.release_status == "RETIRED":
                    raise ModelReleaseTransitionError("model release is already retired")
                release.release_status = "RETIRED"
                release.release_reason = clean_reason
            else:
                raise ValueError(f"unsupported model release target_status={target_status}")
            try:
                session.commit()
            except Exception:
                session.rollback()
                logger.exception(
                    "baseline_analysis._transition_model_release >>> transition failed, release_id=%s, target=%s",
                    model_release_id,
                    target_status,
                )
                raise
            session.refresh(release)
        logger.info(
            "baseline_analysis._transition_model_release >>> transition completed, release_id=%s, status=%s",
            release.model_release_id,
            release.release_status,
        )
        return release


def score_stock_feature_snapshot(
    snapshot: FeatureSnapshot,
    active_release: AnalysisModelRelease | None,
) -> ForecastResultUpsert:
    """将一条特征快照转换为符合状态约束的基线评分记录。"""
    model_version = active_release.model_version if active_release else STOCK_BASELINE_MODEL_VERSION
    if snapshot.eligibility_status == "NOT_APPLICABLE":
        return _non_scored_record(
            snapshot,
            model_version=model_version,
            score_status="NOT_APPLICABLE",
            explanation="该基金类别不在当前股票型基线试点范围内，未生成方向性评分。",
        )
    if snapshot.eligibility_status != "SCORABLE":
        return _non_scored_record(
            snapshot,
            model_version=model_version,
            score_status="DATA_INSUFFICIENT",
            explanation=f"特征快照不满足评分条件：{snapshot.unavailable_reason or '数据完整度不足'}。",
        )
    if active_release is None:
        return _non_scored_record(
            snapshot,
            model_version=model_version,
            score_status="MODEL_REJECTED",
            explanation="当前基线模型没有 ACTIVE 发布版本，未生成方向性评分。",
        )
    if active_release.feature_version != snapshot.feature_version:
        return _non_scored_record(
            snapshot,
            model_version=model_version,
            score_status="MODEL_REJECTED",
            explanation="ACTIVE 模型的特征版本与当前快照不兼容，未生成方向性评分。",
        )

    return_20d = _payload_decimal(snapshot.feature_payload, "return_20d")
    volatility_20d = _payload_decimal(snapshot.feature_payload, "volatility_20d")
    max_drawdown_60d = _payload_decimal(snapshot.feature_payload, "max_drawdown_60d")
    if return_20d is None or volatility_20d is None or max_drawdown_60d is None:
        return _non_scored_record(
            snapshot,
            model_version=model_version,
            score_status="DATA_INSUFFICIENT",
            explanation="可评分特征缺少 20 日收益、波动率或最大回撤，未生成方向性评分。",
        )

    up_probability = _baseline_up_probability(return_20d)
    direction = _direction_from_probability(up_probability)
    confidence = _baseline_confidence(up_probability)
    risk_level = _risk_level(volatility_20d, max_drawdown_60d)
    explanation = (
        "股票型 20 个交易日动量基线的统计输出："
        f"20 日收益={_decimal_text(return_20d)}，20 日波动率={_decimal_text(volatility_20d)}，"
        f"60 日最大回撤={_decimal_text(max_drawdown_60d)}。"
        "该结果仅用于已发布模型的可回放验证，不构成交易指令或收益承诺。"
    )
    return _forecast_record(
        snapshot,
        model_version=model_version,
        model_release_id=active_release.model_release_id,
        score_status="SCORED",
        direction=direction,
        directional_probability=up_probability,
        confidence=confidence,
        risk_level=risk_level,
        max_drawdown_estimate=max_drawdown_60d.quantize(_DRAWDOWN_QUANTUM, rounding=ROUND_HALF_UP),
        explanation=explanation,
    )


def evaluate_stock_rolling_backtest(
    points: tuple[BacktestNavPoint, ...],
    config: RollingBacktestConfig,
    benchmark_points: tuple[BenchmarkNavPoint, ...] = (),
) -> RollingBacktestEvaluation:
    """使用扩展训练、验证和独立测试窗口评估固定动量基线，不读取未来输入。"""
    config.validate()
    observations = _build_backtest_observations(points, config.target_horizon_trading_days)
    if not observations:
        raise ValueError("INSUFFICIENT_NAV_HISTORY_FOR_BACKTEST")
    folds = _build_rolling_folds(observations, config)
    if not folds:
        raise ValueError("INSUFFICIENT_DATES_FOR_ROLLING_SPLIT")

    fold_metrics: list[dict[str, float]] = []
    baseline_results: list[dict[str, float]] = []
    benchmark_values_by_date = {point.nav_date: point.closing_value for point in benchmark_points}
    all_test_samples = 0
    for fold in folds:
        test_observations = tuple(
            observation
            for observation in observations
            if fold.test_start <= observation.signal_date <= fold.test_end
        )
        if not test_observations:
            continue
        fold_metrics.append(_evaluate_test_observations(test_observations, config))
        baseline_results.append(_evaluate_baselines(test_observations, benchmark_values_by_date))
        all_test_samples += len(test_observations)
    if not fold_metrics:
        raise ValueError("NO_TEST_OBSERVATIONS_IN_ROLLING_SPLIT")

    metrics = _aggregate_metric_sets(fold_metrics)
    metrics.update(
        {
            "sample_count": all_test_samples,
            "rolling_fold_count": len(fold_metrics),
            "target_horizon_trading_days": config.target_horizon_trading_days,
            "fee_rate": _decimal_text(config.fee_rate),
            "data_cutoff": folds[-1].test_end.isoformat(),
            "train_sample_end": folds[-1].train_end.isoformat(),
            "validation_sample_end": folds[-1].validation_end.isoformat(),
        }
    )
    baselines = _aggregate_baseline_sets(baseline_results, config)
    failure_reasons = _publication_failure_reasons(metrics, baselines, config)
    return RollingBacktestEvaluation(
        window_start=min(observation.signal_date for observation in observations),
        window_end=folds[-1].test_end,
        train_end=folds[-1].train_end,
        validation_end=folds[-1].validation_end,
        test_start=folds[-1].test_start,
        test_end=folds[-1].test_end,
        metrics=metrics,
        baselines=baselines,
        publication_status="ELIGIBLE" if not failure_reasons else "INELIGIBLE",
        failure_reason="; ".join(failure_reasons) if failure_reasons else None,
    )


def _build_backtest_observations(
    points: tuple[BacktestNavPoint, ...],
    horizon: int,
) -> tuple[BacktestObservation, ...]:
    """仅用信号日及之前净值计算动量，未来净值只作为离线评估标签。"""
    points_by_fund: dict[str, list[BacktestNavPoint]] = defaultdict(list)
    for point in points:
        points_by_fund[point.fund_code].append(point)
    observations: list[BacktestObservation] = []
    for fund_code, fund_points in points_by_fund.items():
        ordered = sorted(fund_points, key=lambda point: point.nav_date)
        values = _select_consistent_nav_values(ordered)
        if len(values) <= horizon * 2:
            continue
        for index in range(horizon, len(values) - horizon):
            signal_value = values[index]
            prior_value = values[index - horizon]
            future_value = values[index + horizon]
            if signal_value <= 0 or prior_value <= 0 or future_value <= 0:
                continue
            previous_return = signal_value / prior_value - Decimal("1")
            realized_return = future_value / signal_value - Decimal("1")
            up_probability = _baseline_up_probability(previous_return)
            direction = _direction_from_probability(up_probability)
            observations.append(
                BacktestObservation(
                    fund_code=fund_code,
                    signal_date=ordered[index].nav_date,
                    future_date=ordered[index + horizon].nav_date,
                    signal_nav=signal_value,
                    future_nav=future_value,
                    previous_return=previous_return,
                    realized_return=realized_return,
                    direction=direction,
                )
            )
    return tuple(sorted(observations, key=lambda item: (item.signal_date, item.fund_code)))


def _select_consistent_nav_values(points: list[BacktestNavPoint]) -> tuple[Decimal, ...]:
    """与特征构建保持一致：累计净值任一缺失时整段回退单位净值。"""
    accumulated_values = tuple(point.accumulated_nav for point in points)
    if all(value is not None and value > 0 for value in accumulated_values):
        return tuple(value for value in accumulated_values if value is not None)
    return tuple(point.unit_nav for point in points)


def _build_rolling_folds(
    observations: tuple[BacktestObservation, ...],
    config: RollingBacktestConfig,
) -> tuple[RollingFold, ...]:
    """创建扩展训练窗口；每个测试段严格晚于验证段和训练段。"""
    dates = tuple(sorted({observation.signal_date for observation in observations}))
    required_dates = config.min_train_dates + config.validation_dates + config.test_dates
    if len(dates) < required_dates:
        return ()
    folds: list[RollingFold] = []
    test_start_index = config.min_train_dates + config.validation_dates
    while test_start_index + config.test_dates <= len(dates):
        folds.append(
            RollingFold(
                train_end=dates[test_start_index - config.validation_dates - 1],
                validation_end=dates[test_start_index - 1],
                test_start=dates[test_start_index],
                test_end=dates[test_start_index + config.test_dates - 1],
            )
        )
        test_start_index += config.rolling_step_dates
    return tuple(folds)


def _evaluate_test_observations(
    observations: tuple[BacktestObservation, ...],
    config: RollingBacktestConfig,
) -> dict[str, float]:
    """计算模型测试收益、回撤、波动、命中率和换手；费用只在有暴露时扣除。"""
    strategy_returns = [
        float(observation.realized_return - config.fee_rate) if observation.direction == "UP" else 0.0
        for observation in observations
    ]
    hit_count = sum(
        (observation.direction == "UP" and observation.realized_return > 0)
        or (observation.direction == "DOWN" and observation.realized_return <= 0)
        or (observation.direction == "NEUTRAL" and abs(observation.realized_return) <= config.fee_rate)
        for observation in observations
    )
    horizon = config.target_horizon_trading_days
    return {
        "annualized_return": _annualized_return(strategy_returns, horizon),
        "max_drawdown": _max_drawdown(strategy_returns),
        "volatility": _annualized_volatility(strategy_returns, horizon),
        "hit_rate": hit_count / len(observations),
        "turnover": sum(observation.direction == "UP" for observation in observations) / len(observations),
    }


def _evaluate_baselines(
    observations: tuple[BacktestObservation, ...],
    benchmark_values_by_date: dict[date, Decimal],
) -> dict[str, float]:
    """以同一独立测试窗口记录长期持有、定投和同日期业绩基准对照。"""
    by_fund: dict[str, list[BacktestObservation]] = defaultdict(list)
    for observation in observations:
        by_fund[observation.fund_code].append(observation)
    long_hold_returns: list[float] = []
    dca_returns: list[float] = []
    benchmark_returns: list[float] = []
    for fund_observations in by_fund.values():
        ordered = sorted(fund_observations, key=lambda observation: observation.signal_date)
        start_value = ordered[0].signal_nav
        end_value = ordered[-1].future_nav
        if start_value > 0 and end_value > 0:
            long_hold_returns.append(float(end_value / start_value - Decimal("1")))
            contribution_returns = tuple(
                float(end_value / observation.signal_nav - Decimal("1"))
                for observation in ordered
                if observation.signal_nav > 0
            )
            if contribution_returns:
                dca_returns.append(fmean(contribution_returns))
    for observation in observations:
        benchmark_start = benchmark_values_by_date.get(observation.signal_date)
        benchmark_end = benchmark_values_by_date.get(observation.future_date)
        if benchmark_start is not None and benchmark_end is not None and benchmark_start > 0 and benchmark_end > 0:
            benchmark_returns.append(float(benchmark_end / benchmark_start - Decimal("1")))
    return {
        "long_hold_result": fmean(long_hold_returns) if long_hold_returns else math.nan,
        "dca_result": fmean(dca_returns) if dca_returns else math.nan,
        "benchmark_result": fmean(benchmark_returns) if benchmark_returns else math.nan,
        "benchmark_sample_count": float(len(benchmark_returns)),
        "benchmark_expected_sample_count": float(len(observations)),
    }


def _aggregate_metric_sets(metric_sets: list[dict[str, float]]) -> dict[str, object]:
    """以滚动折为单位汇总模型指标，保留足够精度供 JSON 展示和审核。"""
    return {
        key: _json_number(fmean(metric_set[key] for metric_set in metric_sets))
        for key in ("annualized_return", "max_drawdown", "volatility", "hit_rate", "turnover")
    }


def _aggregate_baseline_sets(
    baseline_sets: list[dict[str, float]],
    config: RollingBacktestConfig,
) -> dict[str, object]:
    """合并对照结果；缺少已授权基准时保留明确的不可准入标记。"""
    long_hold_values = [item["long_hold_result"] for item in baseline_sets if math.isfinite(item["long_hold_result"])]
    dca_values = [item["dca_result"] for item in baseline_sets if math.isfinite(item["dca_result"])]
    benchmark_values = [item["benchmark_result"] for item in baseline_sets if math.isfinite(item["benchmark_result"])]
    benchmark_sample_count = sum(item["benchmark_sample_count"] for item in baseline_sets)
    benchmark_expected_sample_count = sum(item["benchmark_expected_sample_count"] for item in baseline_sets)
    benchmark_coverage = (
        benchmark_sample_count / benchmark_expected_sample_count if benchmark_expected_sample_count else 0.0
    )
    benchmark_status = "NOT_CONFIGURED"
    if config.benchmark_id is not None:
        benchmark_status = (
            "AVAILABLE"
            if benchmark_values and benchmark_coverage >= float(config.min_benchmark_coverage)
            else "DATA_INSUFFICIENT"
        )
    return {
        "benchmark_id": config.benchmark_id,
        "benchmark_result": _json_number(fmean(benchmark_values)) if benchmark_status == "AVAILABLE" else None,
        "benchmark_status": benchmark_status,
        "benchmark_coverage": _json_number(benchmark_coverage),
        "benchmark_sample_count": int(benchmark_sample_count),
        "benchmark_expected_sample_count": int(benchmark_expected_sample_count),
        "long_hold_result": _json_number(fmean(long_hold_values)) if long_hold_values else None,
        "dca_result": _json_number(fmean(dca_values)) if dca_values else None,
        "comparison_basis": (
            "独立测试窗口内各基金等权汇总；长期持有、定投和同日期业绩基准均为实验对照，不代表个人持仓。"
        ),
    }


def _publication_failure_reasons(
    metrics: dict[str, object],
    baselines: dict[str, object],
    config: RollingBacktestConfig,
) -> tuple[str, ...]:
    """将未满足的发布条件完整落库，默认不因缺失基准而放行。"""
    reasons: list[str] = []
    sample_count = int(metrics["sample_count"])
    hit_rate = _as_float(metrics["hit_rate"])
    strategy_return = _as_float(metrics["annualized_return"])
    long_hold_return = _as_float(baselines["long_hold_result"])
    dca_return = _as_float(baselines["dca_result"])
    benchmark_return = _as_float(baselines["benchmark_result"])
    if sample_count < config.min_test_samples:
        reasons.append(f"TEST_SAMPLE_SHORTAGE: observed={sample_count}, required={config.min_test_samples}")
    if baselines["benchmark_status"] == "NOT_CONFIGURED":
        reasons.append("BENCHMARK_NOT_CONFIGURED")
    elif baselines["benchmark_status"] != "AVAILABLE":
        reasons.append("BENCHMARK_DATA_INSUFFICIENT")
    if hit_rate is None or hit_rate < float(config.min_hit_rate):
        reasons.append(f"HIT_RATE_BELOW_THRESHOLD: required={_decimal_text(config.min_hit_rate)}")
    if strategy_return is None or long_hold_return is None or dca_return is None or benchmark_return is None:
        reasons.append("BASELINE_METRICS_UNAVAILABLE")
    elif strategy_return <= max(long_hold_return, dca_return, benchmark_return):
        reasons.append("STRATEGY_NOT_BETTER_THAN_BASELINES")
    return tuple(reasons)


def _failed_backtest_evaluation(
    points: tuple[BacktestNavPoint, ...],
    failure_reason: str,
    config: RollingBacktestConfig,
) -> RollingBacktestEvaluation:
    """数据不足等可预期失败也写入可追溯 INELIGIBLE 回测记录。"""
    available_dates = sorted({point.nav_date for point in points})
    fallback_date = available_dates[0] if available_dates else date(2000, 1, 1)
    window_end = available_dates[-1] if available_dates else date(2000, 1, 2)
    if fallback_date >= window_end:
        fallback_date = date(2000, 1, 1)
        window_end = date(2000, 1, 2)
    return RollingBacktestEvaluation(
        window_start=fallback_date,
        window_end=window_end,
        train_end=None,
        validation_end=None,
        test_start=None,
        test_end=None,
        metrics={
            "sample_count": 0,
            "annualized_return": None,
            "max_drawdown": None,
            "volatility": None,
            "hit_rate": None,
            "turnover": None,
            "data_cutoff": window_end.isoformat(),
        },
        baselines={
            "benchmark_id": config.benchmark_id,
            "benchmark_result": None,
            "benchmark_status": "NOT_CONFIGURED" if config.benchmark_id is None else "DATA_INSUFFICIENT",
            "benchmark_coverage": None,
            "benchmark_sample_count": 0,
            "benchmark_expected_sample_count": 0,
            "long_hold_result": None,
            "dca_result": None,
        },
        publication_status="INELIGIBLE",
        failure_reason=failure_reason,
    )


def _upsert_candidate_release(
    session: Session,
    *,
    config: RollingBacktestConfig,
    run: BacktestRun,
    evaluation: RollingBacktestEvaluation,
    now: datetime,
) -> AnalysisModelRelease | None:
    """回测只产生 DRAFT 或 ELIGIBLE 候选，绝不在任务内自动激活模型。"""
    release = session.scalar(
        select(AnalysisModelRelease)
        .where(
            AnalysisModelRelease.model_code == config.model_code,
            AnalysisModelRelease.model_version == config.model_version,
            AnalysisModelRelease.fund_type == config.fund_type,
        )
        .with_for_update()
    )
    candidate_status = "ELIGIBLE" if evaluation.publication_status == "ELIGIBLE" else "DRAFT"
    release_reason = evaluation.failure_reason or "回测满足候选准入，仍需系统管理员审核激活。"
    if release is None:
        release = AnalysisModelRelease(
            model_code=config.model_code,
            model_version=config.model_version,
            feature_version=config.feature_version,
            fund_type=config.fund_type,
            backtest_run_id=run.run_id,
            release_status=candidate_status,
            release_reason=release_reason,
            config_hash=config.config_hash,
        )
        session.add(release)
        return release
    if release.release_status == "ACTIVE":
        logger.warning(
            "baseline_analysis._upsert_candidate_release >>> active version retained, model_code=%s, model_version=%s",
            config.model_code,
            config.model_version,
        )
        return release
    if release.release_status in {"SUSPENDED", "RETIRED"}:
        logger.warning(
            "baseline_analysis._upsert_candidate_release >>> non-reusable release retained, release_id=%s, status=%s",
            release.model_release_id,
            release.release_status,
        )
        return release
    release.backtest_run_id = run.run_id
    release.feature_version = config.feature_version
    release.release_status = candidate_status
    release.release_reason = release_reason
    release.config_hash = config.config_hash
    release.effective_at = None
    release.suspended_at = None
    release.wide_updated_at = now
    return release


def _assert_release_can_activate(session: Session, release: AnalysisModelRelease) -> None:
    """在激活前双重校验候选状态及其关联回测状态。"""
    if release.release_status != "ELIGIBLE":
        raise ModelReleaseTransitionError(f"only ELIGIBLE release can activate, actual={release.release_status}")
    backtest_run = find_backtest_run(session, release.backtest_run_id)
    if backtest_run is None:
        raise ModelReleaseTransitionError("linked backtest run is missing")
    if backtest_run.status != "COMPLETED" or backtest_run.publication_status != "ELIGIBLE":
        raise ModelReleaseTransitionError("linked backtest run is not eligible for publication")


def _scoring_summary(
    records: tuple[ForecastResultUpsert, ...],
    active_release: AnalysisModelRelease | None,
    write_stats: ForecastWriteStats,
) -> BaselineScoringSummary:
    """按评分状态汇总本次持久化结果。"""
    return BaselineScoringSummary(
        status="COMPLETED",
        fund_type=STOCK_FUND_TYPE,
        model_version=active_release.model_version if active_release else STOCK_BASELINE_MODEL_VERSION,
        active_model_release_id=active_release.model_release_id if active_release else None,
        attempted_count=len(records),
        scored_count=sum(record.score_status == "SCORED" for record in records),
        data_insufficient_count=sum(record.score_status == "DATA_INSUFFICIENT" for record in records),
        not_applicable_count=sum(record.score_status == "NOT_APPLICABLE" for record in records),
        model_rejected_count=sum(record.score_status == "MODEL_REJECTED" for record in records),
        created_count=write_stats.created_count,
        updated_count=write_stats.updated_count,
        skipped_count=write_stats.skipped_count,
    )


def _non_scored_record(
    snapshot: FeatureSnapshot,
    *,
    model_version: str,
    score_status: str,
    explanation: str,
) -> ForecastResultUpsert:
    """构造没有方向、概率和置信度的合法降级评分。"""
    return _forecast_record(
        snapshot,
        model_version=model_version,
        model_release_id=None,
        score_status=score_status,
        direction=None,
        directional_probability=None,
        confidence=None,
        risk_level=None,
        max_drawdown_estimate=None,
        explanation=explanation,
    )


def _forecast_record(
    snapshot: FeatureSnapshot,
    *,
    model_version: str,
    model_release_id: UUID | None,
    score_status: str,
    direction: str | None,
    directional_probability: Decimal | None,
    confidence: Decimal | None,
    risk_level: str | None,
    max_drawdown_estimate: Decimal | None,
    explanation: str,
) -> ForecastResultUpsert:
    """生成结果哈希，确保相同输入与发布状态的重复任务可跳过。"""
    payload = {
        "feature_id": str(snapshot.feature_id),
        "fund_code": snapshot.fund_code,
        "as_of_date": snapshot.as_of_date.isoformat(),
        "model_version": model_version,
        "feature_version": snapshot.feature_version,
        "model_release_id": str(model_release_id) if model_release_id else None,
        "score_status": score_status,
        "direction": direction,
        "directional_probability": _decimal_text(directional_probability),
        "confidence": _decimal_text(confidence),
        "risk_level": risk_level,
        "max_drawdown_estimate": _decimal_text(max_drawdown_estimate),
        "explanation": explanation,
    }
    return ForecastResultUpsert(
        feature_id=snapshot.feature_id,
        fund_code=snapshot.fund_code,
        as_of_date=snapshot.as_of_date,
        model_version=model_version,
        feature_version=snapshot.feature_version,
        model_release_id=model_release_id,
        score_status=score_status,
        direction=direction,
        directional_probability=directional_probability,
        confidence=confidence,
        risk_level=risk_level,
        max_drawdown_estimate=max_drawdown_estimate,
        explanation=explanation,
        result_hash=_stable_hash(payload),
    )


def _payload_decimal(payload: dict[str, object], key: str) -> Decimal | None:
    """只读取固定 metrics 字段；无效、缺失或非有限数值保持为空。"""
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    raw_value = metrics.get(key)
    if raw_value is None or isinstance(raw_value, bool):
        return None
    try:
        value = Decimal(str(raw_value))
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() else None


def _baseline_up_probability(return_20d: Decimal) -> Decimal:
    """将已观测 20 日收益映射为有界概率，避免把收益幅度当作确定性结论。"""
    raw_probability = Decimal("0.5") + return_20d * Decimal("2")
    return min(Decimal("0.9500"), max(Decimal("0.0500"), raw_probability)).quantize(
        _PROBABILITY_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def _baseline_confidence(up_probability: Decimal) -> Decimal:
    """置信度来自概率偏离中性的位置，最小保留 0.5 以避免伪造高确定性。"""
    confidence = Decimal("0.5") + abs(up_probability - Decimal("0.5"))
    return min(Decimal("0.9500"), confidence).quantize(_PROBABILITY_QUANTUM, rounding=ROUND_HALF_UP)


def _direction_from_probability(up_probability: Decimal) -> str:
    """在固定阈值下归类方向，阈值中间区间显式保持中性。"""
    if up_probability >= Decimal("0.5500"):
        return "UP"
    if up_probability <= Decimal("0.4500"):
        return "DOWN"
    return "NEUTRAL"


def _risk_level(volatility_20d: Decimal, max_drawdown_60d: Decimal) -> str:
    """按已观察波动和回撤分层，不把它表达为个人风险承受能力建议。"""
    if volatility_20d >= Decimal("0.0300") or max_drawdown_60d <= Decimal("-0.1500"):
        return "HIGH"
    if volatility_20d >= Decimal("0.0150") or max_drawdown_60d <= Decimal("-0.0800"):
        return "MEDIUM"
    return "LOW"


def _annualized_return(returns: list[float], horizon: int) -> float:
    """将平均持有期收益年化；遇到不可年化值时保持 NaN 供上层拒绝发布。"""
    if not returns:
        return math.nan
    average_return = fmean(returns)
    if average_return <= -1:
        return math.nan
    return (1 + average_return) ** (252 / horizon) - 1


def _annualized_volatility(returns: list[float], horizon: int) -> float:
    """以测试期样本标准差估算年化波动率。"""
    if len(returns) < 2:
        return math.nan
    return pstdev(returns) * math.sqrt(252 / horizon)


def _max_drawdown(returns: list[float]) -> float:
    """由按时间排序的测试期策略收益计算累计最大回撤。"""
    cumulative = 1.0
    peak = 1.0
    maximum_drawdown = 0.0
    for period_return in returns:
        cumulative *= 1 + period_return
        peak = max(peak, cumulative)
        if peak > 0:
            maximum_drawdown = min(maximum_drawdown, cumulative / peak - 1)
    return maximum_drawdown


def _as_float(value: object) -> float | None:
    """将 JSON 指标安全转换为有限浮点数，空值和 NaN 不参与发布判断。"""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_number(value: float) -> float | None:
    """JSON 不保存 NaN 或 Infinity，避免将计算异常伪装成有效指标。"""
    return round(value, 8) if math.isfinite(value) else None


def _stable_hash(payload: dict[str, object]) -> str:
    """为可重放的业务输入生成稳定 SHA-256 内容哈希。"""
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _decimal_text(value: Decimal | None) -> str | None:
    """JSON 和解释文本使用无科学计数法的十进制字符串。"""
    return format(value, "f") if value is not None else None


@contextmanager
def _analysis_lock(engine: Engine, lock_key: int, operation: str) -> Iterator[None]:
    """使用 PostgreSQL 会话级咨询锁串行化单类别评分或回测任务。"""
    with engine.connect() as connection:
        acquired = bool(connection.scalar(select(func.pg_try_advisory_lock(lock_key))))
        if not acquired:
            raise AnalysisRunInProgressError(f"{operation} is already running")
        try:
            yield
        finally:
            released = bool(connection.scalar(select(func.pg_advisory_unlock(lock_key))))
            if not released:
                logger.error(
                    "baseline_analysis._analysis_lock >>> advisory lock release failed, operation=%s",
                    operation,
                )
