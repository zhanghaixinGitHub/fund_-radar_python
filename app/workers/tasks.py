"""后台任务入口；外部数据同步只能通过已登记的受控服务执行。"""

from datetime import UTC, date, datetime
from decimal import Decimal

from app.core.logging import get_logger
from app.integrations.tushare import TushareIntegrationError
from app.services.analysis_runs import execute_fund_explanation, execute_stock_rolling_backtest
from app.services.baseline_analysis import BaselineAnalysisService, RollingBacktestConfig
from app.services.stock_feature_snapshot import FeatureSnapshotBuildInProgressError, StockFeatureSnapshotService
from app.services.tushare_fund_sync import SyncOutcome, TushareFundSyncService
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(name="fund_ai.system.health_probe")
def health_probe() -> dict[str, str]:
    """返回安全的 Worker 探针结果，不访问数据库、模型或外部系统。"""
    logger.info("tasks.health_probe >>> worker health probe completed")
    return {"service": "fund-ai-worker", "status": "UP", "time": datetime.now(UTC).isoformat()}


@celery_app.task(
    name="fund_ai.tushare.sync_catalog",
    autoretry_for=(TushareIntegrationError,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 2},
)
def sync_tushare_catalog() -> dict[str, str | int | None]:
    """异步执行基金目录同步；仅可恢复的 Tushare 异常才会有限重试。"""
    service = TushareFundSyncService()
    try:
        return service.sync_catalog().to_payload()
    finally:
        service.close()


@celery_app.task(
    name="fund_ai.tushare.sync_nav_daily",
    autoretry_for=(TushareIntegrationError,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 2},
)
def sync_tushare_nav_daily(nav_date: str) -> dict[str, object]:
    """异步同步指定净值日；日期无效时由调用方修正，不做无意义重试。"""
    parsed_nav_date = date.fromisoformat(nav_date)
    service = TushareFundSyncService()
    try:
        return _with_feature_snapshot_payload(service.sync_nav_daily(parsed_nav_date))
    finally:
        service.close()


@celery_app.task(
    name="fund_ai.tushare.sync_market_nav_incremental",
    autoretry_for=(TushareIntegrationError,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 2},
)
def sync_market_nav_incremental(as_of_date: str | None = None) -> dict[str, object]:
    """按基金市场中每只启用份额的同源水位补齐净值；不执行全量历史回填。"""
    parsed_as_of_date = date.fromisoformat(as_of_date) if as_of_date else None
    service = TushareFundSyncService()
    try:
        return _with_feature_snapshot_payload(service.sync_market_nav_incremental(as_of_date=parsed_as_of_date))
    finally:
        service.close()


def _with_feature_snapshot_payload(outcome: SyncOutcome) -> dict[str, object]:
    """在来源净值成功后自动构建特征；特征失败不重复执行外部来源同步。"""
    payload: dict[str, object] = outcome.to_payload()
    try:
        summary = StockFeatureSnapshotService().build()
    except FeatureSnapshotBuildInProgressError:
        payload["feature_snapshot"] = {
            "status": "IN_PROGRESS",
            "error_code": "FEATURE_SYNC_IN_PROGRESS",
            "message": "特征快照正在由其他任务生成，来源净值同步结果已保留。",
        }
        return payload
    except Exception:
        logger.exception(
            "tasks._with_feature_snapshot_payload >>> feature build failed after source sync, sync_run_id=%s",
            outcome.sync_run_id,
        )
        payload["feature_snapshot"] = {
            "status": "FAILED",
            "error_code": "FEATURE_SNAPSHOT_BUILD_FAILED",
            "message": "来源净值同步成功，但特征快照未完成；请在同步中心手动重试。",
        }
        return payload
    if summary.status != "COMPLETED":
        payload["feature_snapshot"] = {
            "status": summary.status,
            "error_code": "FEATURE_SOURCE_NOT_READY",
            "message": "来源净值同步成功，但特征来源尚未就绪；请在同步中心手动重试。",
        }
        return payload
    payload["feature_snapshot"] = summary.to_payload()
    return payload


@celery_app.task(name="fund_ai.analysis.build_stock_feature_snapshots")
def build_stock_feature_snapshots() -> dict[str, str | int | None]:
    """手动构建 M3-G1 特征快照；不注册定时计划、不调用外部来源。"""
    return StockFeatureSnapshotService().build().to_payload()


@celery_app.task(name="fund_ai.analysis.score_stock_baseline")
def score_stock_baseline() -> dict[str, str | int | None]:
    """对最新股票型特征执行受控评分；没有 ACTIVE 模型时只写 MODEL_REJECTED。"""
    return BaselineAnalysisService().score_latest_stock_features().to_payload()


@celery_app.task(name="fund_ai.analysis.run_stock_rolling_backtest")
def run_stock_rolling_backtest(
    *,
    fee_rate: str = "0.001500",
    benchmark_id: str | None = None,
) -> dict[str, str | None]:
    """运行股票型固定基线回测；缺少已授权基准时结果明确保持不可发布。"""
    config = RollingBacktestConfig(fee_rate=Decimal(fee_rate), benchmark_id=benchmark_id)
    return BaselineAnalysisService().run_rolling_backtest(config).to_payload()


@celery_app.task(name="fund_ai.analysis.run_stock_rolling_backtest_controlled")
def run_controlled_stock_rolling_backtest(analysis_run_id: str) -> dict[str, str | None]:
    """执行已由内部控制面持久化的回测，不接受浏览器或外部来源直接触发。"""
    from uuid import UUID

    return execute_stock_rolling_backtest(UUID(analysis_run_id))


@celery_app.task(name="fund_ai.analysis.run_fund_explanation_controlled")
def run_controlled_fund_explanation(analysis_run_id: str) -> dict[str, str | None]:
    """执行由内部控制面排队的 DeepSeek 解释任务，不接受浏览器直连或自由提示词。"""
    from uuid import UUID

    return execute_fund_explanation(UUID(analysis_run_id))
