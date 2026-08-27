"""后台任务入口；外部数据同步只能通过已登记的受控服务执行。"""

from datetime import UTC, date, datetime

from app.core.config import get_settings
from app.core.logging import get_logger
from app.integrations.tushare import TushareIntegrationError
from app.services.tushare_fund_sync import TushareFundSyncService
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
def sync_tushare_nav_daily(nav_date: str) -> dict[str, str | int | None]:
    """异步同步指定净值日；日期无效时由调用方修正，不做无意义重试。"""
    parsed_nav_date = date.fromisoformat(nav_date)
    service = TushareFundSyncService()
    try:
        return service.sync_nav_daily(parsed_nav_date).to_payload()
    finally:
        service.close()


@celery_app.task(
    name="fund_ai.tushare.sync_focused_nav_incremental",
    autoretry_for=(TushareIntegrationError,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 2},
)
def sync_focused_nav_incremental(as_of_date: str | None = None) -> dict[str, str | int | None]:
    """按每只重点基金的同源水位补齐净值；不执行全量历史回填。"""
    parsed_as_of_date = date.fromisoformat(as_of_date) if as_of_date else None
    service = TushareFundSyncService()
    try:
        return service.sync_focused_nav_incremental(
            get_settings().focused_fund_ts_codes,
            as_of_date=parsed_as_of_date,
        ).to_payload()
    finally:
        service.close()
