"""后台任务入口；真实采集任务只能在 M1 数据源获批后加入。"""

from datetime import UTC, datetime

from app.core.logging import get_logger
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(name="fund_ai.system.health_probe")
def health_probe() -> dict[str, str]:
    """返回安全的 Worker 探针结果，不访问数据库、模型或外部系统。"""
    logger.info("tasks.health_probe >>> worker health probe completed")
    return {"service": "fund-ai-worker", "status": "UP", "time": datetime.now(UTC).isoformat()}
