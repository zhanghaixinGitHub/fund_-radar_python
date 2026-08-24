"""Background task entry points; real collection starts only in M1."""

from datetime import UTC, datetime

from app.core.logging import get_logger
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(name="fund_ai.system.health_probe")
def health_probe() -> dict[str, str]:
    """Return a safe worker probe payload without accessing external systems."""
    logger.info("tasks.health_probe >>> worker health probe completed")
    return {"service": "fund-ai-worker", "status": "UP", "time": datetime.now(UTC).isoformat()}
