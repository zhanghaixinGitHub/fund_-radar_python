"""随API进程运行的净值补拉检查；不依赖未启动的Celery，数据库锁防止多实例重复取数。"""

from threading import Event, Thread

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.tushare_fund_sync import MarketNavIncrementalInProgressError, TushareFundSyncService

logger = get_logger(__name__)


def repair_once():
    """每轮重新扫描缺口，重试时刻与配置错误状态从数据库恢复；实例退出不丢待补日期。"""
    service = TushareFundSyncService()
    try:
        return service.sync_market_nav_incremental(automatic=True)
    finally:
        service.close()


class NavRepairScheduler:
    def __init__(self):
        self._stop = Event()
        self._thread = None

    def start(self):
        if not get_settings().tushare_market_incremental_enabled:
            return
        self._thread = Thread(target=self._run, name="nav-repair", daemon=True)
        self._thread.start()
        logger.info("nav_repair_scheduler.start >>> 净值自动补拉已启动，每轮结束30分钟后检查")

    def _run(self):
        delay = 30
        while not self._stop.wait(delay):
            try:
                repair_once()
            except MarketNavIncrementalInProgressError:
                logger.info("nav_repair_scheduler._run >>> 已有净值同步执行，下一轮继续检查")
            except Exception:
                logger.exception("nav_repair_scheduler._run >>> 自动补拉未完成，保留持久状态等待后续检查")
            delay = 1800

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
