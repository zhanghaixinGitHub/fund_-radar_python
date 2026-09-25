"""随API进程运行的净值补拉检查；不依赖未启动的Celery，数据库锁防止多实例重复取数。"""

from threading import Event, Thread

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.fund_exposure_runtime import maintenance_once as exposure_maintenance_once
from app.services.fund_materials_sync import schedule_if_due as materials_schedule_if_due
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
            # 002112 研究输入与正式预测隔离；本地控制文件未启用时立即返回。
            # 即便净值补拉失败也可恢复已完成的行情工作，异常不影响下一轮净值维护。
            try:
                exposure_result = exposure_maintenance_once()
                if exposure_result["status"] not in {"NOT_ENABLED", "NOT_DUE", "BUSY"}:
                    logger.info("nav_repair_scheduler._run >>> 002112 输入维护状态=%s", exposure_result["status"])
            except Exception as exc:
                logger.warning("nav_repair_scheduler._run >>> 002112 输入维护暂未完成，异常类型=%s", type(exc).__name__)
            try:
                # 旧输入维护释放基金锁后，向同步中心登记到期的资料任务，不能并行重复取数。
                materials_schedule_if_due()
            except Exception:
                logger.exception("nav_repair_scheduler._run >>> 基金资料任务登记暂未完成")
            delay = 1800

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
