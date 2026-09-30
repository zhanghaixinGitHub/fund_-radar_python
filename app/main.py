"""基金雷达 AI 内部服务的 FastAPI 应用入口。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.middleware import TraceIdMiddleware
from app.services.sync_jobs import close_sync_job_manager

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """管理进程级资源的启停生命周期，不在启动阶段加载任何行情或外部数据。"""
    logger.info("main.lifespan >>> FastAPI AI service started")
    # 净值补拉、002112 输入维护和资料补齐统一由同步中心手动触发。
    # 启动服务不再创建自动取数线程，避免与每日一键同步重复执行。
    logger.info("main.lifespan >>> 净值补拉、002112 输入维护和资料补齐仅手动触发，未启动定时采集")
    try:
        yield
    finally:
        close_sync_job_manager()
    logger.info("main.lifespan >>> FastAPI AI service stopped")


def create_application() -> FastAPI:
    """创建仅供 Java 核心服务调用的 FastAPI 应用，并注册日志、中间件和内部路由。"""
    settings = get_settings()
    configure_logging(settings.log_level)

    application = FastAPI(
        title="Fund Radar AI Internal API",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    application.add_middleware(TraceIdMiddleware)
    application.include_router(api_router, prefix="/internal/v1")
    return application


app = create_application()
