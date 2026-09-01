"""汇总仅供 Java 核心服务访问的内部路由。"""

from fastapi import APIRouter

from app.api.routes.events import router as events_router
from app.api.routes.features import router as features_router
from app.api.routes.funds import router as funds_router
from app.api.routes.health import router as health_router
from app.api.routes.signals import router as signals_router
from app.api.routes.sources import router as sources_router

"""内部 API 根路由；由应用入口统一加上 `/internal/v1` 前缀。"""
api_router = APIRouter()
api_router.include_router(health_router, tags=["system"])
api_router.include_router(funds_router, prefix="/funds", tags=["funds"])
api_router.include_router(events_router, prefix="/events", tags=["events"])
api_router.include_router(features_router, prefix="/features", tags=["features"])
api_router.include_router(signals_router, prefix="/signals", tags=["signals"])
api_router.include_router(sources_router, prefix="/sources", tags=["sources"])
