"""汇总仅供 Java 核心服务访问的内部路由。"""

from fastapi import APIRouter

from app.api.routes.analysis import router as analysis_router
from app.api.routes.cash_reinvestment_batch import router as cash_reinvestment_batch_router
from app.api.routes.cash_reinvestment_samples import router as cash_reinvestment_samples_router
from app.api.routes.events import router as events_router
from app.api.routes.features import router as features_router
from app.api.routes.funds import router as funds_router
from app.api.routes.health import router as health_router
from app.api.routes.historical_nav_calibration import router as historical_nav_calibration_router
from app.api.routes.historical_nav_evaluation import router as historical_nav_evaluation_router
from app.api.routes.historical_nav_storage import router as historical_nav_storage_router
from app.api.routes.historical_nav_training import router as historical_nav_training_router
from app.api.routes.nav_basis_audit import router as nav_basis_audit_router
from app.api.routes.signals import router as signals_router
from app.api.routes.sources import router as sources_router
from app.api.routes.trading_nav_window import router as trading_nav_window_router

"""内部 API 根路由；由应用入口统一加上 `/internal/v1` 前缀。"""
api_router = APIRouter()
api_router.include_router(health_router, tags=["system"])
api_router.include_router(funds_router, prefix="/funds", tags=["funds"])
api_router.include_router(events_router, prefix="/events", tags=["events"])
api_router.include_router(features_router, prefix="/features", tags=["features"])
api_router.include_router(historical_nav_storage_router, prefix="/features", tags=["features"])
api_router.include_router(historical_nav_evaluation_router, prefix="/features", tags=["features"])
api_router.include_router(historical_nav_training_router, prefix="/features", tags=["features"])
api_router.include_router(historical_nav_calibration_router, prefix="/features", tags=["features"])
api_router.include_router(trading_nav_window_router, prefix="/features", tags=["features"])
api_router.include_router(nav_basis_audit_router, prefix="/features", tags=["features"])
api_router.include_router(cash_reinvestment_samples_router, prefix="/features", tags=["features"])
api_router.include_router(cash_reinvestment_batch_router, prefix="/features", tags=["features"])
api_router.include_router(signals_router, prefix="/signals", tags=["signals"])
api_router.include_router(analysis_router, prefix="/analysis", tags=["analysis"])
api_router.include_router(sources_router, prefix="/sources", tags=["sources"])
