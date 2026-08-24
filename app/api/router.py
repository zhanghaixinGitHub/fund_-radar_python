"""Internal route aggregation."""

from fastapi import APIRouter

from app.api.routes.funds import router as funds_router
from app.api.routes.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["system"])
api_router.include_router(funds_router, prefix="/funds", tags=["funds"])
