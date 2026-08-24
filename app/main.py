"""FastAPI application entry point for the internal AI service."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.middleware import TraceIdMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize and close process-scoped resources without loading market data."""
    logger.info("main.lifespan >>> FastAPI AI service started")
    yield
    logger.info("main.lifespan >>> FastAPI AI service stopped")


def create_application() -> FastAPI:
    """Create the internal-only FastAPI application."""
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
