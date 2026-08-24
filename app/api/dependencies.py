"""Dependencies that protect Java-to-Python internal calls."""

import hmac
from typing import Annotated

from fastapi import Header, HTTPException, status

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

ServiceTokenHeader = Annotated[str | None, Header(alias="X-Service-Token")]
OriginHeader = Annotated[str | None, Header(alias="Origin")]


async def require_service_token(service_token: ServiceTokenHeader = None, origin: OriginHeader = None) -> None:
    """Reject browser and unauthenticated access to internal API routes.

    Args:
        service_token: Service identity token sent by the Java core service.

    Raises:
        HTTPException: If the local token is missing or does not match.
    """
    expected_token = get_settings().ai_service_token.get_secret_value()
    if origin:
        logger.warning("dependencies.require_service_token >>> rejected browser-originated internal API request")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Internal API is not available to browsers.")

    if not expected_token:
        logger.error("dependencies.require_service_token >>> AI service token is not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service authentication is not configured.",
        )

    if service_token is None or not hmac.compare_digest(service_token, expected_token):
        logger.warning("dependencies.require_service_token >>> rejected internal API request")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Service authentication failed.")
