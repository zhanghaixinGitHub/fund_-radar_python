"""保护 Java 到 Python 内部调用边界的 FastAPI 依赖项。"""

import hmac
from typing import Annotated

from fastapi import Header, HTTPException, status

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

ServiceTokenHeader = Annotated[str | None, Header(alias="X-Service-Token")]
OriginHeader = Annotated[str | None, Header(alias="Origin")]


async def require_service_token(service_token: ServiceTokenHeader = None, origin: OriginHeader = None) -> None:
    """拒绝浏览器来源和未通过服务身份验证的内部接口请求。

    Args:
        service_token: Java 核心服务通过请求头传入的服务身份令牌。
        origin: 浏览器跨域请求可能附带的来源地址；内部接口不接受该类请求。

    Raises:
        HTTPException: 未配置令牌、令牌不一致或请求来自浏览器时抛出。
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
