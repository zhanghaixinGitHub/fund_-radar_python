"""系统级响应的数据契约。"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class InternalHealthResponse(BaseModel):
    """通过服务身份认证后返回的内部健康检查响应。"""

    model_config = ConfigDict(frozen=True)

    service: str
    status: str
    time: datetime
