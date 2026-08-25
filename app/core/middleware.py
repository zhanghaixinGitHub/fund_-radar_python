"""内部服务调用的追踪标识传递机制。"""

from contextvars import ContextVar
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

TRACE_ID_HEADER = "X-Trace-Id"
REQUEST_ID_HEADER = "X-Request-Id"
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    """返回当前请求的追踪标识；未处于请求上下文时临时生成一个标识。"""
    trace_id = _trace_id.get()
    return trace_id or str(uuid4())


class TraceIdMiddleware(BaseHTTPMiddleware):
    """将请求关联标识保存到上下文，并在响应头中返回给调用方。"""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """优先复用 X-Trace-Id 或 X-Request-Id；请求结束后清理上下文，避免串请求。"""
        trace_id = request.headers.get(TRACE_ID_HEADER) or request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
        token = _trace_id.set(trace_id)
        try:
            response = await call_next(request)
            response.headers[TRACE_ID_HEADER] = trace_id
            return response
        finally:
            _trace_id.reset(token)
