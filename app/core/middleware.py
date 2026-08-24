"""Trace propagation for internal calls."""

from contextvars import ContextVar
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

TRACE_ID_HEADER = "X-Trace-Id"
REQUEST_ID_HEADER = "X-Request-Id"
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    """Return the current request trace identifier, generating one only if absent."""
    trace_id = _trace_id.get()
    return trace_id or str(uuid4())


class TraceIdMiddleware(BaseHTTPMiddleware):
    """Forward request IDs as trace IDs and return them to the caller."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        trace_id = request.headers.get(TRACE_ID_HEADER) or request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
        token = _trace_id.set(trace_id)
        try:
            response = await call_next(request)
            response.headers[TRACE_ID_HEADER] = trace_id
            return response
        finally:
            _trace_id.reset(token)
