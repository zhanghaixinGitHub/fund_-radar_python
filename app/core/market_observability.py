"""模拟行情查询的请求与数据库阶段耗时；不输出凭据、SQL 或行情原文。"""

import asyncio
import logging
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock
from time import perf_counter
from typing import Any
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.middleware import get_trace_id

logger = logging.getLogger(__name__)
# 只做观测，不改变 Java 的 3 秒读取超时；提前一次告警，留下当时正在等待的阶段。
PENDING_SECONDS = 2.0
SLOW_SECONDS = 1.0
_MARKET_PATH = re.compile(r"/internal/v1/simulation/funds/([0-9]{6})")


@dataclass
class MarketObservation:
    """单请求观测数据；毫秒使用单调时钟，锁保护异步请求与同步工作线程的共享状态。"""

    trace_id: str
    fund_code: str
    started: float = field(default_factory=perf_counter)
    current_stage: str = "routing_or_worker_wait"
    failed_stage: str = ""
    stage_ms: dict[str, float] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    def snapshot(self) -> tuple[str, str, dict[str, float]]:
        with self.lock:
            return self.current_stage, self.failed_stage, dict(self.stage_ms)


_observation: ContextVar[MarketObservation | None] = ContextVar("market_observation", default=None)


@contextmanager
def market_stage(name: str) -> Iterator[None]:
    """记录一个固定名称阶段；仅行情 HTTP 请求启用，其他调用不增加日志或数据库操作。

    数据库查询阶段包括执行与结果读取；connection 阶段包括连接池等待、建连及存活检查。
    嵌套阶段包含子阶段耗时，不能把所有阶段相加当作总耗时。
    """
    observation = _observation.get()
    if observation is None:
        yield
        return
    started = perf_counter()
    with observation.lock:
        previous = observation.current_stage
        observation.current_stage = name
    try:
        yield
    except BaseException:
        with observation.lock:
            # 保留最内层失败点，避免被外层 fund_detail 覆盖。
            if not observation.failed_stage:
                observation.failed_stage = name
        raise
    finally:
        elapsed = round((perf_counter() - started) * 1000, 2)
        with observation.lock:
            observation.stage_ms[name] = observation.stage_ms.get(name, 0) + elapsed
            observation.current_stage = previous


def observe_market_read(function: Callable[..., Any]) -> Callable[..., Any]:
    """记录进入同步服务前的延迟和服务总耗时，区分请求排队与服务内查询。"""
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        observation = _observation.get()
        if observation is None:
            return function(*args, **kwargs)
        with observation.lock:
            observation.stage_ms["service_start_delay"] = round((perf_counter() - observation.started) * 1000, 2)
        try:
            with market_stage("read_market"):
                return function(*args, **kwargs)
        finally:
            with observation.lock:
                observation.current_stage = "response_serialization"
    return wrapped


def observe_market_connection(session: Any, stage: str) -> None:
    """仅在行情观测启用时提前借出本来首条查询就会用到的连接，其他读链路保持原状。"""
    if _observation.get() is not None:
        with market_stage(stage):
            session.connection()


class SimulationMarketLoggingMiddleware:
    """仅观测模拟行情 GET；记录进入、仍未结束、响应结束或失败，保留现有鉴权。

    本中间件位于 TraceIdMiddleware 内层，共用 Java 传入的请求号。response_complete
    只表示 ASGI 响应发送完成，不代表 Java 已收到；disconnect 只记录实际收到的断开事件。
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        match = _MARKET_PATH.fullmatch(scope.get("path", ""))
        if scope["type"] != "http" or scope.get("method") != "GET" or match is None:
            await self.app(scope, receive, send)
            return
        trace_id = get_trace_id()
        # 只把有限、单行的标识写入新增日志；不读取或打印请求头、查询串、请求体。
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", trace_id):
            trace_id = str(uuid4())
        observation = MarketObservation(trace_id, match[1])
        token = _observation.set(observation)
        status: int | None = None
        response_complete = False
        disconnected = False
        outcome = "incomplete"
        logger.info("SimulationMarket.http   >>> phase=start traceId=%s fundCode=%s", trace_id, match[1])

        def pending() -> None:
            stage, failed, timings = observation.snapshot()
            logger.warning(
                "SimulationMarket.http   >>> phase=pending traceId=%s fundCode=%s elapsedMs=%.2f "
                "stage=%s failedStage=%s stageMs=%s",
                trace_id, match[1], (perf_counter() - observation.started) * 1000, stage, failed, timings,
            )

        async def tracked_receive() -> Message:
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal status, response_complete
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                response_complete = True

        # 一次性计时回调；请求结束立即取消，不创建轮询线程或长期后台任务。
        timer = asyncio.get_running_loop().call_later(PENDING_SECONDS, pending)
        try:
            await self.app(scope, tracked_receive, tracked_send)
            outcome = "response_complete" if response_complete else "incomplete"
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as error:
            outcome = "failed"
            # 数据库异常文本可能包含 SQL 参数；保留原堆栈位置，只用异常类型替换消息。
            safe_error = RuntimeError(type(error).__name__)
            logger.error(
                "SimulationMarket.http   >>> phase=error traceId=%s fundCode=%s errorType=%s",
                trace_id, match[1], type(error).__name__,
                exc_info=(RuntimeError, safe_error, error.__traceback__),
            )
            raise
        finally:
            timer.cancel()
            elapsed = perf_counter() - observation.started
            stage, failed, timings = observation.snapshot()
            abnormal = elapsed >= SLOW_SECONDS or outcome != "response_complete" or (status or 0) >= 400
            level = logging.WARNING if abnormal else logging.INFO
            # 正常调度每分钟逐基金调用，INFO 只留一行总耗时；异常或 DEBUG 才展开阶段明细。
            detail = f" stage={stage} failedStage={failed} stageMs={timings}" if (
                abnormal or logger.isEnabledFor(logging.DEBUG)
            ) else ""
            logger.log(
                level,
                "SimulationMarket.http   >>> phase=end traceId=%s fundCode=%s status=%s outcome=%s "
                "elapsedMs=%.2f disconnectObserved=%s%s",
                trace_id, match[1], status, outcome, elapsed * 1000, disconnected, detail,
            )
            _observation.reset(token)
