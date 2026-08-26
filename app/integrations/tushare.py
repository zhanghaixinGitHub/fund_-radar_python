"""Tushare 公募基金 HTTP 适配器。

此模块只负责外部 API 协议、最小字段请求、超时和受控重试；不写数据库，
也绝不在异常、日志或返回值中暴露 Token。
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

CATALOG_MARKETS: tuple[str, ...] = ("E", "O")
CATALOG_STATUSES: tuple[str, ...] = ("L", "D", "I")
_RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


class TushareIntegrationError(RuntimeError):
    """Tushare 协议、授权、传输或字段校验失败。

    Attributes:
        api_name: 发生错误的 Tushare 接口名称。
        retryable: 调用方是否可以在受控退避后重试。
    """

    def __init__(self, api_name: str, summary: str, *, retryable: bool = False) -> None:
        super().__init__(f"{api_name}: {summary}")
        self.api_name = api_name
        self.retryable = retryable


@dataclass(frozen=True)
class TushareFundCompany:
    """基金公司最小规范化记录，仅用于管理人名称映射。"""

    name: str
    short_name: str | None


@dataclass(frozen=True)
class TushareFundBasic:
    """基金目录接口的最小字段记录。"""

    ts_code: str
    name: str
    management: str | None
    fund_type: str | None
    found_date: date | None
    status: str | None
    market: str | None


@dataclass(frozen=True)
class TushareFundNav:
    """基金净值接口的最小字段记录。"""

    ts_code: str
    ann_date: date | None
    nav_date: date
    unit_nav: Decimal
    accumulated_nav: Decimal | None


class TushareFundClient:
    """通过 HTTPS 调用 Tushare 公募基金接口的同步客户端。

    Args:
        token: 仅从环境配置读取的 Tushare Token。
        api_url: Tushare HTTP API 地址。
        connect_timeout_seconds: 建连超时秒数。
        read_timeout_seconds: 读取超时秒数。
        max_retries: 仅对传输和服务端可恢复错误的额外重试次数。
        catalog_max_rows_per_query: 单个 `fund_basic` 分片的允许最大记录数。
        transport: 仅用于自动化测试的 HTTPX transport。

    Raises:
        ValueError: Token 为空或超时/重试参数不合法。
    """

    def __init__(
        self,
        *,
        token: str,
        api_url: str,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
        max_retries: int,
        catalog_max_rows_per_query: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not token.strip():
            raise ValueError("Tushare Token is not configured.")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative.")
        self._token = token
        self._max_retries = max_retries
        self._catalog_max_rows_per_query = catalog_max_rows_per_query
        timeout = httpx.Timeout(
            connect=connect_timeout_seconds,
            read=read_timeout_seconds,
            write=read_timeout_seconds,
            pool=connect_timeout_seconds,
        )
        self._client = httpx.Client(timeout=timeout, transport=transport, follow_redirects=False)
        self._api_url = api_url

    def close(self) -> None:
        """关闭底层 HTTP 客户端及其连接资源。"""
        self._client.close()

    def __enter__(self) -> TushareFundClient:
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        self.close()

    def list_fund_companies(self) -> tuple[TushareFundCompany, ...]:
        """读取基金公司名称和简称，用于基金目录管理人规范化。"""
        rows = self._query("fund_company", params={}, fields="name,shortname")
        return tuple(
            TushareFundCompany(
                name=_required_text(row, "name", "fund_company"), short_name=_optional_text(row.get("shortname"))
            )
            for row in rows
        )

    def list_fund_basics(self) -> tuple[TushareFundBasic, ...]:
        """按市场和存续状态分片读取基金目录，拒绝达到单片上限的结果。"""
        by_ts_code: dict[str, TushareFundBasic] = {}
        fields = "ts_code,name,management,fund_type,found_date,status,market"
        for market in CATALOG_MARKETS:
            for status in CATALOG_STATUSES:
                rows = self._query("fund_basic", params={"market": market, "status": status}, fields=fields)
                if len(rows) >= self._catalog_max_rows_per_query:
                    raise TushareIntegrationError(
                        "fund_basic",
                        f"market={market}, status={status} reached configured row limit; refusing partial catalog",
                    )
                for row in rows:
                    item = TushareFundBasic(
                        ts_code=_required_text(row, "ts_code", "fund_basic"),
                        name=_required_text(row, "name", "fund_basic"),
                        management=_optional_text(row.get("management")),
                        fund_type=_optional_text(row.get("fund_type")),
                        found_date=_optional_date(row.get("found_date"), "found_date", "fund_basic"),
                        status=_optional_text(row.get("status")),
                        market=_optional_text(row.get("market")),
                    )
                    existing = by_ts_code.get(item.ts_code)
                    if existing is not None and existing != item:
                        raise TushareIntegrationError("fund_basic", f"conflicting duplicate ts_code={item.ts_code}")
                    by_ts_code[item.ts_code] = item
        return tuple(by_ts_code[ts_code] for ts_code in sorted(by_ts_code))

    def list_nav_daily(self, nav_date: date) -> tuple[TushareFundNav, ...]:
        """按净值日期批量读取公募基金净值，不逐基金发起远程请求。"""
        rows = self._query(
            "fund_nav",
            params={"nav_date": nav_date.strftime("%Y%m%d")},
            fields="ts_code,ann_date,nav_date,unit_nav,accum_nav",
        )
        return tuple(
            TushareFundNav(
                ts_code=_required_text(row, "ts_code", "fund_nav"),
                ann_date=_optional_date(row.get("ann_date"), "ann_date", "fund_nav"),
                nav_date=_required_date(row.get("nav_date"), "nav_date", "fund_nav"),
                unit_nav=_required_decimal(row.get("unit_nav"), "unit_nav", "fund_nav"),
                accumulated_nav=_optional_decimal(row.get("accum_nav"), "accum_nav", "fund_nav"),
            )
            for row in rows
        )

    def _query(
        self, api_name: str, *, params: Mapping[str, str], fields: str
    ) -> tuple[dict[str, Any], ...]:
        payload = {"api_name": api_name, "token": self._token, "params": dict(params), "fields": fields}
        last_error: BaseException | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(self._api_url, json=payload)
                if response.status_code in _RETRYABLE_HTTP_STATUS_CODES:
                    raise TushareIntegrationError(
                        api_name, f"retryable HTTP status={response.status_code}", retryable=True
                    )
                response.raise_for_status()
                body = response.json()
                return _parse_response(api_name, body)
            except TushareIntegrationError as error:
                if not error.retryable:
                    raise
                last_error = error
            except (httpx.HTTPError, ValueError) as error:
                last_error = error
            if attempt < self._max_retries:
                time.sleep(0.25 * (2**attempt))
        raise TushareIntegrationError(api_name, _safe_error_summary(last_error), retryable=True) from last_error


def _parse_response(api_name: str, body: object) -> tuple[dict[str, Any], ...]:
    """校验 Tushare HTTP 响应的列和值结构。"""
    if not isinstance(body, dict):
        raise TushareIntegrationError(api_name, "response body is not an object")
    code = body.get("code")
    if code != 0:
        raise TushareIntegrationError(api_name, f"API code={code}; {_safe_error_summary(body.get('msg'))}")
    data = body.get("data")
    if not isinstance(data, dict):
        raise TushareIntegrationError(api_name, "response data is missing")
    fields = data.get("fields")
    items = data.get("items")
    if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
        raise TushareIntegrationError(api_name, "response fields are invalid")
    if not isinstance(items, list):
        raise TushareIntegrationError(api_name, "response items are invalid")
    records: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, list) or len(item) != len(fields):
            raise TushareIntegrationError(api_name, "response item does not match fields")
        records.append(dict(zip(fields, item, strict=True)))
    return tuple(records)


def _required_text(row: Mapping[str, Any], field_name: str, api_name: str) -> str:
    value = _optional_text(row.get(field_name))
    if value is None:
        raise TushareIntegrationError(api_name, f"required field {field_name} is blank")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_date(value: Any, field_name: str, api_name: str) -> date:
    parsed = _optional_date(value, field_name, api_name)
    if parsed is None:
        raise TushareIntegrationError(api_name, f"required field {field_name} is blank")
    return parsed


def _optional_date(value: Any, field_name: str, api_name: str) -> date | None:
    text = _optional_text(value)
    if text is None:
        return None
    try:
        return datetime.strptime(text, "%Y%m%d").date()
    except ValueError as error:
        raise TushareIntegrationError(api_name, f"field {field_name} is not YYYYMMDD") from error


def _required_decimal(value: Any, field_name: str, api_name: str) -> Decimal:
    parsed = _optional_decimal(value, field_name, api_name)
    if parsed is None:
        raise TushareIntegrationError(api_name, f"required field {field_name} is blank")
    return parsed


def _optional_decimal(value: Any, field_name: str, api_name: str) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise TushareIntegrationError(api_name, f"field {field_name} is not decimal") from error
    if not parsed.is_finite() or parsed < 0:
        raise TushareIntegrationError(api_name, f"field {field_name} must be a non-negative finite decimal")
    return parsed


def _safe_error_summary(error: object) -> str:
    """将外部错误压缩为不包含请求体和 Token 的短摘要。"""
    text = str(error or "unknown error").replace("\n", " ").replace("\r", " ")
    return text[:300]
