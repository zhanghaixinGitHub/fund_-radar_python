"""Tushare 已授权场内基金与市场参考指数 HTTP 适配器。

该模块只请求已在免费数据补齐范围中列明的结构化字段。它不写数据库、
不读取新闻或持仓，也不会在异常、日志或返回对象中泄露 Token。
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

import httpx

from app.integrations.tushare import (
    TushareIntegrationError,
    _optional_date,
    _optional_decimal,
    _optional_signed_decimal,
    _optional_text,
    _parse_response,
    _required_date,
    _required_decimal,
    _required_text,
    _safe_error_summary,
)

_RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
_FUND_DAILY_FIELDS = "ts_code,trade_date,pre_close,open,high,low,close,change,pct_chg,vol,amount"
_INDEX_BASIC_FIELDS = (
    "ts_code,name,market,publisher,category,base_date,base_point,list_date,list_point,weight_rule,desc,exp_date"
)
_INDEX_CLASSIFY_FIELDS = "index_code,industry_name,parent_code,level,industry_code,is_pub,src"
_INDEX_DAILY_FIELDS = "ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount"
_INDEX_WEIGHT_FIELDS = "index_code,con_code,trade_date,weight"


@dataclass(frozen=True)
class TushareFundDaily:
    """一条场内基金交易日行情。"""

    ts_code: str
    trade_date: date
    previous_close_price: Decimal | None
    open_price: Decimal | None
    high_price: Decimal | None
    low_price: Decimal | None
    close_price: Decimal
    change_value: Decimal | None
    change_percent: Decimal | None
    volume: Decimal | None
    amount: Decimal | None


@dataclass(frozen=True)
class TushareIndexBasic:
    """一条指数目录记录。"""

    index_code: str
    display_name: str
    market: str | None
    publisher: str | None
    category: str | None
    base_date: date | None
    list_date: date | None
    expiry_date: date | None


@dataclass(frozen=True)
class TushareIndexClassification:
    """一条指数来源分类记录。"""

    classification_code: str
    classification_name: str
    parent_classification_code: str | None
    hierarchy_level: int | None
    source_name: str | None


@dataclass(frozen=True)
class TushareIndexDaily:
    """一条指数交易日行情。"""

    index_code: str
    trade_date: date
    close_price: Decimal


@dataclass(frozen=True)
class TushareIndexWeight:
    """一条指数成分股权重记录。"""

    index_code: str
    constituent_code: str
    trade_date: date
    weight: Decimal


class TushareMarketReferenceClient:
    """调用 Tushare 场内基金与指数免费数据接口的最小客户端。

    Args:
        token: 仅从服务端配置读取的 Tushare Token。
        api_url: Tushare HTTPS API 地址。
        connect_timeout_seconds: 建连超时秒数。
        read_timeout_seconds: 读取和写入超时秒数。
        max_retries: 可恢复 HTTP 或网络异常的最大重试次数。
        catalog_max_rows_per_query: 单个指数目录市场分片允许的最大行数。
        max_rows_per_query: 场内日线、指数日线及权重单请求的允许最大行数。
        transport: 仅自动化测试使用的 HTTPX transport。
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
        max_rows_per_query: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not token.strip():
            raise ValueError("Tushare Token is not configured.")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative.")
        if catalog_max_rows_per_query < 1 or max_rows_per_query < 1:
            raise ValueError("Tushare response limits must be positive.")
        self._token = token
        self._api_url = api_url
        self._max_retries = max_retries
        self._catalog_max_rows_per_query = catalog_max_rows_per_query
        self._max_rows_per_query = max_rows_per_query
        timeout = httpx.Timeout(
            connect=connect_timeout_seconds,
            read=read_timeout_seconds,
            write=read_timeout_seconds,
            pool=connect_timeout_seconds,
        )
        self._client = httpx.Client(timeout=timeout, transport=transport, follow_redirects=False)

    def close(self) -> None:
        """关闭底层 HTTP 连接池。"""
        self._client.close()

    def __enter__(self) -> TushareMarketReferenceClient:
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        self.close()

    def list_fund_exchange_daily(
        self, ts_code: str, *, start_date: date, end_date: date
    ) -> tuple[TushareFundDaily, ...]:
        """读取明确场内基金代码和日期窗口的日线；不会猜测交易代码。"""
        if start_date > end_date:
            raise ValueError("start_date must not be after end_date.")
        rows = self._query(
            "fund_daily",
            params={
                "ts_code": ts_code,
                "start_date": start_date.strftime("%Y%m%d"),
                "end_date": end_date.strftime("%Y%m%d"),
            },
            fields=_FUND_DAILY_FIELDS,
        )
        self._ensure_not_truncated("fund_daily", rows, ts_code)
        records = tuple(_to_fund_daily(row) for row in rows)
        if any(item.ts_code != ts_code for item in records):
            raise TushareIntegrationError("fund_daily", f"ts_code={ts_code} response contains another fund")
        return records

    def list_index_basics(self, markets: tuple[str, ...]) -> tuple[TushareIndexBasic, ...]:
        """按市场分片读取指数目录，达到来源上限时拒绝写入不完整分片。"""
        if not markets or len(set(markets)) != len(markets):
            raise ValueError("markets must be non-empty and unique.")
        records_by_code: dict[str, TushareIndexBasic] = {}
        for market in markets:
            rows = self._query("index_basic", params={"market": market}, fields=_INDEX_BASIC_FIELDS)
            if len(rows) >= self._catalog_max_rows_per_query:
                raise TushareIntegrationError(
                    "index_basic",
                    f"market={market} reached configured row limit; refusing partial index catalog",
                )
            for row in rows:
                item = _to_index_basic(row)
                existing = records_by_code.get(item.index_code)
                if existing is not None and existing != item:
                    raise TushareIntegrationError("index_basic", f"conflicting duplicate index_code={item.index_code}")
                records_by_code[item.index_code] = item
        return tuple(records_by_code[index_code] for index_code in sorted(records_by_code))

    def list_index_classifications(self) -> tuple[TushareIndexClassification, ...]:
        """读取来源公开的指数分类层级，不将其解释为基金行业暴露。"""
        rows = self._query("index_classify", params={}, fields=_INDEX_CLASSIFY_FIELDS)
        self._ensure_not_truncated("index_classify", rows, "GLOBAL")
        records_by_code: dict[str, TushareIndexClassification] = {}
        for row in rows:
            item = _to_index_classification(row)
            existing = records_by_code.get(item.classification_code)
            if existing is not None and existing != item:
                raise TushareIntegrationError(
                    "index_classify", f"conflicting duplicate classification_code={item.classification_code}"
                )
            records_by_code[item.classification_code] = item
        return tuple(records_by_code[code] for code in sorted(records_by_code))

    def list_index_daily(
        self, index_code: str, *, start_date: date, end_date: date
    ) -> tuple[TushareIndexDaily, ...]:
        """读取一条已批准指数的日线，返回代码必须与请求完全一致。"""
        if start_date > end_date:
            raise ValueError("start_date must not be after end_date.")
        rows = self._query(
            "index_daily",
            params={
                "ts_code": index_code,
                "start_date": start_date.strftime("%Y%m%d"),
                "end_date": end_date.strftime("%Y%m%d"),
            },
            fields=_INDEX_DAILY_FIELDS,
        )
        self._ensure_not_truncated("index_daily", rows, index_code)
        records = tuple(_to_index_daily(row) for row in rows)
        if any(item.index_code != index_code for item in records):
            raise TushareIntegrationError("index_daily", f"index_code={index_code} response contains another index")
        return records

    def list_index_weights(
        self, index_code: str, *, start_date: date, end_date: date
    ) -> tuple[TushareIndexWeight, ...]:
        """读取一条已批准指数的权重快照，保留来源权重单位。"""
        if start_date > end_date:
            raise ValueError("start_date must not be after end_date.")
        rows = self._query(
            "index_weight",
            params={
                "index_code": index_code,
                "start_date": start_date.strftime("%Y%m%d"),
                "end_date": end_date.strftime("%Y%m%d"),
            },
            fields=_INDEX_WEIGHT_FIELDS,
        )
        self._ensure_not_truncated("index_weight", rows, index_code)
        records = tuple(_to_index_weight(row) for row in rows)
        if any(item.index_code != index_code for item in records):
            raise TushareIntegrationError("index_weight", f"index_code={index_code} response contains another index")
        return records

    def _ensure_not_truncated(self, api_name: str, rows: tuple[dict[str, Any], ...], entity_key: str) -> None:
        if len(rows) >= self._max_rows_per_query:
            raise TushareIntegrationError(
                api_name, f"entity={entity_key} reached configured row limit; refusing partial response"
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
                return _parse_response(api_name, response.json())
            except TushareIntegrationError as error:
                if not error.retryable:
                    raise
                last_error = error
            except (httpx.HTTPError, ValueError) as error:
                last_error = error
            if attempt < self._max_retries:
                time.sleep(0.25 * (2**attempt))
        raise TushareIntegrationError(api_name, _safe_error_summary(last_error), retryable=True) from last_error


def _to_fund_daily(row: Mapping[str, Any]) -> TushareFundDaily:
    return TushareFundDaily(
        ts_code=_required_text(row, "ts_code", "fund_daily"),
        trade_date=_required_date(row.get("trade_date"), "trade_date", "fund_daily"),
        previous_close_price=_optional_decimal(row.get("pre_close"), "pre_close", "fund_daily"),
        open_price=_optional_decimal(row.get("open"), "open", "fund_daily"),
        high_price=_optional_decimal(row.get("high"), "high", "fund_daily"),
        low_price=_optional_decimal(row.get("low"), "low", "fund_daily"),
        close_price=_required_decimal(row.get("close"), "close", "fund_daily"),
        change_value=_optional_signed_decimal(row.get("change"), "change", "fund_daily"),
        change_percent=_optional_signed_decimal(row.get("pct_chg"), "pct_chg", "fund_daily"),
        volume=_optional_decimal(row.get("vol"), "vol", "fund_daily"),
        amount=_optional_decimal(row.get("amount"), "amount", "fund_daily"),
    )


def _to_index_basic(row: Mapping[str, Any]) -> TushareIndexBasic:
    return TushareIndexBasic(
        index_code=_required_text(row, "ts_code", "index_basic"),
        display_name=_required_text(row, "name", "index_basic"),
        market=_optional_text(row.get("market")),
        publisher=_optional_text(row.get("publisher")),
        category=_optional_text(row.get("category")),
        base_date=_optional_date(row.get("base_date"), "base_date", "index_basic"),
        list_date=_optional_date(row.get("list_date"), "list_date", "index_basic"),
        expiry_date=_optional_date(row.get("exp_date"), "exp_date", "index_basic"),
    )


def _to_index_classification(row: Mapping[str, Any]) -> TushareIndexClassification:
    level = _optional_text(row.get("level"))
    if level is None:
        hierarchy_level = None
    else:
        match = re.fullmatch(r"(?:L)?([1-9]\d*)", level, flags=re.IGNORECASE)
        if match is None:
            raise TushareIntegrationError("index_classify", "invalid level")
        hierarchy_level = int(match.group(1))
    return TushareIndexClassification(
        classification_code=_required_text(row, "index_code", "index_classify"),
        classification_name=_required_text(row, "industry_name", "index_classify"),
        parent_classification_code=_optional_text(row.get("parent_code")),
        hierarchy_level=hierarchy_level,
        source_name=_optional_text(row.get("src")),
    )


def _to_index_daily(row: Mapping[str, Any]) -> TushareIndexDaily:
    return TushareIndexDaily(
        index_code=_required_text(row, "ts_code", "index_daily"),
        trade_date=_required_date(row.get("trade_date"), "trade_date", "index_daily"),
        close_price=_required_decimal(row.get("close"), "close", "index_daily"),
    )


def _to_index_weight(row: Mapping[str, Any]) -> TushareIndexWeight:
    return TushareIndexWeight(
        index_code=_required_text(row, "index_code", "index_weight"),
        constituent_code=_required_text(row, "con_code", "index_weight"),
        trade_date=_required_date(row.get("trade_date"), "trade_date", "index_weight"),
        weight=_required_decimal(row.get("weight"), "weight", "index_weight"),
    )
