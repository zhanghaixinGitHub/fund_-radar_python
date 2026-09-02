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
FUND_TS_CODE_SUFFIXES: tuple[str, ...] = ("OF", "SH", "SZ")
_RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
_FUND_BASIC_MINIMAL_FIELDS = "ts_code,name,management,fund_type,found_date,status,market"
_FUND_BASIC_DETAIL_FIELDS = (
    "ts_code,name,management,custodian,fund_type,found_date,due_date,list_date,issue_date,delist_date,"
    "issue_amount,m_fee,c_fee,duration_year,p_value,min_amount,exp_return,benchmark,status,invest_type,type,"
    "trustee,purc_startdate,redm_startdate,market"
)
_FUND_NAV_FIELDS = "ts_code,ann_date,nav_date,unit_nav,accum_nav,accum_div,net_asset,total_netasset,adj_nav"
_FUND_MANAGER_FIELDS = "ts_code,ann_date,name,edu,begin_date,end_date"
_FUND_SHARE_FIELDS = "ts_code,trade_date,fd_share"
_FUND_DIVIDEND_FIELDS = (
    "ts_code,ann_date,imp_anndate,base_date,div_proc,record_date,ex_date,pay_date,earpay_date,net_ex_date,"
    "div_cash,base_unit,ear_distr,ear_amount,account_date,base_year"
)


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
    custodian: str | None = None
    due_date: date | None = None
    list_date: date | None = None
    issue_date: date | None = None
    delist_date: date | None = None
    issue_amount: Decimal | None = None
    management_fee: Decimal | None = None
    custodian_fee: Decimal | None = None
    duration_year: Decimal | None = None
    par_value: Decimal | None = None
    min_purchase_amount: Decimal | None = None
    expected_return: Decimal | None = None
    benchmark: str | None = None
    invest_type: str | None = None
    source_fund_type: str | None = None
    trustee: str | None = None
    purchase_start_date: date | None = None
    redemption_start_date: date | None = None


@dataclass(frozen=True)
class TushareFundNav:
    """基金净值接口的最小字段记录。"""

    ts_code: str
    ann_date: date | None
    nav_date: date
    unit_nav: Decimal
    accumulated_nav: Decimal | None
    accumulated_dividend: Decimal | None = None
    net_asset: Decimal | None = None
    total_net_asset: Decimal | None = None
    adjusted_nav: Decimal | None = None


@dataclass(frozen=True)
class TushareFundManager:
    """基金经理任职记录的最小展示字段。"""

    ts_code: str
    ann_date: date | None
    name: str
    education: str | None
    begin_date: date | None
    end_date: date | None


@dataclass(frozen=True)
class TushareFundShare:
    """基金份额规模变动记录，单位遵循来源的“万份”口径。"""

    ts_code: str
    trade_date: date
    fund_share: Decimal


@dataclass(frozen=True)
class TushareFundDividend:
    """基金分红事件记录，不包含公告正文。"""

    ts_code: str
    ann_date: date | None
    implementation_ann_date: date | None
    base_date: date | None
    process_status: str | None
    record_date: date | None
    ex_date: date | None
    pay_date: date | None
    earnings_pay_date: date | None
    nav_ex_date: date | None
    cash_dividend: Decimal | None
    base_unit: Decimal | None
    distributable_earnings: Decimal | None
    earnings_amount: Decimal | None
    reinvestment_arrival_date: date | None
    base_year: str | None


class TushareFundClient:
    """通过 HTTPS 调用 Tushare 公募基金接口的同步客户端。

    Args:
        token: 仅从环境配置读取的 Tushare Token。
        api_url: Tushare HTTP API 地址。
        connect_timeout_seconds: 建连超时秒数。
        read_timeout_seconds: 读取超时秒数。
        max_retries: 仅对传输和服务端可恢复错误的额外重试次数。
        catalog_max_rows_per_query: 单个 `fund_basic` 分片的允许最大记录数。
        nav_max_rows_per_query: 单只基金份额历史净值请求的允许最大记录数。
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
        nav_max_rows_per_query: int = 10_000,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not token.strip():
            raise ValueError("Tushare Token is not configured.")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative.")
        if nav_max_rows_per_query < 1:
            raise ValueError("nav_max_rows_per_query must be positive.")
        self._token = token
        self._max_retries = max_retries
        self._catalog_max_rows_per_query = catalog_max_rows_per_query
        self._nav_max_rows_per_query = nav_max_rows_per_query
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
        fields = _FUND_BASIC_MINIMAL_FIELDS
        for market in CATALOG_MARKETS:
            for status in CATALOG_STATUSES:
                rows = self._query("fund_basic", params={"market": market, "status": status}, fields=fields)
                if len(rows) >= self._catalog_max_rows_per_query:
                    raise TushareIntegrationError(
                        "fund_basic",
                        f"market={market}, status={status} reached configured row limit; refusing partial catalog",
                    )
                for row in rows:
                    item = _to_fund_basic(row)
                    existing = by_ts_code.get(item.ts_code)
                    if existing is not None and existing != item:
                        raise TushareIntegrationError("fund_basic", f"conflicting duplicate ts_code={item.ts_code}")
                    by_ts_code[item.ts_code] = item
        return tuple(by_ts_code[ts_code] for ts_code in sorted(by_ts_code))

    def list_fund_basics_by_ts_codes(self, ts_codes: tuple[str, ...]) -> tuple[TushareFundBasic, ...]:
        """按已知完整 Tushare 代码读取基金目录，不触发全市场分片查询。

        每个代码必须恰好返回自身的一条记录；缺失、重复或代码错配都直接失败，
        防止把部分指定清单误当作完整清单写入。
        """
        if not ts_codes:
            raise ValueError("ts_codes must not be empty.")
        if len(set(ts_codes)) != len(ts_codes):
            raise ValueError("ts_codes must not contain duplicates.")
        fields = _FUND_BASIC_MINIMAL_FIELDS
        records: list[TushareFundBasic] = []
        for ts_code in ts_codes:
            rows = self._query("fund_basic", params={"ts_code": ts_code}, fields=fields)
            if len(rows) != 1:
                raise TushareIntegrationError(
                    "fund_basic", f"ts_code={ts_code} expected exactly one record but received {len(rows)}"
                )
            item = _to_fund_basic(rows[0])
            if item.ts_code != ts_code:
                raise TushareIntegrationError(
                    "fund_basic", f"ts_code={ts_code} returned mismatched code={item.ts_code}"
                )
            records.append(item)
        return tuple(records)

    def resolve_fund_basics_by_fund_codes(self, fund_codes: tuple[str, ...]) -> tuple[TushareFundBasic, ...]:
        """以来源目录验证六位展示代码对应的唯一完整 Tushare 代码。

        该方法仅用于补齐历史数据缺失的来源后缀。它会逐个查询标准后缀，
        但只接受 API 返回的唯一精确记录，绝不把任一候选后缀当作推断结果写库。
        """
        if not fund_codes or len(set(fund_codes)) != len(fund_codes):
            raise ValueError("fund_codes must be non-empty and unique.")
        fields = _FUND_BASIC_MINIMAL_FIELDS
        records: list[TushareFundBasic] = []
        for fund_code in fund_codes:
            matches: list[TushareFundBasic] = []
            for suffix in FUND_TS_CODE_SUFFIXES:
                ts_code = f"{fund_code}.{suffix}"
                rows = self._query("fund_basic", params={"ts_code": ts_code}, fields=fields)
                if not rows:
                    continue
                if len(rows) != 1:
                    raise TushareIntegrationError(
                        "fund_basic", f"ts_code={ts_code} expected zero or one record but received {len(rows)}"
                    )
                item = _to_fund_basic(rows[0])
                if item.ts_code != ts_code:
                    raise TushareIntegrationError(
                        "fund_basic", f"ts_code={ts_code} returned mismatched code={item.ts_code}"
                    )
                matches.append(item)
            if len(matches) != 1:
                raise TushareIntegrationError(
                    "fund_basic", f"fund_code={fund_code} could not be resolved to one exact Tushare code"
                )
            records.append(matches[0])
        return tuple(records)

    def list_nav_daily(self, nav_date: date) -> tuple[TushareFundNav, ...]:
        """按净值日期批量读取公募基金净值，不逐基金发起远程请求。"""
        rows = self._query(
            "fund_nav",
            params={"nav_date": nav_date.strftime("%Y%m%d")},
            fields=_FUND_NAV_FIELDS,
        )
        return tuple(_to_fund_nav(row) for row in rows)

    def list_nav_history(
        self, ts_code: str, *, start_date: date | None = None, end_date: date | None = None
    ) -> tuple[TushareFundNav, ...]:
        """读取一只基金份额的历史净值，并拒绝达到受控行数上限的可疑响应。"""
        if start_date and end_date and start_date > end_date:
            raise ValueError("start_date must not be after end_date.")
        params = {"ts_code": ts_code}
        if start_date is not None:
            params["start_date"] = start_date.strftime("%Y%m%d")
        if end_date is not None:
            params["end_date"] = end_date.strftime("%Y%m%d")
        rows = self._query(
            "fund_nav",
            params=params,
            fields=_FUND_NAV_FIELDS,
        )
        if len(rows) >= self._nav_max_rows_per_query:
            raise TushareIntegrationError(
                "fund_nav", f"ts_code={ts_code} reached configured row limit; refusing partial history"
            )
        records = tuple(_to_fund_nav(row) for row in rows)
        if any(item.ts_code != ts_code for item in records):
            raise TushareIntegrationError("fund_nav", f"ts_code={ts_code} response contains another fund")
        return records

    def list_fund_detail_basics_by_ts_codes(self, ts_codes: tuple[str, ...]) -> tuple[TushareFundBasic, ...]:
        """读取指定来源代码的完整基础资料，不回退至全市场目录。"""
        if not ts_codes or len(set(ts_codes)) != len(ts_codes):
            raise ValueError("ts_codes must be non-empty and unique.")
        records: list[TushareFundBasic] = []
        for ts_code in ts_codes:
            rows = self._query("fund_basic", params={"ts_code": ts_code}, fields=_FUND_BASIC_DETAIL_FIELDS)
            if len(rows) != 1:
                raise TushareIntegrationError(
                    "fund_basic", f"ts_code={ts_code} expected exactly one detail record but received {len(rows)}"
                )
            item = _to_fund_basic(rows[0])
            if item.ts_code != ts_code:
                raise TushareIntegrationError(
                    "fund_basic", f"ts_code={ts_code} returned mismatched code={item.ts_code}"
                )
            records.append(item)
        return tuple(records)

    def list_fund_managers(self, ts_code: str) -> tuple[TushareFundManager, ...]:
        """读取一只基金的经理任职历史，不请求简历和其他非必要个人资料。"""
        rows = self._query("fund_manager", params={"ts_code": ts_code}, fields=_FUND_MANAGER_FIELDS)
        records = tuple(_to_fund_manager(row) for row in rows)
        if any(item.ts_code != ts_code for item in records):
            raise TushareIntegrationError("fund_manager", f"ts_code={ts_code} response contains another fund")
        return records

    def list_fund_share_history(
        self, ts_code: str, *, start_date: date, end_date: date
    ) -> tuple[TushareFundShare, ...]:
        """读取明确日期窗口的基金份额规模记录，防止无界历史请求。"""
        if start_date > end_date:
            raise ValueError("start_date must not be after end_date.")
        rows = self._query(
            "fund_share",
            params={
                "ts_code": ts_code,
                "start_date": start_date.strftime("%Y%m%d"),
                "end_date": end_date.strftime("%Y%m%d"),
            },
            fields=_FUND_SHARE_FIELDS,
        )
        records = tuple(_to_fund_share(row) for row in rows)
        if any(item.ts_code != ts_code for item in records):
            raise TushareIntegrationError("fund_share", f"ts_code={ts_code} response contains another fund")
        return records

    def list_fund_dividends(self, ts_code: str) -> tuple[TushareFundDividend, ...]:
        """读取一只基金的结构化分红记录，不读取任何资讯正文。"""
        rows = self._query("fund_div", params={"ts_code": ts_code}, fields=_FUND_DIVIDEND_FIELDS)
        records = tuple(_to_fund_dividend(row) for row in rows)
        if any(item.ts_code != ts_code for item in records):
            raise TushareIntegrationError("fund_div", f"ts_code={ts_code} response contains another fund")
        return records

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


def _to_fund_basic(row: Mapping[str, Any]) -> TushareFundBasic:
    """将单行基金目录响应转换为强类型记录。"""
    return TushareFundBasic(
        ts_code=_required_text(row, "ts_code", "fund_basic"),
        name=_required_text(row, "name", "fund_basic"),
        management=_optional_text(row.get("management")),
        fund_type=_optional_text(row.get("fund_type")),
        found_date=_optional_date(row.get("found_date"), "found_date", "fund_basic"),
        status=_optional_text(row.get("status")),
        market=_optional_text(row.get("market")),
        custodian=_optional_text(row.get("custodian")),
        due_date=_optional_date(row.get("due_date"), "due_date", "fund_basic"),
        list_date=_optional_date(row.get("list_date"), "list_date", "fund_basic"),
        issue_date=_optional_date(row.get("issue_date"), "issue_date", "fund_basic"),
        delist_date=_optional_date(row.get("delist_date"), "delist_date", "fund_basic"),
        issue_amount=_optional_decimal(row.get("issue_amount"), "issue_amount", "fund_basic"),
        management_fee=_optional_decimal(row.get("m_fee"), "m_fee", "fund_basic"),
        custodian_fee=_optional_decimal(row.get("c_fee"), "c_fee", "fund_basic"),
        duration_year=_optional_decimal(row.get("duration_year"), "duration_year", "fund_basic"),
        par_value=_optional_decimal(row.get("p_value"), "p_value", "fund_basic"),
        min_purchase_amount=_optional_decimal(row.get("min_amount"), "min_amount", "fund_basic"),
        expected_return=_optional_decimal(row.get("exp_return"), "exp_return", "fund_basic"),
        benchmark=_optional_text(row.get("benchmark")),
        invest_type=_optional_text(row.get("invest_type")),
        source_fund_type=_optional_text(row.get("type")),
        trustee=_optional_text(row.get("trustee")),
        purchase_start_date=_optional_date(row.get("purc_startdate"), "purc_startdate", "fund_basic"),
        redemption_start_date=_optional_date(row.get("redm_startdate"), "redm_startdate", "fund_basic"),
    )


def _to_fund_nav(row: Mapping[str, Any]) -> TushareFundNav:
    """将单行基金净值响应转换为强类型记录。"""
    return TushareFundNav(
        ts_code=_required_text(row, "ts_code", "fund_nav"),
        ann_date=_optional_date(row.get("ann_date"), "ann_date", "fund_nav"),
        nav_date=_required_date(row.get("nav_date"), "nav_date", "fund_nav"),
        unit_nav=_required_decimal(row.get("unit_nav"), "unit_nav", "fund_nav"),
        accumulated_nav=_optional_decimal(row.get("accum_nav"), "accum_nav", "fund_nav"),
        accumulated_dividend=_optional_decimal(row.get("accum_div"), "accum_div", "fund_nav"),
        net_asset=_optional_decimal(row.get("net_asset"), "net_asset", "fund_nav"),
        total_net_asset=_optional_decimal(row.get("total_netasset"), "total_netasset", "fund_nav"),
        adjusted_nav=_optional_decimal(row.get("adj_nav"), "adj_nav", "fund_nav"),
    )


def _to_fund_manager(row: Mapping[str, Any]) -> TushareFundManager:
    """将经理接口行转换为最小公开资料记录。"""
    return TushareFundManager(
        ts_code=_required_text(row, "ts_code", "fund_manager"),
        ann_date=_optional_date(row.get("ann_date"), "ann_date", "fund_manager"),
        name=_required_text(row, "name", "fund_manager"),
        education=_optional_text(row.get("edu")),
        begin_date=_optional_date(row.get("begin_date"), "begin_date", "fund_manager"),
        end_date=_optional_date(row.get("end_date"), "end_date", "fund_manager"),
    )


def _to_fund_share(row: Mapping[str, Any]) -> TushareFundShare:
    """将基金份额规模接口行转换为强类型记录。"""
    return TushareFundShare(
        ts_code=_required_text(row, "ts_code", "fund_share"),
        trade_date=_required_date(row.get("trade_date"), "trade_date", "fund_share"),
        fund_share=_required_decimal(row.get("fd_share"), "fd_share", "fund_share"),
    )


def _to_fund_dividend(row: Mapping[str, Any]) -> TushareFundDividend:
    """将基金分红接口行转换为结构化事件，不保留资讯文本。"""
    return TushareFundDividend(
        ts_code=_required_text(row, "ts_code", "fund_div"),
        ann_date=_optional_date(row.get("ann_date"), "ann_date", "fund_div"),
        implementation_ann_date=_optional_date(row.get("imp_anndate"), "imp_anndate", "fund_div"),
        base_date=_optional_date(row.get("base_date"), "base_date", "fund_div"),
        process_status=_optional_text(row.get("div_proc")),
        record_date=_optional_date(row.get("record_date"), "record_date", "fund_div"),
        ex_date=_optional_date(row.get("ex_date"), "ex_date", "fund_div"),
        pay_date=_optional_date(row.get("pay_date"), "pay_date", "fund_div"),
        earnings_pay_date=_optional_date(row.get("earpay_date"), "earpay_date", "fund_div"),
        nav_ex_date=_optional_date(row.get("net_ex_date"), "net_ex_date", "fund_div"),
        cash_dividend=_optional_decimal(row.get("div_cash"), "div_cash", "fund_div"),
        base_unit=_optional_decimal(row.get("base_unit"), "base_unit", "fund_div"),
        distributable_earnings=_optional_decimal(row.get("ear_distr"), "ear_distr", "fund_div"),
        earnings_amount=_optional_decimal(row.get("ear_amount"), "ear_amount", "fund_div"),
        reinvestment_arrival_date=_optional_date(row.get("account_date"), "account_date", "fund_div"),
        base_year=_optional_text(row.get("base_year")),
    )


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


def _optional_signed_decimal(value: Any, field_name: str, api_name: str) -> Decimal | None:
    """解析可正可负的有限十进制数，适用于涨跌额和涨跌幅等方向性字段。"""
    if value is None or value == "":
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise TushareIntegrationError(api_name, f"field {field_name} is not decimal") from error
    if not parsed.is_finite():
        raise TushareIntegrationError(api_name, f"field {field_name} must be a finite decimal")
    return parsed


def _safe_error_summary(error: object) -> str:
    """将外部错误压缩为不包含请求体和 Token 的短摘要。"""
    text = str(error or "unknown error").replace("\n", " ").replace("\r", " ")
    return text[:300]
