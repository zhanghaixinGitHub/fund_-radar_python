"""已授权场内基金与市场参考指数 Tushare 适配器的离线测试。"""

import json
from datetime import date
from decimal import Decimal

import httpx
import pytest
from app.integrations.tushare import TushareIntegrationError
from app.integrations.tushare_market_reference import (
    TushareFundDaily,
    TushareIndexBasic,
    TushareIndexClassification,
    TushareIndexDaily,
    TushareIndexWeight,
    TushareMarketReferenceClient,
)


def _create_client(handler: httpx.MockTransport) -> TushareMarketReferenceClient:
    return TushareMarketReferenceClient(
        token="sensitive-test-token",
        api_url="https://tushare.test",
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=0,
        catalog_max_rows_per_query=8_000,
        max_rows_per_query=8_000,
        transport=handler,
    )


def test_market_reference_client_uses_exact_fields_and_preserves_rows() -> None:
    """场内基金与指数请求必须使用明确代码、日期窗口和最小字段集。"""
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        rows_by_api = {
            "fund_daily": (
                [
                    "ts_code",
                    "trade_date",
                    "pre_close",
                    "open",
                    "high",
                    "low",
                    "close",
                    "change",
                    "pct_chg",
                    "vol",
                    "amount",
                ],
                [["510300.SH", "20260901", "4.1", "4.2", "4.3", "4.0", "4.25", "-0.15", "-3.66", "10", "20"]],
            ),
            "index_basic": (
                [
                    "ts_code",
                    "name",
                    "market",
                    "publisher",
                    "category",
                    "base_date",
                    "base_point",
                    "list_date",
                    "list_point",
                    "weight_rule",
                    "desc",
                    "exp_date",
                ],
                [["000300.SH", "沪深300", "SSE", "中证", "规模", "20041231", "1000", "20050104", "1000", "", "", None]],
            ),
            "index_classify": (
                ["index_code", "industry_name", "parent_code", "level", "industry_code", "is_pub", "src"],
                [["801010", "农林牧渔", "", "L1", "801010", "1", "SW"]],
            ),
            "index_daily": (
                [
                    "ts_code",
                    "trade_date",
                    "close",
                    "open",
                    "high",
                    "low",
                    "pre_close",
                    "change",
                    "pct_chg",
                    "vol",
                    "amount",
                ],
                [["000300.SH", "20260901", "3600", "3590", "3610", "3580", "3588", "12", "0.33", "100", "200"]],
            ),
            "index_weight": (
                ["index_code", "con_code", "trade_date", "weight"],
                [["000300.SH", "600000.SH", "20260829", "1.23"]],
            ),
        }
        fields, items = rows_by_api[payload["api_name"]]
        return httpx.Response(200, json={"code": 0, "data": {"fields": fields, "items": items}})

    client = _create_client(httpx.MockTransport(handler))
    try:
        fund_rows = client.list_fund_exchange_daily(
            "510300.SH", start_date=date(2026, 8, 1), end_date=date(2026, 9, 1)
        )
        catalog_rows = client.list_index_basics(("SSE",))
        classification_rows = client.list_index_classifications()
        index_rows = client.list_index_daily(
            "000300.SH", start_date=date(2026, 8, 1), end_date=date(2026, 9, 1)
        )
        weight_rows = client.list_index_weights(
            "000300.SH", start_date=date(2026, 8, 1), end_date=date(2026, 9, 1)
        )
    finally:
        client.close()

    assert fund_rows == (
        TushareFundDaily(
            ts_code="510300.SH",
            trade_date=date(2026, 9, 1),
            previous_close_price=Decimal("4.1"),
            open_price=Decimal("4.2"),
            high_price=Decimal("4.3"),
            low_price=Decimal("4.0"),
            close_price=Decimal("4.25"),
            change_value=Decimal("-0.15"),
            change_percent=Decimal("-3.66"),
            volume=Decimal("10"),
            amount=Decimal("20"),
        ),
    )
    assert catalog_rows == (
        TushareIndexBasic(
            index_code="000300.SH",
            display_name="沪深300",
            market="SSE",
            publisher="中证",
            category="规模",
            base_date=date(2004, 12, 31),
            list_date=date(2005, 1, 4),
            expiry_date=None,
        ),
    )
    assert classification_rows == (
        TushareIndexClassification("801010", "农林牧渔", None, 1, "SW"),
    )
    assert index_rows == (TushareIndexDaily("000300.SH", date(2026, 9, 1), Decimal("3600")),)
    assert weight_rows == (TushareIndexWeight("000300.SH", "600000.SH", date(2026, 8, 29), Decimal("1.23")),)
    assert [request["params"] for request in requests] == [
        {"ts_code": "510300.SH", "start_date": "20260801", "end_date": "20260901"},
        {"market": "SSE"},
        {},
        {"ts_code": "000300.SH", "start_date": "20260801", "end_date": "20260901"},
        {"index_code": "000300.SH", "start_date": "20260801", "end_date": "20260901"},
    ]


def test_index_catalog_refuses_a_market_fragment_at_source_row_limit() -> None:
    """来源达到目录行上限时必须拒绝写入，不得把截断目录当完整数据。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "fields": ["ts_code", "name", "market"],
                    "items": [[f"{number:06d}.CSI", "测试指数", "CSI"] for number in range(8_000)],
                },
            },
        )

    client = _create_client(httpx.MockTransport(handler))
    try:
        with pytest.raises(TushareIntegrationError, match="refusing partial index catalog"):
            client.list_index_basics(("CSI",))
    finally:
        client.close()


def test_index_classification_rejects_an_unknown_hierarchy_level() -> None:
    """分类层级只兼容来源的数字或 L 加数字格式。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "fields": ["index_code", "industry_name", "parent_code", "level", "industry_code", "is_pub", "src"],
                    "items": [["801010", "农林牧渔", "", "LEVEL_ONE", "801010", "1", "SW"]],
                },
            },
        )

    client = _create_client(httpx.MockTransport(handler))
    try:
        with pytest.raises(TushareIntegrationError, match="index_classify: invalid level"):
            client.list_index_classifications()
    finally:
        client.close()


def test_market_reference_api_error_never_echoes_token() -> None:
    """异常摘要必须保留来源错误语义，但不能输出服务端 Token。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 40203, "msg": "permission denied", "data": None})

    client = _create_client(httpx.MockTransport(handler))
    try:
        with pytest.raises(TushareIntegrationError) as error:
            client.list_index_classifications()
    finally:
        client.close()

    assert "permission denied" in str(error.value)
    assert "sensitive-test-token" not in str(error.value)


def test_cursor_failure_initializes_the_orm_default_before_incrementing() -> None:
    """新游标的服务端默认值尚未回填时，失败计数也必须从一开始且不掩盖原始异常。"""
    from uuid import UUID

    from app.repositories.market_reference_sync import mark_cursor_failure

    class StubSession:
        def __init__(self) -> None:
            self.cursor = None

        def get(self, _model: object, _identity: object) -> object:
            return self.cursor

        def add(self, cursor: object) -> None:
            self.cursor = cursor

    session = StubSession()
    mark_cursor_failure(
        session,  # type: ignore[arg-type]
        source_id=UUID("00000000-0000-0000-0000-000000000401"),
        dataset_code="FUND_EXCHANGE_DAILY",
        entity_key="510300.SH",
        error_summary="TushareIntegrationError: negative change is valid market data",
    )

    assert session.cursor is not None
    assert session.cursor.consecutive_failure_count == 1
    assert session.cursor.last_error_summary == "TushareIntegrationError: negative change is valid market data"
