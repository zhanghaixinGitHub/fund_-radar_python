"""Tushare 基金适配器和同步规范化的离线测试。"""

import json
from datetime import date
from decimal import Decimal

import httpx
import pytest
from app.integrations.tushare import (
    TushareFundBasic,
    TushareFundClient,
    TushareFundCompany,
    TushareFundNav,
    TushareIntegrationError,
)
from app.services.tushare_fund_sync import (
    _normalize_catalog_records,
    _normalize_market_nav_history_records,
    _normalize_nav_records,
)


def test_tushare_client_splits_catalog_by_market_and_status() -> None:
    """目录同步必须以市场和状态分片，避免单次上限造成不完整目录。"""
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        api_name = payload["api_name"]
        if api_name == "fund_company":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {"fields": ["name", "shortname"], "items": [["甲基金管理有限公司", "甲基金"]]},
                },
            )
        if api_name == "fund_basic":
            params = payload["params"]
            market = params["market"]
            status = params["status"]
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "fields": ["ts_code", "name", "management", "fund_type", "found_date", "status", "market"],
                        "items": [
                            [
                                f"{market}{status}0001.OF",
                                f"基金{market}{status}A",
                                "甲基金",
                                "股票型",
                                "20200101",
                                status,
                                market,
                            ]
                        ],
                    },
                },
            )
        raise AssertionError(f"unexpected API {api_name}")

    with TushareFundClient(
        token="test-token",
        api_url="https://tushare.test",
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=0,
        catalog_max_rows_per_query=15_000,
        transport=httpx.MockTransport(handler),
    ) as client:
        companies = client.list_fund_companies()
        basics = client.list_fund_basics()

    assert companies == (TushareFundCompany(name="甲基金管理有限公司", short_name="甲基金"),)
    assert len(basics) == 6
    catalog_params = [request["params"] for request in requests if request["api_name"] == "fund_basic"]
    assert catalog_params == [
        {"market": "E", "status": "L"},
        {"market": "E", "status": "D"},
        {"market": "E", "status": "I"},
        {"market": "O", "status": "L"},
        {"market": "O", "status": "D"},
        {"market": "O", "status": "I"},
    ]


def test_tushare_api_error_does_not_echo_token() -> None:
    """外部授权错误的异常摘要不得包含请求 Token。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 40203, "msg": "permission denied", "data": None})

    client = TushareFundClient(
        token="sensitive-test-token",
        api_url="https://tushare.test",
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=0,
        catalog_max_rows_per_query=15_000,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(TushareIntegrationError) as error:
            client.list_fund_companies()
    finally:
        client.close()

    assert "permission denied" in str(error.value)
    assert "sensitive-test-token" not in str(error.value)


def test_tushare_client_reads_exact_market_catalog_and_history_by_known_code() -> None:
    """基金市场维护只请求明确的完整代码和日期窗口，不会回退到全市场目录。"""
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if payload["api_name"] == "fund_basic":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "fields": ["ts_code", "name", "management", "fund_type", "found_date", "status", "market"],
                        "items": [["002112.OF", "测试基金C", "测试基金", "混合型", "20150619", "L", "O"]],
                    },
                },
            )
        if payload["api_name"] == "fund_nav":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "fields": ["ts_code", "ann_date", "nav_date", "unit_nav", "accum_nav"],
                        "items": [["002112.OF", "20260826", "20260825", "4.8936", "5.0416"]],
                    },
                },
            )
        raise AssertionError(f"unexpected API {payload['api_name']}")

    with TushareFundClient(
        token="test-token",
        api_url="https://tushare.test",
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=0,
        catalog_max_rows_per_query=15_000,
        transport=httpx.MockTransport(handler),
    ) as client:
        basics = client.list_fund_basics_by_ts_codes(("002112.OF",))
        history = client.list_nav_history(
            "002112.OF", start_date=date(2026, 8, 1), end_date=date(2026, 8, 25)
        )

    assert basics[0].ts_code == "002112.OF"
    assert history[0].nav_date == date(2026, 8, 25)
    assert requests == [
        {
            "api_name": "fund_basic",
            "token": "test-token",
            "params": {"ts_code": "002112.OF"},
            "fields": "ts_code,name,management,fund_type,found_date,status,market",
        },
        {
            "api_name": "fund_nav",
            "token": "test-token",
            "params": {"ts_code": "002112.OF", "start_date": "20260801", "end_date": "20260825"},
            "fields": "ts_code,ann_date,nav_date,unit_nav,accum_nav",
        },
    ]


def test_tushare_client_resolves_suffix_only_from_a_unique_catalog_response() -> None:
    """历史记录缺少后缀时，只能依据来源目录的唯一精确响应补齐。"""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        ts_code = payload["params"]["ts_code"]
        items = (
            [["160323.SZ", "华夏磐泰混合（LOF）A", "华夏基金", "混合型", "20161226", "L", "E"]]
            if ts_code == "160323.SZ"
            else []
        )
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "fields": ["ts_code", "name", "management", "fund_type", "found_date", "status", "market"],
                    "items": items,
                },
            },
        )

    with TushareFundClient(
        token="test-token",
        api_url="https://tushare.test",
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_retries=0,
        catalog_max_rows_per_query=15_000,
        transport=httpx.MockTransport(handler),
    ) as client:
        resolved = client.resolve_fund_basics_by_fund_codes(("160323",))

    assert [item.ts_code for item in resolved] == ["160323.SZ"]


def test_catalog_normalization_uses_company_full_name_and_conservative_share_class() -> None:
    """管理人简称要映射全称，ETF 等非份额字母结尾的简称不得误判份额类别。"""
    records, invalid_count = _normalize_catalog_records(
        (TushareFundCompany(name="安信基金管理有限责任公司", short_name="安信基金"),),
        (
            TushareFundBasic(
                ts_code="010710.OF",
                name="安信医药健康主题股票C",
                management="安信基金",
                fund_type="股票型",
                found_date=date(2021, 1, 12),
                status="L",
                market="O",
            ),
            TushareFundBasic(
                ts_code="510050.SH",
                name="上证50ETF",
                management="安信基金",
                fund_type="指数型",
                found_date=None,
                status="L",
                market="E",
            ),
        ),
    )

    assert invalid_count == 0
    assert records[0].fund_code == "010710"
    assert records[0].manager_name == "安信基金管理有限责任公司"
    assert records[0].fund_type == "STOCK"
    assert records[0].status == "ACTIVE"
    assert records[0].share_class == "C"
    assert records[1].share_class == "UNSPECIFIED"


def test_nav_normalization_rejects_conflicting_duplicate_values() -> None:
    """同一代码和日期的不同净值不能静默覆盖，必须阻断本次同步。"""
    navs = (
        TushareFundNav("010710.OF", None, date(2026, 8, 25), Decimal("1.1"), Decimal("1.2")),
        TushareFundNav("010710.OF", None, date(2026, 8, 25), Decimal("1.2"), Decimal("1.3")),
    )

    with pytest.raises(TushareIntegrationError, match="conflicting duplicate NAV"):
        _normalize_nav_records(navs, date(2026, 8, 25))


def test_market_nav_history_rejects_records_outside_requested_window() -> None:
    """基金市场历史同步不能把请求窗口外的外部记录静默写入。"""
    records, invalid_count = _normalize_market_nav_history_records(
        (
            TushareFundNav("002112.OF", None, date(2026, 8, 25), Decimal("4.8"), Decimal("5.0")),
            TushareFundNav("002112.OF", None, date(2026, 7, 31), Decimal("4.7"), Decimal("4.9")),
        ),
        ts_codes=("002112.OF",),
        start_date=date(2026, 8, 1),
        end_date=date(2026, 8, 25),
    )

    assert invalid_count == 1
    assert [(record.fund_code, record.nav_date) for record in records] == [("002112", date(2026, 8, 25))]
