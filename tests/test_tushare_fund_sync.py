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
    TushareFundDividend,
    TushareFundManager,
    TushareFundNav,
    TushareFundShare,
    TushareIntegrationError,
)
from app.services.tushare_fund_sync import (
    _normalize_catalog_records,
    _normalize_market_dividend_records,
    _normalize_market_manager_records,
    _normalize_market_nav_history_records,
    _normalize_market_profile_records,
    _normalize_market_share_records,
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
            "fields": "ts_code,ann_date,nav_date,unit_nav,accum_nav,accum_div,net_asset,total_netasset,adj_nav",
        },
    ]


def test_tushare_client_reads_detail_endpoints_with_minimal_display_fields() -> None:
    """详情同步只请求本期需要的结构化字段，不读取资讯正文或经理简历。"""
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        api_name = payload["api_name"]
        if api_name == "fund_basic":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "fields": [
                            "ts_code",
                            "name",
                            "management",
                            "custodian",
                            "fund_type",
                            "found_date",
                            "due_date",
                            "list_date",
                            "issue_date",
                            "delist_date",
                            "issue_amount",
                            "m_fee",
                            "c_fee",
                            "duration_year",
                            "p_value",
                            "min_amount",
                            "exp_return",
                            "benchmark",
                            "status",
                            "invest_type",
                            "type",
                            "trustee",
                            "purc_startdate",
                            "redm_startdate",
                            "market",
                        ],
                        "items": [
                            [
                                "002112.OF",
                                "测试基金C",
                                "测试基金",
                                "测试银行",
                                "混合型",
                                "20150619",
                                None,
                                None,
                                None,
                                None,
                                "10.2",
                                "0.015",
                                "0.0025",
                                None,
                                "1",
                                "10",
                                None,
                                "中证测试指数",
                                "L",
                                "主动权益",
                                "混合型",
                                None,
                                None,
                                None,
                                "O",
                            ]
                        ],
                    },
                },
            )
        if api_name == "fund_manager":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "fields": ["ts_code", "ann_date", "name", "edu", "begin_date", "end_date"],
                        "items": [["002112.OF", "20200101", "张三", "硕士", "20200101", None]],
                    },
                },
            )
        if api_name == "fund_share":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "fields": ["ts_code", "trade_date", "fd_share"],
                        "items": [["002112.OF", "20260825", "12345.6"]],
                    },
                },
            )
        if api_name == "fund_div":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "fields": [
                            "ts_code",
                            "ann_date",
                            "imp_anndate",
                            "base_date",
                            "div_proc",
                            "record_date",
                            "ex_date",
                            "pay_date",
                            "earpay_date",
                            "net_ex_date",
                            "div_cash",
                            "base_unit",
                            "ear_distr",
                            "ear_amount",
                            "account_date",
                            "base_year",
                        ],
                        "items": [
                            [
                                "002112.OF",
                                "20260801",
                                None,
                                None,
                                "实施",
                                "20260810",
                                "20260811",
                                "20260812",
                                None,
                                None,
                                "0.01",
                                "10",
                                None,
                                None,
                                None,
                                "2026",
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
        basics = client.list_fund_detail_basics_by_ts_codes(("002112.OF",))
        managers = client.list_fund_managers("002112.OF")
        shares = client.list_fund_share_history("002112.OF", start_date=date(2026, 1, 1), end_date=date(2026, 8, 25))
        dividends = client.list_fund_dividends("002112.OF")

    assert basics[0].custodian == "测试银行"
    assert managers[0].name == "张三"
    assert shares[0].fund_share == Decimal("12345.6")
    assert dividends[0].cash_dividend == Decimal("0.01")
    assert [request["api_name"] for request in requests] == [
        "fund_basic",
        "fund_manager",
        "fund_share",
        "fund_div",
    ]
    assert requests[2]["params"] == {
        "ts_code": "002112.OF",
        "start_date": "20260101",
        "end_date": "20260825",
    }


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


def test_market_detail_normalizers_keep_only_the_requested_fund_scope() -> None:
    """详情同步必须在写库前过滤跨基金、窗口外或无事件键的数据。"""
    profile_records, profile_invalid_count = _normalize_market_profile_records(
        (TushareFundCompany(name="测试基金管理有限公司", short_name="测试基金"),),
        (
            TushareFundBasic(
                ts_code="002112.OF",
                name="测试基金C",
                management="测试基金",
                fund_type="混合型",
                found_date=date(2015, 6, 19),
                status="L",
                market="O",
                custodian="测试银行",
            ),
        ),
    )
    manager_records, manager_invalid_count = _normalize_market_manager_records(
        (
            TushareFundManager("002112.OF", date(2020, 1, 1), "张三", "硕士", date(2020, 1, 1), None),
            TushareFundManager("999999.OF", date(2020, 1, 1), "李四", None, date(2020, 1, 1), None),
        ),
        ts_codes=("002112.OF",),
    )
    share_records, share_invalid_count = _normalize_market_share_records(
        (
            TushareFundShare("002112.OF", date(2026, 8, 25), Decimal("100.5")),
            TushareFundShare("002112.OF", date(2025, 12, 31), Decimal("99")),
        ),
        ts_codes=("002112.OF",),
        start_date=date(2026, 1, 1),
        end_date=date(2026, 8, 25),
    )
    dividend_records, dividend_invalid_count = _normalize_market_dividend_records(
        (
            TushareFundDividend(
                "002112.OF",
                date(2026, 8, 1),
                None,
                None,
                "实施",
                None,
                None,
                None,
                None,
                None,
                Decimal("0.01"),
                None,
                None,
                None,
                None,
                "2026",
            ),
            TushareFundDividend(
                "002112.OF",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        ),
        ts_codes=("002112.OF",),
    )

    assert profile_invalid_count == 0
    assert profile_records[0].management_company_name == "测试基金管理有限公司"
    assert manager_invalid_count == 1
    assert [record.manager_name for record in manager_records] == ["张三"]
    assert share_invalid_count == 1
    assert [record.fund_share for record in share_records] == [Decimal("100.5")]
    assert dividend_invalid_count == 1
    assert len(dividend_records) == 1
