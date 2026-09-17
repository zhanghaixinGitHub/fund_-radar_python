"""七项持仓诊断事实的 VALID / CHANGED / INSUFFICIENT 分支与响应契约；使用内存桩数据，不接触数据库。"""

import dataclasses
import os
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
from app.core.config import get_settings
from app.db.base import Base
from app.repositories.fund_read import FundNavHistorySnapshot, FundShareHistorySnapshot
from app.services import portfolio_advice as service
from app.services.portfolio_advice import (
    DiagnosisComparisonInput,
    DiagnosisComparisonRow,
    DiagnosisDividendInput,
    DiagnosisInputs,
    DiagnosisManagerInput,
    DiagnosisProfileInput,
)

AS_OF = date(2026, 9, 17)
FUND = "006730"


def _nav_history(values):
    base = date(2025, 8, 1)
    return tuple(
        FundNavHistorySnapshot(
            nav_date=base + timedelta(days=index),
            unit_nav=Decimal(value),
            accumulated_nav=Decimal(value),
        )
        for index, value in enumerate(values)
    )


def _default_nav_values():
    # 峰值 2.0、谷底 1.5（最大回撤 -25%），末端 1.9（当前回撤 -5%）：默认 VALID。
    return ["1"] * 150 + ["2"] + ["1.5"] * 50 + ["1.9"] * 99


def _benchmark_points(nav_history, value="100"):
    return tuple((point.nav_date, Decimal(value)) for point in nav_history)


def _row(code, rate):
    return DiagnosisComparisonRow(
        fund_code=code,
        nav_date=date(2026, 9, 16),
        month_change_rate=Decimal(rate),
        data_source="TUSHARE_PRO_FUND",
    )


def _comparison(peer_count=5):
    peers = [
        _row(code, rate)
        for code, rate in zip(
            ("100001", "100002", "100003", "100004", "100005"),
            ("0.10", "0.08", "0.06", "0.04", "0.02"),
            strict=True,
        )
    ][:peer_count]
    return DiagnosisComparisonInput(
        target_status="ACTIVE",
        target_source_code="TUSHARE_PRO_FUND",
        as_of_date=date(2026, 9, 16),
        target_month_change_rate=Decimal("0.05"),
        rows=(_row(FUND, "0.05"), *peers),
    )


def _profile(**overrides):
    fields = {
        "benchmark": "沪深300指数收益率×80%＋中证全债指数收益率×20%",
        "management_fee": Decimal("0.015"),
        "custodian_fee": Decimal("0.0025"),
        "updated_at": datetime(2026, 9, 16, tzinfo=UTC),
    }
    return DiagnosisProfileInput(**(fields | overrides))


def make_inputs(**overrides):
    nav_history = _nav_history(_default_nav_values())
    fields = {
        "fund_code": FUND,
        "fund_type": "STOCK",
        "fund_status": "ACTIVE",
        "source_code": "TUSHARE_PRO_FUND",
        "benchmark_code": "CSI300",
        "nav_unavailable_reason": None,
        "nav_history": nav_history,
        "managers": (
            DiagnosisManagerInput(
                manager_name="张三", ann_date=date(2020, 1, 2), begin_date=date(2020, 1, 1), end_date=None
            ),
        ),
        "shares": (
            FundShareHistorySnapshot(
                trade_date=date(2026, 6, 30), fund_share=Decimal("100"), source_code="TUSHARE_PRO_FUND"
            ),
            FundShareHistorySnapshot(
                trade_date=date(2026, 9, 16), fund_share=Decimal("110"), source_code="TUSHARE_PRO_FUND"
            ),
        ),
        "comparison": _comparison(),
        "profile": _profile(),
        "benchmark_series_status": "ACTIVE",
        "benchmark_points": _benchmark_points(nav_history),
        "dividends_verified_at": datetime(2026, 9, 16, tzinfo=UTC),
        "dividends": (
            DiagnosisDividendInput(
                ann_date=date(2026, 3, 10), ex_date=date(2026, 3, 12), cash_dividend=Decimal("0.05")
            ),
        ),
    }
    return DiagnosisInputs(**(fields | overrides))


def items_by_key(result):
    return {item.item: item for item in result.items}


def facts(inputs, as_of=AS_OF):
    with patch.object(service, "read_diagnosis_inputs", lambda fund_code, as_of_date: inputs):
        return service.get_diagnosis_facts(FUND, as_of, now=datetime(2026, 9, 17, tzinfo=UTC))


# --- 经理任职 ---


def test_manager_valid_reports_current_combination():
    inputs = make_inputs(
        managers=(
            DiagnosisManagerInput(
                manager_name="张三", ann_date=date(2020, 1, 2), begin_date=date(2020, 1, 1), end_date=None
            ),
            DiagnosisManagerInput(
                manager_name="李四", ann_date=date(2023, 5, 8), begin_date=date(2023, 5, 6), end_date=None
            ),
            DiagnosisManagerInput(
                manager_name="王五", ann_date=date(2018, 1, 3), begin_date=date(2018, 1, 1), end_date=date(2020, 1, 1)
            ),
        )
    )
    item = items_by_key(facts(inputs))["MANAGER"]
    assert item.verdict == "VALID"
    assert item.source == "fund_manager_assignment"
    assert item.data_as_of_date == date(2023, 5, 8)
    assert item.facts["current_managers"] == [
        {"manager_name": "张三", "begin_date": date(2020, 1, 1)},
        {"manager_name": "李四", "begin_date": date(2023, 5, 6)},
    ]
    assert item.facts["latest_change_date"] == date(2023, 5, 6)
    assert "不等于管理能力恶化" in item.evidence


def test_manager_insufficient_without_records():
    item = items_by_key(facts(make_inputs(managers=())))["MANAGER"]
    assert item.verdict == "INSUFFICIENT"
    assert "无基金经理任职记录" in item.evidence


# --- 规模 ---


def test_scale_valid_when_change_below_thresholds():
    item = items_by_key(facts(make_inputs()))["SCALE"]
    assert item.verdict == "VALID"
    assert item.facts["latest_share"] == Decimal("110")
    assert item.facts["previous_share"] == Decimal("100")
    assert item.facts["change_ratio"] == Decimal("0.100000")
    assert item.data_as_of_date == date(2026, 9, 16)


@pytest.mark.parametrize(
    ("latest", "expected"),
    [("250", "CHANGED"), ("40", "CHANGED")],
)
def test_scale_changed_at_document_thresholds(latest, expected):
    inputs = make_inputs(
        shares=(
            FundShareHistorySnapshot(
                trade_date=date(2026, 6, 30), fund_share=Decimal("100"), source_code="TUSHARE_PRO_FUND"
            ),
            FundShareHistorySnapshot(
                trade_date=date(2026, 9, 16), fund_share=Decimal(latest), source_code="TUSHARE_PRO_FUND"
            ),
        )
    )
    item = items_by_key(facts(inputs))["SCALE"]
    assert item.verdict == expected
    assert "Java" in item.evidence


def test_scale_insufficient_without_history():
    item = items_by_key(facts(make_inputs(shares=())))["SCALE"]
    assert item.verdict == "INSUFFICIENT"
    assert "无规模历史" in item.evidence


# --- 相对同类排名 ---


def test_same_type_valid_carries_controlled_sample_note():
    item = items_by_key(facts(make_inputs()))["SAME_TYPE_RANK"]
    assert item.verdict == "VALID"
    assert item.facts["rank"] == 4
    assert item.facts["comparable_count"] == 6
    assert item.facts["scope"] == "CURRENT_MARKET_ACTIVE_TUSHARE_PRO_FUND"
    assert "不得解释为全市场排名" in item.evidence


def test_same_type_insufficient_when_sample_too_small():
    item = items_by_key(facts(make_inputs(comparison=_comparison(peer_count=3))))["SAME_TYPE_RANK"]
    assert item.verdict == "INSUFFICIENT"
    assert item.facts["comparable_count"] == 4
    assert "不得解释为全市场排名" in item.evidence


def test_same_type_insufficient_when_comparison_unavailable():
    item = items_by_key(facts(make_inputs(comparison=None)))["SAME_TYPE_RANK"]
    assert item.verdict == "INSUFFICIENT"


# --- 相对业绩基准 ---


def test_benchmark_valid_when_not_underperforming():
    item = items_by_key(facts(make_inputs()))["BENCHMARK"]
    assert item.verdict == "VALID"
    assert item.facts["benchmark_code"] == "CSI300"
    assert item.facts["return_diff_60d"] == Decimal("0.000000")
    assert "不以同类排名代替基准" in item.evidence


def test_benchmark_changed_when_underperforming_60d():
    nav_history = _nav_history(_default_nav_values())
    rising = tuple(
        (point.nav_date, Decimal("100") + Decimal(index)) for index, point in enumerate(nav_history)
    )
    item = items_by_key(facts(make_inputs(benchmark_points=rising)))["BENCHMARK"]
    assert item.verdict == "CHANGED"
    assert item.facts["return_diff_60d"] < 0


def test_benchmark_insufficient_without_benchmark_text():
    item = items_by_key(facts(make_inputs(profile=_profile(benchmark=None))))["BENCHMARK"]
    assert item.verdict == "INSUFFICIENT"
    assert "基准文本缺失" in item.evidence


def test_benchmark_insufficient_when_coverage_below_60():
    nav_history = _nav_history(_default_nav_values())
    item = items_by_key(facts(make_inputs(benchmark_points=_benchmark_points(nav_history)[:59])))["BENCHMARK"]
    assert item.verdict == "INSUFFICIENT"
    assert "覆盖不足" in item.evidence


def test_benchmark_insufficient_when_series_not_active():
    item = items_by_key(facts(make_inputs(benchmark_series_status="DRAFT")))["BENCHMARK"]
    assert item.verdict == "INSUFFICIENT"


# --- 当前回撤 ---


def test_drawdown_valid_when_far_from_max():
    item = items_by_key(facts(make_inputs()))["DRAWDOWN"]
    assert item.verdict == "VALID"
    assert item.facts["history_days"] == 300
    assert item.facts["nav_basis"] == "ACCUMULATED"
    assert item.facts["max_drawdown"] == Decimal("-0.250000")
    assert item.facts["current_drawdown"] == Decimal("-0.050000")


def test_drawdown_changed_at_80_percent_of_max():
    # 峰值 2.0、谷底 1.5（最大回撤 -25%），末端 1.6（当前回撤 -20% = 80%×-25%）。
    values = ["1"] * 150 + ["2"] + ["1.5"] * 50 + ["1.6"] * 49
    item = items_by_key(facts(make_inputs(nav_history=_nav_history(values))))["DRAWDOWN"]
    assert item.verdict == "CHANGED"
    assert item.facts["current_vs_max_ratio"] == Decimal("0.800000")
    assert "不是卖出指令" in item.evidence


def test_drawdown_insufficient_when_history_below_250_days():
    item = items_by_key(facts(make_inputs(nav_history=_nav_history(["1"] * 249))))["DRAWDOWN"]
    assert item.verdict == "INSUFFICIENT"
    assert item.facts["history_days"] == 249
    assert "max_drawdown" not in item.facts


def test_drawdown_insufficient_when_nav_source_unavailable():
    inputs = make_inputs(nav_unavailable_reason="该基金的净值来源未启用或尚无成功同步记录。", nav_history=())
    item = items_by_key(facts(inputs))["DRAWDOWN"]
    assert item.verdict == "INSUFFICIENT"
    assert "净值来源未启用" in item.evidence


# --- 费用 ---


def test_fee_valid_with_both_fields():
    item = items_by_key(facts(make_inputs()))["FEE"]
    assert item.verdict == "VALID"
    assert item.facts["management_fee"] == Decimal("0.015")
    assert item.facts["custodian_fee"] == Decimal("0.0025")
    assert "覆盖式快照" in item.evidence


def test_fee_insufficient_when_fields_missing():
    item = items_by_key(facts(make_inputs(profile=_profile(management_fee=None))))["FEE"]
    assert item.verdict == "INSUFFICIENT"
    assert "不补造数值" in item.evidence


def test_fee_insufficient_without_profile():
    item = items_by_key(facts(make_inputs(profile=None)))["FEE"]
    assert item.verdict == "INSUFFICIENT"


# --- 分红 ---


def test_dividend_valid_with_latest_event():
    item = items_by_key(facts(make_inputs()))["DIVIDEND"]
    assert item.verdict == "VALID"
    assert item.facts["total_events"] == 1
    assert item.facts["latest_ann_date"] == date(2026, 3, 10)
    assert item.facts["latest_cash_dividend"] == Decimal("0.05")
    assert "不解释为恶化" in item.evidence


def test_dividend_insufficient_when_not_synced():
    item = items_by_key(facts(make_inputs(dividends_verified_at=None)))["DIVIDEND"]
    assert item.verdict == "INSUFFICIENT"
    assert "尚未完成同步核验" in item.evidence


# --- 总体结论规则 ---


def test_overall_valid_when_all_valid():
    assert facts(make_inputs()).overall == "VALID"


def test_overall_insufficient_when_any_insufficient_without_changed():
    assert facts(make_inputs(shares=())).overall == "INSUFFICIENT"


def test_overall_changed_when_any_changed():
    inputs = make_inputs(
        shares=(
            FundShareHistorySnapshot(
                trade_date=date(2026, 6, 30), fund_share=Decimal("100"), source_code="TUSHARE_PRO_FUND"
            ),
            FundShareHistorySnapshot(
                trade_date=date(2026, 9, 16), fund_share=Decimal("250"), source_code="TUSHARE_PRO_FUND"
            ),
        )
    )
    result = facts(inputs)
    assert result.overall == "CHANGED"
    assert len(result.items) == 7
    assert result.basis == "HOLDING_DIAGNOSIS_FACTS_V1"


# --- 接口契约与鉴权 ---


def test_internal_endpoint_contract_and_token(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("AI_SERVICE_TOKEN", "diagnosis-test-only-token")
    get_settings.cache_clear()
    monkeypatch.setattr(service, "read_diagnosis_inputs", lambda fund_code, as_of: make_inputs())
    from app.main import create_application

    try:
        with TestClient(create_application()) as client:
            url = f"/internal/v1/portfolio-advice/{FUND}/diagnosis-facts?asOfDate=2026-09-17"
            assert client.get(url).status_code == 403
            assert (
                client.get(
                    url,
                    headers={"X-Service-Token": "diagnosis-test-only-token", "Origin": "http://localhost:5173"},
                ).status_code
                == 403
            )
            response = client.get(url, headers={"X-Service-Token": "diagnosis-test-only-token"})
            assert response.status_code == 200
            body = response.json()
            assert body["fundCode"] == FUND
            assert body["asOfDate"] == "2026-09-17"
            assert body["overall"] == "VALID"
            assert body["basis"] == "HOLDING_DIAGNOSIS_FACTS_V1"
            assert {item["item"] for item in body["items"]} == {
                "MANAGER", "SCALE", "SAME_TYPE_RANK", "BENCHMARK", "DRAWDOWN", "FEE", "DIVIDEND"
            }
            for item in body["items"]:
                assert {"item", "verdict", "evidence", "source", "dataAsOfDate", "facts"} <= set(item)
                assert item["verdict"] in ("VALID", "CHANGED", "INSUFFICIENT")
            # 受限 JSON：响应任何层级都不含用户身份字段。
            assert "userId" not in body and "user_id" not in body
            assert all("userId" not in item and "user_id" not in item for item in body["items"])
    finally:
        get_settings.cache_clear()


def test_internal_endpoint_defaults_as_of_to_current_date(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("AI_SERVICE_TOKEN", "diagnosis-test-only-token")
    get_settings.cache_clear()
    monkeypatch.setattr(service, "read_diagnosis_inputs", lambda fund_code, as_of: make_inputs())
    from app.main import create_application

    try:
        with TestClient(create_application()) as client:
            response = client.get(
                f"/internal/v1/portfolio-advice/{FUND}/diagnosis-facts",
                headers={"X-Service-Token": "diagnosis-test-only-token"},
            )
            assert response.status_code == 200
            assert response.json()["asOfDate"] is not None
    finally:
        get_settings.cache_clear()


# --- 回归：session 边界 ---


def _contains_orm_instance(value):
    """递归检查纯数据结构；返回 True 表示混入了 SQLAlchemy 映射实例。"""
    if isinstance(value, Base):
        return True
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.astuple(value)
    if isinstance(value, (tuple, list)):
        return any(_contains_orm_instance(item) for item in value)
    return False


def test_diagnosis_inputs_carry_no_orm_instances():
    """契约回归：read_diagnosis_inputs 的返回结构只允许纯数据，ORM 实例出 session 边界即过期。"""
    assert not _contains_orm_instance(make_inputs())
    for field in dataclasses.fields(DiagnosisInputs):
        assert "app.models" not in str(field.type), f"{field.name} 不得声明为 ORM 类型：{field.type}"


@pytest.mark.skipif(os.getenv("RUN_NAV_STORAGE_PG_TESTS") != "1", reason="显式启用隔离PG")
def test_read_diagnosis_inputs_survive_session_close_against_real_db():
    """真实库回归：session 关闭后逐字段访问返回值；旧实现在此抛 DetachedInstanceError。"""
    inputs = service.read_diagnosis_inputs(FUND, date.today())
    assert inputs.fund_type and inputs.fund_status and inputs.source_code
    for manager in inputs.managers:
        _ = (manager.manager_name, manager.ann_date, manager.begin_date, manager.end_date)
    if inputs.comparison is not None:
        _ = (
            inputs.comparison.target_status,
            inputs.comparison.target_source_code,
            inputs.comparison.as_of_date,
            inputs.comparison.target_month_change_rate,
        )
        for row in inputs.comparison.rows:
            _ = (row.fund_code, row.nav_date, row.month_change_rate, row.data_source)
    if inputs.profile is not None:
        _ = (inputs.profile.benchmark, inputs.profile.management_fee, inputs.profile.custodian_fee)
    for dividend in inputs.dividends:
        _ = (dividend.ann_date, dividend.ex_date, dividend.cash_dividend)
    _ = (inputs.nav_history, inputs.shares, inputs.benchmark_points, inputs.dividends_verified_at)
    assert not _contains_orm_instance(inputs)
