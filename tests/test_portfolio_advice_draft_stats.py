"""规则草案统计的分位复算、三档映射、触发统计、数据不足/不适用分支、缓存与接口契约；使用合成净值，不接触数据库。"""

import math
import statistics
from datetime import date, timedelta
from decimal import Decimal

import pytest
from app.core.config import get_settings
from app.repositories.fund_read import FundNavHistorySnapshot
from app.schemas.portfolio_advice import DraftStats, DraftTier, DraftTriggerStats
from app.services import portfolio_advice as service
from app.services.portfolio_advice import DraftStatsInputs

FUND = "006730"
BASE_DATE = date(2020, 1, 2)


def _nav(values, base=BASE_DATE):
    return tuple(
        FundNavHistorySnapshot(
            nav_date=base + timedelta(days=index),
            unit_nav=Decimal(str(value)),
            accumulated_nav=Decimal(str(value)),
        )
        for index, value in enumerate(values)
    )


def _default_values(days=600):
    # 确定性合成序列：趋势上行叠加周期波动，包含多次回撤与修复。
    return [round(1 + 0.0004 * i + 0.08 * math.sin(i / 7.0), 8) for i in range(days)]


def _peer_values(days=253):
    # 平缓上行、波动率远低于目标基金的同类样本。
    return [round(1 + 0.02 * i / (days - 1), 8) for i in range(days)]


def make_draft_inputs(**overrides):
    fields = {
        "fund_code": FUND,
        "fund_type": "STOCK",
        "fund_status": "ACTIVE",
        "nav_unavailable_reason": None,
        "nav_history": _nav(_default_values()),
        "peer_nav_points": tuple((f"1000{i:02d}", _nav(_peer_values())) for i in range(6)),
    }
    return DraftStatsInputs(**(fields | overrides))


@pytest.fixture(autouse=True)
def clear_draft_cache():
    service._draft_stats_cache.clear()
    yield
    service._draft_stats_cache.clear()


# --- 纯函数复算 ---


def test_quantile_matches_statistics_inclusive_method():
    data = sorted(Decimal(i) * Decimal("0.37") for i in range(1, 100))
    expected = statistics.quantiles([float(x) for x in data], n=100, method="inclusive")
    for index, p in ((49, "0.5"), (74, "0.75"), (89, "0.9"), (94, "0.95")):
        assert abs(float(service._quantile(data, Decimal(p))) - expected[index]) < 1e-9


def test_window_metrics_hand_computed():
    series = _nav(["10", "12", "9", "15", "14", "20"])
    metrics = service._window_metrics(tuple((d, v) for d, v in ((p.nav_date, p.unit_nav) for p in series)), window=3)
    assert len(metrics) == 4
    assert service._q6(metrics[0][0]) == Decimal("-0.100000")
    assert service._q6(metrics[0][1]) == Decimal("0.250000")
    assert service._q6(metrics[1][0]) == Decimal("0.250000")
    assert service._q6(metrics[2][1]) == service._q6(Decimal(1) - Decimal(14) / Decimal(15))


def test_trigger_stats_hand_computed():
    values = ("10", "12", "11", "9", "8", "13", "14", "7", "15")
    series = tuple((p.nav_date, p.unit_nav) for p in _nav(values))
    metrics = service._window_metrics(series, window=5)
    reduce = service._trigger_stats(series, metrics, threshold=Decimal("0.3"), kind="reduce", forward=3)
    assert reduce.trigger_count == 4
    # 触发窗口 0/1 有完整前向窗口：续跌 0.125 与 6/13，中位 (0.125 + 6/13) / 2。
    assert reduce.median_further_decline == service._q6((Decimal("0.125") + Decimal(6) / 13) / 2)
    # 触发窗口 0/1/3 分别 1/1/1 天修复，中位 1。
    assert reduce.median_recovery_days == Decimal("1")
    # 窗口 3 缺前向窗口计 1 次；窗口 4 缺前向窗口且始终未修复计 2 次。
    assert reduce.censored_count == 3
    profit = service._trigger_stats(series, metrics, threshold=Decimal("0.1"), kind="profit", forward=3)
    assert profit.trigger_count == 2
    assert profit.median_further_decline is None
    assert profit.median_recovery_days == Decimal("2")


def test_drawdown_episodes_hand_computed():
    series = tuple((p.nav_date, p.unit_nav) for p in _nav(["10", "8", "12", "6", "9", "12", "13"]))
    episodes, unrecovered = service._drawdown_episodes(series)
    assert [(service._q6(depth), days) for depth, days, _, _ in episodes] == [
        (Decimal("0.200000"), 1),
        (Decimal("0.500000"), 2),
    ]
    assert unrecovered == 0
    longer = series + ((series[-1][0] + timedelta(days=1), Decimal("5")),)
    _, unrecovered = service._drawdown_episodes(longer)
    assert unrecovered == 1


# --- 全量管线 ---


def _naive_dd_distribution(values, window=63):
    result = []
    for start in range(len(values) - window + 1):
        peak = values[start]
        dd = 0.0
        for t in range(start, start + window):
            peak = max(peak, values[t])
            dd = max(dd, 1 - values[t] / peak)
        result.append(dd)
    return result


def test_pipeline_quantiles_and_tiers_recomputable():
    inputs = make_draft_inputs()
    first = service._build_draft_stats(inputs)
    second = service._build_draft_stats(inputs)
    assert first == second, "同一净值历史两次计算必须一致"
    assert first.status == "AVAILABLE"
    assert first.history_days == 600 and first.window_count == 600 - 63 + 1
    assert first.nav_basis == "ACCUMULATED"
    assert first.stats_cutoff_date == BASE_DATE + timedelta(days=599)
    assert "分红不自行还原复权" in first.assumption
    values = _default_values()
    naive_dd = _naive_dd_distribution(values)
    naive_r = [values[i + 62] / values[i] - 1 for i in range(len(values) - 62)]
    tiers = {tier.tier: tier for tier in first.tiers}
    assert set(tiers) == {"CONSERVATIVE", "BALANCED", "LOOSE"}
    expected_dd = statistics.quantiles(naive_dd, n=100, method="inclusive")
    expected_r = statistics.quantiles(naive_r, n=100, method="inclusive")
    assert abs(float(-tiers["CONSERVATIVE"].reduce_drawdown_pct) - expected_dd[49]) < 2e-6
    assert abs(float(-tiers["BALANCED"].reduce_drawdown_pct) - expected_dd[74]) < 2e-6
    assert abs(float(-tiers["LOOSE"].reduce_drawdown_pct) - expected_dd[89]) < 2e-6
    assert abs(float(tiers["CONSERVATIVE"].take_profit_pct) - expected_r[69]) < 2e-6
    assert abs(float(tiers["BALANCED"].take_profit_pct) - expected_r[84]) < 2e-6
    assert abs(float(tiers["LOOSE"].take_profit_pct) - expected_r[94]) < 2e-6
    # 触发次数与对外公布阈值可相互复算。
    reduce_line = float(-tiers["CONSERVATIVE"].reduce_drawdown_pct)
    assert tiers["CONSERVATIVE"].reduce_trigger.trigger_count == sum(1 for dd in naive_dd if dd >= reduce_line)
    # 三档单调性：减仓线逐级更深，止盈线逐级更高。
    lines = [tiers[name] for name in ("CONSERVATIVE", "BALANCED", "LOOSE")]
    assert lines[0].reduce_drawdown_pct >= lines[1].reduce_drawdown_pct >= lines[2].reduce_drawdown_pct
    assert lines[0].take_profit_pct <= lines[1].take_profit_pct <= lines[2].take_profit_pct
    # 统计块含文档要求的分布与同类分位；同类样本 6 只且波动更低，分位为 1。
    stats = first.stats
    assert set(stats["dd_quantiles"]) == {"0.5", "0.75", "0.9"}
    assert set(stats["r_quantiles"]) == {"0.5", "0.7", "0.85", "0.95"}
    assert set(stats["l20"]) == {"min", "p10", "p50"}
    assert stats["annualized_volatility"] > 0
    assert stats["same_type_volatility"]["percentile"] == Decimal("1.000000")
    assert stats["same_type_volatility"]["sample_size"] == 6
    assert stats["max_drawdown"] <= 0
    assert stats["recovery_days"]["episodes"] >= 1


def test_same_type_volatility_marked_missing_when_sample_small():
    result = service._build_draft_stats(
        make_draft_inputs(peer_nav_points=tuple((f"1000{i:02d}", _nav(_peer_values())) for i in range(4)))
    )
    assert result.status == "AVAILABLE"
    same_type = result.stats["same_type_volatility"]
    assert same_type["percentile"] is None and same_type["sample_size"] == 4
    assert "标缺" in same_type["reason"]


# --- 数据不足与不适用分支 ---


def test_short_history_returns_explicit_status_without_numbers():
    result = service._build_draft_stats(make_draft_inputs(nav_history=_nav(_default_values(499))))
    assert result.status == "DATA_INSUFFICIENT"
    assert "历史太短，分位数不稳定" in result.reason
    assert result.tiers == () and result.stats is None
    assert result.history_days == 499


def test_nav_unavailable_returns_insufficient():
    result = service._build_draft_stats(
        make_draft_inputs(nav_unavailable_reason="该基金的净值来源未启用或尚无成功同步记录。", nav_history=())
    )
    assert result.status == "DATA_INSUFFICIENT"
    assert "净值来源未启用" in result.reason


@pytest.mark.parametrize(
    ("fund_type", "fund_status", "keyword"),
    [
        ("MONEY", "ACTIVE", "货币基金不适用"),
        ("QDII", "ACTIVE", "暂不适用"),
        ("STOCK", "SUSPENDED", "未启用"),
    ]
)
def test_not_applicable_categories(monkeypatch, fund_type, fund_status, keyword):
    monkeypatch.setattr(
        service, "read_draft_stats_watermark", lambda code: (fund_type, fund_status, date(2026, 9, 16))
    )
    monkeypatch.setattr(
        service, "read_draft_stats_inputs", lambda code: pytest.fail("不适用类别不得读取全历史")
    )
    result = service.get_draft_stats(FUND)
    assert result.status == "NOT_APPLICABLE"
    assert keyword in result.reason
    assert result.tiers == () and result.stats is None


# --- 缓存 ---


def test_cache_keyed_by_nav_watermark_without_user_identity(monkeypatch):
    calls = {"inputs": 0}
    watermark = {"value": date(2026, 9, 16)}

    def fake_watermark(code):
        return "STOCK", "ACTIVE", watermark["value"]

    def fake_inputs(code):
        calls["inputs"] += 1
        return make_draft_inputs()

    monkeypatch.setattr(service, "read_draft_stats_watermark", fake_watermark)
    monkeypatch.setattr(service, "read_draft_stats_inputs", fake_inputs)
    first = service.get_draft_stats(FUND)
    second = service.get_draft_stats(FUND)
    assert first is second and calls["inputs"] == 1, "同一净值水位不得重复全历史重算"
    watermark["value"] = date(2026, 9, 17)
    third = service.get_draft_stats(FUND)
    assert third is not first and calls["inputs"] == 2, "净值水位推进后必须重算"
    assert set(service._draft_stats_cache) == {(FUND, date(2026, 9, 16)), (FUND, date(2026, 9, 17))}
    for key in service._draft_stats_cache:
        assert all(part is None or isinstance(part, (str, date)) for part in key)


# --- 接口契约与鉴权 ---


def _stub_service(monkeypatch):
    monkeypatch.setattr(service, "read_draft_stats_watermark", lambda code: ("STOCK", "ACTIVE", date(2026, 9, 16)))
    monkeypatch.setattr(service, "read_draft_stats_inputs", lambda code: make_draft_inputs())


def _assert_no_user_identity_keys(payload):
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert "user" not in key.lower(), f"响应不得携带用户身份字段：{key}"
            _assert_no_user_identity_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            _assert_no_user_identity_keys(item)


def test_response_models_declare_no_user_identity_fields():
    for model in (DraftStats, DraftTier, DraftTriggerStats):
        for name, field in model.model_fields.items():
            assert "user" not in name.lower()
            assert "user" not in (field.alias or "").lower()


def test_internal_endpoint_contract_and_token(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("AI_SERVICE_TOKEN", "draft-stats-test-only-token")
    get_settings.cache_clear()
    _stub_service(monkeypatch)
    from app.main import create_application

    try:
        with TestClient(create_application()) as client:
            url = f"/internal/v1/portfolio-advice/{FUND}/draft-stats"
            assert client.get(url).status_code == 403
            assert (
                client.get(
                    url,
                    headers={"X-Service-Token": "draft-stats-test-only-token", "Origin": "http://localhost:5173"},
                ).status_code
                == 403
            )
            response = client.get(url, headers={"X-Service-Token": "draft-stats-test-only-token"})
            assert response.status_code == 200
            body = response.json()
            assert body["fundCode"] == FUND
            assert body["status"] == "AVAILABLE"
            assert body["basis"] == "HOLDING_RULE_DRAFT_STATS_V1"
            assert body["statsCutoffDate"] is not None
            assert body["historyDays"] == 600 and body["windowCount"] == 538
            assert len(body["tiers"]) == 3
            for tier in body["tiers"]:
                assert {
                    "tier", "reduceDrawdownPct", "takeProfitPct", "reduceTrigger", "takeProfitTrigger"
                } <= set(tier)
                for trigger in (tier["reduceTrigger"], tier["takeProfitTrigger"]):
                    assert {
                        "triggerCount", "medianFurtherDecline", "medianRecoveryDays", "censoredCount"
                    } <= set(trigger)
            _assert_no_user_identity_keys(body)
            # 重复请求命中缓存，不触发第二次全历史重算。
            assert client.get(url, headers={"X-Service-Token": "draft-stats-test-only-token"}).status_code == 200
    finally:
        get_settings.cache_clear()


def test_internal_endpoint_insufficient_returns_status_without_numbers(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("AI_SERVICE_TOKEN", "draft-stats-test-only-token")
    get_settings.cache_clear()
    monkeypatch.setattr(service, "read_draft_stats_watermark", lambda code: ("STOCK", "ACTIVE", date(2026, 9, 16)))
    monkeypatch.setattr(
        service,
        "read_draft_stats_inputs",
        lambda code: make_draft_inputs(nav_history=_nav(_default_values(499))),
    )
    from app.main import create_application

    try:
        with TestClient(create_application()) as client:
            response = client.get(
                f"/internal/v1/portfolio-advice/{FUND}/draft-stats",
                headers={"X-Service-Token": "draft-stats-test-only-token"},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "DATA_INSUFFICIENT"
            assert "历史太短" in body["reason"]
            assert body["tiers"] == [] and body["stats"] is None
    finally:
        get_settings.cache_clear()
