"""逐工作日估值必须有本基金原文，开放交易和估值日期分别处理。"""

from contextlib import nullcontext
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest
from app.services.comprehensive_calendar import QDII_WORKDAY_POLICY, verified_qdii_workday_valuation
from app.services.direction_1d_protocol import digest


def material():
    quotes = {
        "workday": "工作日：指上海证券交易所、深圳证券交易所开放交易的工作日",
        "daily": "基金管理人应在每个工作日对基金资产估值。但根据基金合同的规定暂停估值时除外。",
        "holiday_close": (
            "交易所上市流通的权益类证券，以收盘价估值；估值日无交易且没有重大变化的，"
            "以最近交易日的市价（收盘价）估值；发生重大变化时应调整确定公允价格。"
        ),
    }
    body = "\n".join(quotes.values())
    own = {
        "id": "own",
        "kind": "PROSPECTUS",
        "codes": ["019449"],
        "url": "https://www.cifm.com/fund/prospectus.pdf",
        "body": body,
        "body_hash": digest(body),
    }
    return {
        "fund_code": "019449",
        "calendar_evidence": {
            "policy": "QDII_CNY_VERIFIED_SESSION_V1",
            "future_valuation_policy": QDII_WORKDAY_POLICY,
            "prospectus_ref": "own",
            "workday_valuation_quotes": quotes,
        },
        "documents": [own],
    }


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "wrong_fund",
        "wrong_policy",
        "wrong_host",
        "wrong_kind",
        "hash_changed",
        "missing_clause",
        "altered_quote",
        "joint_workday",
        "no_exception",
        "no_fair_value",
    ],
)
def test_daily_valuation_requires_complete_fund_specific_source(problem):
    data = material()
    own = data["documents"][0]
    evidence = data["calendar_evidence"]
    quotes = evidence["workday_valuation_quotes"]
    if problem == "wrong_fund":
        own["codes"] = ["007280"]
    elif problem == "wrong_policy":
        evidence["policy"] = "CN_NAV_HK_CONNECT_WINDOW_V1"
    elif problem == "wrong_host":
        own["url"] = "https://example.com/prospectus.pdf"
    elif problem == "wrong_kind":
        own["kind"] = "MARKET_DATA"
    elif problem == "hash_changed":
        own["body"] += "未经核验的改动"
    elif problem == "missing_clause":
        del quotes["workday"]
    elif problem == "altered_quote":
        quotes["daily"] += "没有在原文出现的内容"
    elif problem in {"joint_workday", "no_exception", "no_fair_value"}:
        if problem == "joint_workday":
            quotes["workday"] = "工作日：指上海证券交易所、深圳证券交易所和东京证券交易所共同开放交易的工作日"
        elif problem == "no_exception":
            quotes["daily"] = "基金管理人应在每个工作日对基金资产估值。"
        else:
            quotes["holiday_close"] = "权益类证券估值日无交易，以最近交易日的市价（收盘价）估值。"
        own["body"] = "\n".join(quotes.values())
        own["body_hash"] = digest(own["body"])
    assert verified_qdii_workday_valuation(data, {"own": own}) is (problem is None)


@pytest.mark.parametrize("upstream_ready", [True, False])
def test_explicit_daily_valuation_crosses_japan_holiday_but_keeps_base_checks(monkeypatch, upstream_ready):
    from app.services import comprehensive_multi_prediction as service
    from app.services import prediction_features, qdii_valuation_contract, trading_calendar
    from app.services.prediction_contract import PredictionFailure, target_dates

    data = material()
    data["time_context"] = {"baseline_complete": True}
    data["window"] = {"calendar_version": digest(["past", "current"])}
    sessions = tuple(
        date(2026, 11, 11) + timedelta(days=i)
        for i in range(51)
        if (date(2026, 11, 11) + timedelta(days=i)).weekday() < 5
    )
    past = SimpleNamespace(
        sessions=(), definition=SimpleNamespace(coverage_start=date(2025, 1, 1)), content_hash="past"
    )
    current = SimpleNamespace(
        sessions=sessions, definition=SimpleNamespace(coverage_end=date(2026, 12, 31)), content_hash="current"
    )
    monkeypatch.setattr(trading_calendar, "load_live_calendars", lambda: [past, current])
    # 先导入使用日历的模块，再替换依赖，避免模块首次导入时永久绑定测试替身。
    monkeypatch.setattr(prediction_features, "read_fund_data", lambda *args: pytest.fail("不回到不支持的默认QDII日历"))
    monkeypatch.setattr(service, "get_engine", lambda: SimpleNamespace(connect=lambda: nullcontext(None)))
    monkeypatch.setattr(service.repo, "profiles", lambda *args: [{"fund_code": "019449"}])
    monkeypatch.setattr(qdii_valuation_contract, "verified_qdii_window", lambda *args: upstream_ready)
    monkeypatch.setattr(
        service, "_matched_future_sessions", lambda *args: pytest.fail("本基金已核逐工作日估值，不再以休市差异截断")
    )
    at = datetime.fromisoformat("2026-11-12T14:52:35+08:00")
    if not upstream_ready:
        with pytest.raises(PredictionFailure, match="实际投向或估值条款"):
            service.review_calendar(data, "019449", at)
        return
    result = service.review_calendar(data, "019449", at)
    assert result.calendar_id == QDII_WORKDAY_POLICY
    assert date(2026, 11, 23) in result.sessions
    assert target_dates(result, at, "T20_V1")["endDate"] == "2026-12-09"
    half_year = target_dates(result, at, "M6_V1")
    assert half_year["nominalEndDate"] == "2027-05-11"
    assert half_year["endDate"] is None
    assert half_year["endDateStatus"] == "PENDING_OFFICIAL_CALENDAR"
