"""公告身份、新持仓及本人正式读回必须闭环；不能把待核验改名为完成。"""

import hashlib
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest
from app.commands.fund_prediction_followthrough import owner_scope
from app.services.direction_1d_protocol import ZONE, digest
from app.services.fund_exposure_common import save
from app.services.fund_prediction_followthrough import HORIZONS, recovery_actions, verify_outputs

WINDOW = {"base_nav_date": "2026-10-27", "target_nav_date": "2026-10-28"}
DATES = {
    "T5_V1": {"endDateStatus": "RESOLVED", "endDate": "2026-11-03"},
    "T20_V1": {"endDateStatus": "RESOLVED", "endDate": "2026-11-24"},
    "M6_V1": {"endDateStatus": "PENDING_OFFICIAL_CALENDAR"},
}


def records():
    return [
        {
            "id": h,
            "horizon": h,
            "fund_code": "002112",
            "base_date": WINDOW["base_nav_date"],
            "target_date": WINDOW["target_nav_date"],
            "end_date": DATES.get(h, {}).get("endDate", "2026-10-28"),
            "owner_verified": True,
        }
        for h in HORIZONS[:3]
    ]


def test_new_holding_and_failed_notice_produce_concrete_next_actions():
    report = {"id": "new-report", "endDate": "2026-09-30", "holdings": [{"stockCode": "002281.SZ"}]}
    material = {"report": {"endDate": "2026-06-30"}, "companies": []}
    saved = deepcopy(material)
    news = {"status": "PARTIAL_SUCCESS", "catalog_complete": True, "errors": [{"article_id": "AN-audit"}]}
    actions = recovery_actions("002112", report, material, news, None, "2026-10-27")
    assert {a["action"] for a in actions} == {
        "SUPPLEMENT_HOLDING_MATERIALS",
        "VERIFY_NOTICE_IDENTITY_OR_SOURCE",
        "RECOVER_NOTICE_COVERAGE",
        "REVIEW_COMPREHENSIVE_MATERIALS",
        "REVIEW_NEW_REPORT",
    }
    holding = next(a for a in actions if a["action"] == "SUPPLEMENT_HOLDING_MATERIALS")
    assert holding["new_securities"] == ["002281.SZ"]
    assert material == saved  # 只生成补齐清单，不篡改旧完整持仓或就绪标志。


@pytest.mark.parametrize("problem", ["old_quote", "missing_financials"])
def test_existing_security_with_incomplete_current_materials_also_needs_recovery(problem):
    company = {"code": "002281.SZ", "quote": {"date": "2026-10-27"}, "financials": [{"endDate": "2026-09-30"}]}
    if problem == "old_quote":
        company["quote"]["date"] = "2026-10-26"
    else:
        company["financials"] = []
    report = {"id": "r", "endDate": "2026-09-30", "holdings": [{"stockCode": "002281.SZ"}]}
    actions = recovery_actions("002112", report, {"companies": [company]}, {}, None, "2026-10-27")
    assert next(a for a in actions if a["action"] == "SUPPLEMENT_HOLDING_MATERIALS")["incomplete_securities"] == [
        "002281.SZ"
    ]


def test_ready_materials_without_formal_owner_readback_are_not_complete():
    result = verify_outputs("002112", WINDOW, DATES, [], [])
    assert result["status"] == "ACTION_REQUIRED"
    assert [h["status"] for h in result["horizons"]] == ["ACTION_REQUIRED"] * 3 + ["EXTERNAL_BLOCKED"]


def test_target_etf_new_report_and_new_security_cannot_hide_behind_parent_report():
    material = {
        "fund_code": "001632",
        "report": {"endDate": "2026-09-30"},
        "exposure_chain": ["001632", "159736"],
        "companies": [],
        "source_manifest": {},
    }
    news = {"status": "SUCCEEDED", "catalog_complete": True, "errors": []}
    handoff = {"reviewed": True, "gaps": []}
    report = {"id": "parent", "endDate": "2026-09-30", "holdings": []}
    target = {"id": "target-current", "endDate": "2026-09-30", "holdings": [{"stockCode": "301660.SZ"}]}
    actions = recovery_actions(
        "001632", report, material, news, handoff, "2026-11-02", target_reports={"159736": target}
    )
    assert {a["action"] for a in actions} == {"REVIEW_NEW_LOOKTHROUGH_REPORT", "SUPPLEMENT_HOLDING_MATERIALS"}
    assert next(a for a in actions if a["action"] == "SUPPLEMENT_HOLDING_MATERIALS")["new_securities"] == ["301660.SZ"]
    material["source_manifest"]["target_partial_reports"] = {"159736": {"report": target}}
    material["companies"] = [
        {"code": "301660.SZ", "financials": [{"sourceRef": "own-issuer"}], "quote": {"date": "2026-11-02"}}
    ]
    assert (
        recovery_actions("001632", report, material, news, handoff, "2026-11-02", target_reports={"159736": target})
        == []
    )
    assert (
        recovery_actions("001632", report, material, news, handoff, "2026-11-02", target_reports={"159736": None})[0][
            "action"
        ]
        == "COLLECT_TARGET_REPORT"
    )


def test_readback_finishes_supported_periods_and_keeps_calendar_block_separate():
    result = verify_outputs("002112", WINDOW, DATES, records(), [])
    assert result["status"] == "CLOSED"
    assert [h["status"] for h in result["horizons"]] == ["SAVED_AND_READ_BACK"] * 3 + ["EXTERNAL_BLOCKED"]


def test_new_action_invalidates_earlier_same_day_results_for_completion():
    result = verify_outputs("002112", WINDOW, DATES, records(), [{"action": "REVIEW_NEW_REPORT"}])
    assert result["status"] == "ACTION_REQUIRED"


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner_verified", False),
        ("fund_code", "000001"),
        ("base_date", "2026-10-26"),
        ("target_date", "2026-10-27"),
        ("end_date", "2026-10-30"),
    ],
)
def test_cross_owner_fund_date_and_window_cannot_count_as_saved(field, value):
    rows = records()
    rows[0][field] = value
    result = verify_outputs("002112", WINDOW, DATES, rows, [])
    assert result["horizons"][0]["status"] == "ACTION_REQUIRED"


def test_pending_label_or_unsupported_external_claim_cannot_close_work():
    exception = {
        "fund_code": "002112",
        "horizon": "T1",
        "target_date": "2026-10-28",
        "reason": "WAITING_REVIEW",
        "explanation": "待核验" * 40,
        "attempts": [],
    }
    result = verify_outputs("002112", WINDOW, DATES, records()[1:], [], [exception])
    assert result["horizons"][0]["status"] == "ACTION_REQUIRED"


def test_source_failure_without_real_receipt_is_still_pending():
    exception = {
        "fund_code": "002112",
        "horizon": "T1",
        "target_date": "2026-10-28",
        "reason": "SOURCE_UNAVAILABLE",
        "explanation": "真实外部来源失败应当有实际尝试与哈希回执，不接受仅凭文字声明。" * 4,
        "attempts": [
            {"receipt_path": "missing.json", "receipt_hash": "invalid", "attempted_at": "2026-10-28T01:00:00+08:00"}
        ],
    }
    result = verify_outputs("002112", WINDOW, DATES, records()[1:], [], [exception])
    assert result["horizons"][0]["status"] == "ACTION_REQUIRED"


@pytest.mark.parametrize("problem", [None, "pending_label", "wrong_fund", "tampered_source"])
def test_external_failure_requires_matching_period_and_original_evidence(tmp_path, problem):
    source = tmp_path / "response.raw"
    source.write_bytes(b"actual failed source response")
    at = datetime.now(ZONE).isoformat()
    proof = {
        "fund_code": "002112",
        "target_date": WINDOW["target_nav_date"],
        "reason": "SOURCE_UNAVAILABLE",
        "attempted_at": at,
        "outcome": "HTTP_ERROR",
        "source_evidence": [{"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
    }
    if problem == "pending_label":
        proof["outcome"] = "WAITING_REVIEW"
    elif problem == "wrong_fund":
        proof["fund_code"] = "000001"
    elif problem == "tampered_source":
        source.write_bytes(b"changed")
    receipt = tmp_path / "attempt.json"
    save(receipt, proof)
    exception = {
        "fund_code": "002112",
        "horizon": "T1",
        "target_date": WINDOW["target_nav_date"],
        "reason": "SOURCE_UNAVAILABLE",
        "explanation": "实际访问公开来源失败，已经检查正常入口并保留本基金、本期的真实尝试和原始响应。" * 3,
        "attempts": [{"receipt_path": str(receipt), "receipt_hash": digest(proof), "attempted_at": at}],
    }
    result = verify_outputs("002112", WINDOW, DATES, records()[1:], [], [exception])
    assert result["horizons"][0]["status"] == ("ACTION_REQUIRED" if problem else "EXTERNAL_BLOCKED")


def test_yesterday_notice_success_cannot_represent_today_coverage():
    news = {"status": "SUCCEEDED", "catalog_complete": True, "coverage": {"end": "2026-10-27"}, "errors": []}
    actions = recovery_actions("002112", None, None, news, None, "2026-10-27", "2026-10-28")
    assert any(a["action"] == "RECOVER_NOTICE_COVERAGE" for a in actions)


@pytest.mark.parametrize("payload", [{}, None, ["002112", "001021"], ["002112", "002112"], ["not-a-fund"]])
def test_malformed_owner_scope_cannot_turn_into_empty_completed_scope(monkeypatch, payload):
    def handler(request):
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("app.commands.fund_prediction_followthrough.httpx.Client", lambda **_: client)
    settings = SimpleNamespace(
        core_service_base_url="http://core", ai_service_token=SimpleNamespace(get_secret_value=lambda: "test")
    )
    with pytest.raises(ValueError, match="OWNER_SCOPE_"):
        owner_scope(settings, "bound-owner")


def test_new_target_without_owner_archive_does_not_query_uuid_with_empty_integer_set(monkeypatch):
    """新目标尚无结果必须返回待办，不能让UUID空集类型错误中断完成检查。"""
    from unittest.mock import MagicMock

    from app.commands import fund_prediction_followthrough as command

    core = MagicMock()
    core.connect.return_value.__enter__.return_value.execute.return_value.mappings.return_value = []
    monkeypatch.setattr(command, "create_engine", lambda _: core)
    monkeypatch.setattr(
        command,
        "dotenv_values",
        lambda _: {"FUND_CORE_DB_USERNAME": "test", "FUND_CORE_DB_PASSWORD": "test", "FUND_CORE_DB_NAME": "test"},
    )
    monkeypatch.setattr(command, "get_engine", lambda: pytest.fail("空本人归档不应查询Python UUID原记录"))
    assert command.saved_records("test-owner", ["002112"], "2026-10-29", "unused.env") == []
    core.dispose.assert_called_once()


def test_calendar_action_for_long_period_does_not_invalidate_saved_t1():
    result = verify_outputs(
        "002112", WINDOW, DATES, records(), [{"action": "RECHECK_CURRENT_INPUT_AND_CALENDAR", "horizon": "T5_V1"}]
    )
    assert result["horizons"][0]["status"] == "SAVED_AND_READ_BACK"
    assert result["horizons"][1]["status"] == "ACTION_REQUIRED"


@pytest.mark.parametrize("wrong_fund", [False, True])
def test_original_calendar_boundary_is_separate_from_material_actions(wrong_fund):
    dates = deepcopy(DATES)
    dates["T5_V1"] = {
        "endDateStatus": "OUTSIDE_VERIFIED_CALENDAR",
        "fundCode": "000001" if wrong_fund else "002112",
        "calendarSourceHash": "a" * 64,
        "backendFailure": {
            "code": "CALENDAR_RANGE_MISSING",
            "details": {"calendarId": "verified", "coverageEnd": "2026-10-30", "requestedDate": "2026-11-02"},
        },
    }
    result = verify_outputs("002112", WINDOW, dates, records(), [])
    assert result["horizons"][0]["status"] == "SAVED_AND_READ_BACK"
    assert result["horizons"][1]["status"] == ("ACTION_REQUIRED" if wrong_fund else "EXTERNAL_BLOCKED")
    with_material_gap = verify_outputs("002112", WINDOW, dates, records(), [{"action": "REVIEW_NEW_REPORT"}])
    assert with_material_gap["horizons"][1]["status"] == "ACTION_REQUIRED"


def test_command_preserves_t5_when_later_period_calendar_fails(monkeypatch):
    from datetime import date

    from app.commands.fund_prediction_followthrough import horizon_dates
    from app.services import comprehensive_multi_prediction, prediction_contract

    cal = SimpleNamespace(calendar_id="own", coverage_end=date(2026, 11, 3), source_hash="a" * 64)
    monkeypatch.setattr(comprehensive_multi_prediction, "review_calendar", lambda *a: cal)

    def target(calendar, at, horizon):
        if horizon == "T5_V1":
            return {"endDateStatus": "RESOLVED", "endDate": "2026-11-03"}
        raise prediction_contract.PredictionFailure(
            "CALENDAR_RANGE_MISSING",
            "CALENDAR",
            "基金估值日历未覆盖所需日期",
            details={"calendarId": "own", "coverageEnd": "2026-11-03", "requestedDate": "2026-11-04"},
        )

    monkeypatch.setattr(prediction_contract, "target_dates", target)
    dates, actions, failures = horizon_dates({}, "002112", datetime.now(ZONE))
    assert dates["T5_V1"]["endDateStatus"] == "RESOLVED"
    assert dates["T20_V1"]["endDateStatus"] == "OUTSIDE_VERIFIED_CALENDAR"
    assert not actions and len(failures) == 2


def test_insufficient_verified_sessions_preserves_original_backend_failure():
    dates = deepcopy(DATES)
    dates["T5_V1"] = {
        "endDateStatus": "OUTSIDE_VERIFIED_CALENDAR",
        "fundCode": "002112",
        "calendarSourceHash": "a" * 64,
        "backendFailure": {
            "code": "CALENDAR_RANGE_MISSING",
            "details": {"calendarId": "own", "coverageEnd": "2026-10-30", "requestedDate": "2026-10-28"},
        },
        "coverageProof": {
            "baseDate": WINDOW["base_nav_date"],
            "availableSessionsAfterBase": 3,
            "requiredSessions": 5,
            "verifiedCoverageEnd": "2026-10-30",
        },
    }
    assert verify_outputs("002112", WINDOW, dates, records(), [])["horizons"][1]["status"] == "EXTERNAL_BLOCKED"
    dates["T5_V1"]["coverageProof"]["availableSessionsAfterBase"] = 5
    assert verify_outputs("002112", WINDOW, dates, records(), [])["horizons"][1]["status"] == "ACTION_REQUIRED"


@pytest.mark.parametrize(
    "action",
    [
        "VERIFY_NOTICE_IDENTITY_OR_SOURCE",
        "SUPPLEMENT_HOLDING_MATERIALS",
        "REVIEW_NEW_REPORT",
        "REVIEW_NEW_LOOKTHROUGH_REPORT",
    ],
)
def test_official_holiday_does_not_close_processable_material_actions(monkeypatch, action):
    """即使声称有休市回执，公告身份、新证券和新报告仍必须先完成处理。"""
    from app.services.fund_prediction_followthrough import _verified_exception

    monkeypatch.setattr(
        "app.services.fund_exposure_common.read",
        lambda *_: pytest.fail("可处理的资料行动须在休市回执核验前被拒绝关闭"),
    )
    exception = {
        "target_date": WINDOW["target_nav_date"],
        "reason": "VERIFIED_NON_TRADING_TARGET",
        "explanation": "已核官方休市日历，仍不允许因此掩盖公告、持仓和报告方面可处理的内部行动。" * 4,
        "attempts": [{}],
    }
    assert not _verified_exception(exception, WINDOW, [{"action": action}])
