"""报告未到披露期与真实缺件必须分开，测试样本不作为来源证明。"""

from datetime import date, timedelta

from app.services.fund_quality_v2_report_availability import disclosure_window


def document():
    return {
        "text": (
            "工作日：指上海证券交易所、深圳证券交易所的正常交易日。"
            "每个季度结束之日起15个工作日内编制完毕并予以公告。"
        ),
        "ref": {"sha256": "TEST_ONLY"},
    }


def test_quarter_close_not_yet_due_and_no_claim_of_historical_legal_deadline(monkeypatch):
    from app.services import direction_1d_protocol

    monkeypatch.setattr(direction_1d_protocol, "calendar", lambda: ((date(2026, 1, 1), date(2026, 12, 31)), "TEST"))
    value = disclosure_window(
        ["2026-06-30", "2026-09-30"], as_of=date(2026, 9, 30), prospectus=document(), published_dates=set()
    )
    assert value["required_report_dates"] == ["2026-06-30"]
    assert value["checkpoints"][-1]["state"] == "NOT_YET_DUE"
    assert "NOT_LEGAL_OVERDUE" in value["checkpoints"][0]["state"]


def test_missing_calendar_definition_is_unknown_not_automatic_missing_report():
    value = disclosure_window(["2026-09-30"], as_of=date(2026, 10, 17), prospectus={}, published_dates=set())
    assert value["required_report_dates"] == []
    assert value["checkpoints"][0]["state"].endswith("UNVERIFIED")


def test_fifteen_verified_sessions_and_actual_early_publication(monkeypatch):
    from app.services import direction_1d_protocol

    sessions = tuple(date(2026, 9, 30) + timedelta(days=i) for i in range(17))
    monkeypatch.setattr(direction_1d_protocol, "calendar", lambda: (sessions, "TEST"))
    assert disclosure_window(["2026-09-30"], as_of=date(2026, 10, 15), prospectus=document(), published_dates=set())[
        "required_report_dates"
    ] == ["2026-09-30"]
    result = disclosure_window(["2026-09-30"], as_of=date(2026, 9, 30), prospectus={}, published_dates={"2026-09-30"})
    assert result["checkpoints"][0]["state"] == "PUBLISHED_ORIGINAL_AVAILABLE"
