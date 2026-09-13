"""自动留档关键边界：真实截止、重复预算、节假日、故障和原文/时间证据完整性。"""

import json
from datetime import date, datetime

import pytest
from app.services import direction_1d_overnight_collect as collect
from app.services.direction_1d_protocol import ZONE

SOURCE = {
    "source_id": "fixture-source",
    "source_code": "TUSHARE_PRO_FUND",
    "enabled": True,
    "authorization_verified_at": "2026-09-13T09:00:00+08:00",
    "retention_days": 365,
    "rate_limit_per_minute": 200,
}


def at(value):
    return datetime.fromisoformat("2026-09-14T" + value + "+08:00")


@pytest.fixture
def root(tmp_path):
    proof = tmp_path / "proof.json"
    collect.write_new(
        proof,
        {
            "status": "PERMISSION_AVAILABLE_NOT_USAGE_OR_TIMING_PROOF",
            "http_status": 200,
            "provider_code": 0,
            "row_count": 1,
            "reservation": {"api_name": "index_global", "params": {"ts_code": "SPX"}},
        },
    )
    target = tmp_path / "live"
    collect.initialize(target, proof, clock=lambda: datetime(2026, 9, 13, 9, tzinfo=ZONE), source_reader=lambda: SOURCE)
    return target


def response(plan, *, omit_last=False):
    rows = [
        ["SPX", d.replace("-", ""), 100 + i, 99 + i, (1 / (99 + i)) * 100]
        for i, d in enumerate(plan["required_us_dates"])
    ]
    if omit_last:
        rows.pop()
    return json.dumps({"code": 0, "msg": "", "data": {"fields": collect.FIELDS, "items": rows}}).encode()


def execute(root, stamp="07:30:00", *, body=None, source=SOURCE, query=None):
    plan = collect.day_plan(date(2026, 9, 14))
    return collect.tick(
        root, clock=lambda: at(stamp), query=query or (lambda *_: body or response(plan)), source_reader=lambda: source
    )


def test_success_raw_receipt_and_no_duplicate_request(root):
    first = execute(root)
    assert first["day"]["status"] == "READY_OBSERVED"
    assert first["attempt"]["api_calls"] == 1
    raw = root / "days/2026-09-14/0730-response.json"
    saved = raw.read_bytes()
    second = execute(root, query=lambda *_: pytest.fail("同槽不得重复请求"))
    assert second["attempt"]["status"] == "ALREADY_ATTEMPTED"
    assert raw.read_bytes() == saved
    assert first["day"]["observed_at"] == at("07:30:00").isoformat()
    assert first["day"]["provider_first_publication_verified"] is False


def test_missing_data_is_not_ready_and_later_slot_can_succeed(root):
    plan = collect.day_plan(date(2026, 9, 14))
    first = execute(root, body=response(plan, omit_last=True))
    assert first["attempt"]["status"] == "INCOMPLETE"
    assert first["day"]["status"] == "WAITING"
    second = execute(root, "07:50:00")
    assert second["day"]["selected_slot"] == "0750"
    assert len(second["day"]["attempts"]) == 2
    assert (root / "days/2026-09-14/0730-response.json").exists()


def test_late_observation_never_backfills_deadline(root):
    value = execute(root, "08:05:00")
    assert value["attempt"]["status"] == "LATE"
    assert value["day"]["status"] == "MISSED_DEADLINE"
    assert value["day"]["usable_before_u08"] is False


def test_response_arrives_before_cutoff_but_disk_finishes_after_is_late(root):
    plan = collect.day_plan(date(2026, 9, 14))
    stamps = iter([at("07:59:58"), at("07:59:58"), at("07:59:59"), at("08:00:00")])
    result = collect.collect_slot(
        root,
        plan,
        "0758",
        collect.load_contract(root),
        query=lambda *_: response(plan),
        clock=lambda: next(stamps),
        source_reader=lambda: SOURCE,
    )
    assert result["status"] == "LATE"


def test_clock_rollback_is_failure(root):
    plan = collect.day_plan(date(2026, 9, 14))
    stamps = iter([at("07:58:00"), at("07:58:00"), at("07:57:59")])
    result = collect.collect_slot(
        root,
        plan,
        "0758",
        collect.load_contract(root),
        query=lambda *_: response(plan),
        clock=lambda: next(stamps),
        source_reader=lambda: SOURCE,
    )
    assert result["status"] == "FAILED"
    assert collect.read(root / "days/2026-09-14/0758-receipt.json")["error_code"] == "OVERNIGHT_CLOCK_DISCONTINUITY"


@pytest.mark.parametrize("stamp", ["06:00:00", "07:46:00", "08:00:00", "09:00:00"])
def test_start_when_available_never_replays_old_slots(root, stamp):
    result = execute(root, stamp, query=lambda *_: pytest.fail("错过时点不能补跑请求"))
    assert result["attempt"] is None


@pytest.mark.parametrize("day", [date(2026, 9, 13), date(2026, 9, 25), date(2026, 10, 1)])
def test_weekends_and_cn_holidays_do_not_call_provider(root, day):
    result = collect.tick(
        root,
        clock=lambda: datetime.combine(day, at("09:30:00").timetz()),
        query=lambda *_: pytest.fail("非交易日不联网"),
        source_reader=lambda: SOURCE,
    )
    assert result["day"]["status"] == "NON_TRADING_DAY"


def test_year_without_calendar_fails_closed(root):
    with pytest.raises(ValueError, match="CALENDAR_UNAVAILABLE"):
        collect.tick(root, clock=lambda: datetime(2027, 1, 4, 7, 30, tzinfo=ZONE), query=lambda *_: pytest.fail())


def test_china_holiday_includes_all_us_sessions_and_us_holiday_is_zero():
    plan = collect.day_plan(date(2026, 10, 8))
    assert plan["new_us_dates"] == ["2026-09-30", "2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07"]
    plan = collect.day_plan(date(2026, 9, 8))
    assert plan["new_us_dates"] == []
    assert collect.validate_response(response(plan), plan)["diagnostic_return"] == 0


@pytest.mark.parametrize("kind", ["code", "future", "duplicate", "nan", "return", "business"])
def test_invalid_response_is_failure_without_usable_flag(root, kind):
    plan = collect.day_plan(date(2026, 9, 14))
    body = json.loads(response(plan))
    row = body["data"]["items"][0]
    if kind == "code":
        row[0] = "IXIC"
    if kind == "future":
        row[1] = "20260914"
    if kind == "duplicate":
        body["data"]["items"].append(row)
    if kind == "nan":
        row[2] = float("nan")
    if kind == "return":
        row[4] = 50
    if kind == "business":
        body["code"] = 40203
    value = execute(root, body=json.dumps(body).encode())
    assert value["attempt"]["status"] == "FAILED"
    assert value["day"]["usable_before_u08"] is False


def test_revoked_source_skips_external_request(root):
    value = execute(root, source={**SOURCE, "enabled": False}, query=lambda *_: pytest.fail("停用来源不能请求"))
    assert value["attempt"]["status"] == "FAILED"
    assert value["attempt"]["api_calls"] == 0


def test_timeout_does_not_retry_and_no_exception_secret_saved(root):
    def fail(*_):
        raise RuntimeError("password=FAKE_SENSITIVE_TEST")

    value = execute(root, query=fail)
    assert value["attempt"]["status"] == "FAILED"
    assert "FAKE_SENSITIVE_TEST" not in (root / "days/2026-09-14/0730-receipt.json").read_text(encoding="utf-8")
    assert execute(root, query=lambda *_: pytest.fail())["attempt"]["status"] == "ALREADY_ATTEMPTED"


def test_raw_or_receipt_tampering_is_rejected(root):
    execute(root)
    path = root / "days/2026-09-14/0730-response.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="RAW_HASH_CHANGED"):
        collect.day_status(root, date(2026, 9, 14), at("08:10:00"))


def test_first_success_retained_when_later_request_fails(root):
    execute(root)
    result = execute(root, "07:58:00", source={**SOURCE, "enabled": False})
    assert result["day"]["status"] == "READY_OBSERVED"
    assert result["day"]["selected_slot"] == "0730"


def test_diagnostic_probe_cannot_be_counted_as_timely(root):
    stamp = datetime(2026, 9, 13, 9, tzinfo=ZONE)
    value = collect.probe(
        root,
        clock=lambda: stamp,
        source_reader=lambda: SOURCE,
        query=lambda s, e: response({"required_us_dates": [str(s)]}),
    )
    assert value["status"] == "DIAGNOSTIC_ONLY"
    assert value["usable_before_u08"] is False
    assert not (root / "days").exists()


def test_installed_time_cannot_be_backdated(root):
    with pytest.raises(ValueError, match="CLOCK_BEFORE_INSTALL"):
        collect.tick(root, clock=lambda: datetime(2026, 9, 12, 7, tzinfo=ZONE))
