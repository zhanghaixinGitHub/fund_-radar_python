"""近期补齐工具的关键边界测试；全部使用本地替身，不碰业务缓存和数据库。"""

import json

import pytest

from scripts.fund_002112_recent_acceptance_v1 import merge_series, parse_sector, suspension_state
from scripts.fund_002112_recent_completion_v1 import read, save


@pytest.fixture
def event():
    return {
        "start_date": "2025-04-08",
        "resume_date": "2025-04-22",
        "start_available_at": "2025-04-09T08:00:00+08:00",
        "resume_available_at": "2025-04-23T08:00:00+08:00",
    }


def test_suspension_before_start_notice_stays_unknown(event):
    row = suspension_state("2025-04-08", "2025-04-09T07:59:59+08:00", event)
    assert row == {"state": "UNKNOWN", "price": None, "return": None, "resume_known": False}


def test_suspension_never_fills_zero_or_leaks_resume(event):
    row = suspension_state("2025-04-21", "2025-04-22T08:00:00+08:00", event)
    assert row == {"state": "OFFICIAL_SUSPENSION_NO_TRADE", "price": None, "return": None, "resume_known": False}


def test_resume_notice_does_not_invent_a_quote(event):
    row = suspension_state("2025-04-22", "2025-04-23T08:00:00+08:00", event)
    assert row == {"state": "RESUMED_REQUIRE_OBSERVED_QUOTE", "price": None, "return": None, "resume_known": True}


def test_timezone_equivalent_instants_are_equal(event):
    assert suspension_state("2025-04-08", "2025-04-09T00:00:00+00:00", event) == suspension_state(
        "2025-04-08", "2025-04-09T08:00:00+08:00", event
    )


@pytest.mark.parametrize("cutoff", ["2025-04-09T08:00:00", "2025-04-08T08:00:00+08:00"])
def test_no_naive_time_or_future_input(cutoff, event):
    with pytest.raises(ValueError):
        suspension_state("2025-04-08", cutoff, event)


def raw_row(code="399998.SZ", close=10, previous=9):
    return json.dumps(
        {
            "code": 0,
            "data": {
                "fields": ["ts_code", "trade_date", "close", "pre_close"],
                "items": [[code, "20260916", close, previous]],
            },
        }
    )


@pytest.mark.parametrize(
    "code,close,previous",
    [
        ("399986.SZ", 10, 9),
        ("399998.SZ", 0, 9),
        ("399998.SZ", 10, None),
        ("399998.SZ", True, 9),
    ],
)
def test_wrong_identity_and_invalid_prices_rejected(code, close, previous):
    with pytest.raises(ValueError):
        parse_sector(raw_row(code, close, previous), "399998.SZ")


def test_duplicate_date_is_not_silently_overwritten():
    raw = json.loads(raw_row())
    raw["data"]["items"] *= 2
    with pytest.raises(ValueError):
        parse_sector(json.dumps(raw), "399998.SZ")


def test_later_price_revision_is_not_silently_accepted():
    existing = {"2026-09-16": {"close": "10", "pre_close": "9"}}
    merge_series(existing, {"2026-09-16": {"close": "10.00", "pre_close": "9.0"}})
    with pytest.raises(ValueError, match="VERSION_CONFLICT"):
        merge_series(existing, {"2026-09-16": {"close": "11", "pre_close": "9"}})
    assert existing["2026-09-16"]["close"] == "10.00"


def test_output_is_append_only_and_bad_input_hash_rejected(tmp_path):
    p = tmp_path / "evidence.json"
    save(p, {"old_stop_reason": "KEEP"})
    with pytest.raises(FileExistsError):
        save(p, {"old_stop_reason": None})
    assert read(p) == {"old_stop_reason": "KEEP"}
    invalid = tmp_path / "bad.json"
    invalid.write_text('{"hash":"invalid","payload":{"a":1}}', encoding="utf-8")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        read(invalid)
