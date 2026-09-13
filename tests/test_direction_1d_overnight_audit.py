"""验证跨时区、长假累计、缺日拒绝和历史到达时间未知边界；测试数据不来自真实基金答案。"""

import json
from copy import deepcopy

import pytest
from app.services import direction_1d_overnight_audit as audit


@pytest.fixture
def history():
    """构造逐交易日连续变化的价格，便于独立核对是否选中了未来或遗漏的交易日。"""
    result = {}
    previous = 100.0
    for index, day in enumerate(audit.sessions()):
        close = 101.0 + index
        result[day] = {
            "ts_code": "SPX",
            "trade_date": day.replace("-", ""),
            "close": close,
            "pre_close": previous,
            "pct_chg": (close / previous - 1) * 100,
        }
        previous = close
    return result


def payloads(history):
    return [
        {
            "fields": audit.FIELDS,
            "items": [[r[k] for k in audit.FIELDS] for d, r in history.items() if d.startswith(str(y))],
        }
        for y in audit.YEARS
    ]


def test_calendar_counts_and_new_year_juneteenth():
    market = audit.sessions()
    assert [sum(d.startswith(str(y)) for d in market) for y in audit.YEARS] == [252, 251, 250, 252]
    assert "2021-12-31" in market
    assert "2021-06-18" in market
    assert "2022-06-20" not in market
    assert "2023-06-19" not in market


@pytest.mark.parametrize(
    ("day", "china_close"),
    [
        ("2024-03-08", "2024-03-09T05:00:00+08:00"),
        ("2024-03-11", "2024-03-12T04:00:00+08:00"),
        ("2024-11-01", "2024-11-02T04:00:00+08:00"),
        ("2024-11-04", "2024-11-05T05:00:00+08:00"),
        ("2024-07-03", "2024-07-04T01:00:00+08:00"),
        ("2024-11-29", "2024-11-30T02:00:00+08:00"),
        ("2024-12-24", "2024-12-25T02:00:00+08:00"),
    ],
)
def test_daylight_saving_and_early_close(day, china_close):
    assert audit.sessions()[day].isoformat() == china_close


def test_valid_history_and_continuity(history):
    assert audit.validate_history(payloads(history)) == history
    assert audit.coverage(history, audit.sessions())["status"] == "COMPLETE"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("ts_code", "IXIC", "INDEX_DATE_OR_DUPLICATE"),
        ("trade_date", "20250102", "INDEX_DATE_OR_DUPLICATE"),
        ("trade_date", "2024-01-02", "DATE_INVALID"),
        ("close", 0, "NONPOSITIVE_PRICE"),
        ("pre_close", float("nan"), "NUMBER_INVALID"),
        ("close", True, "NUMBER_INVALID"),
        ("pct_chg", 77, "RETURN_INCONSISTENT"),
    ],
)
def test_invalid_history_rejected(history, field, value, error):
    data = payloads(history)
    data[0]["items"][0][audit.FIELDS.index(field)] = value
    with pytest.raises(ValueError, match=error):
        audit.validate_history(data)


def test_duplicate_rejected(history):
    data = payloads(history)
    data[0]["items"].append(data[0]["items"][0])
    with pytest.raises(ValueError, match="DUPLICATE"):
        audit.validate_history(data)


def test_day_after_us_holiday_is_real_zero(history):
    row = audit.align_dates([("2024-01-15", "2024-01-16")], history, audit.sessions())[0]
    assert row["us_sessions"] == []
    assert row["diagnostic_return"] == 0
    assert row["input_available_at"] is None
    assert row["historical_first_version_verified"] is False


def test_ordinary_overnight_does_not_use_target_day_us_close(history):
    row = audit.align_dates([("2024-01-08", "2024-01-09")], history, audit.sessions())[0]
    assert row["baseline_us_date"] == "2024-01-05"
    assert row["us_sessions"] == ["2024-01-08"]
    assert row["latest_us_date"] == "2024-01-08"
    assert row["diagnostic_return"] == history["2024-01-08"]["close"] / history["2024-01-05"]["close"] - 1


def test_china_holiday_accumulates_all_us_closes(history):
    row = audit.align_dates([("2024-09-30", "2024-10-08")], history, audit.sessions())[0]
    assert row["us_sessions"] == ["2024-09-30", "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-07"]
    assert row["diagnostic_return"] == history["2024-10-07"]["close"] / history["2024-09-27"]["close"] - 1


def test_missing_session_cannot_be_imputed_as_holiday(history):
    del history["2024-01-08"]
    assert audit.coverage(history, audit.sessions())["missing"] == ["2024-01-08"]
    with pytest.raises(ValueError, match="COVERAGE_INCOMPLETE"):
        audit.align_dates([("2024-01-08", "2024-01-09")], history, audit.sessions())


def test_previous_close_revision_detected(history):
    history["2024-01-08"]["pre_close"] += 1
    assert audit.coverage(history, audit.sessions())["previous_close_inconsistent"] == ["2024-01-08"]


@pytest.mark.parametrize(
    "pair", [("2024-01-08", "2024-01-10"), ("2024-01-06", "2024-01-08"), ("2024-12-31", "2025-01-02")]
)
def test_bad_or_protected_cn_pair_rejected(history, pair):
    with pytest.raises(ValueError, match="CN_PAIR_INVALID"):
        audit.align_dates([pair], history, audit.sessions())


def test_duplicate_pair_rejected(history):
    with pytest.raises(ValueError, match="PAIR_DUPLICATE"):
        audit.align_dates([("2024-01-08", "2024-01-09")] * 2, history, audit.sessions())


def test_insufficient_initial_baseline_rejected(history):
    with pytest.raises(ValueError, match="BASE_OR_WINDOW_UNCOVERED"):
        audit.align_dates([("2021-01-04", "2021-01-05")], history, audit.sessions())


def test_today_download_cannot_supply_historical_timestamp(history):
    # 手工塞入看似合理的时间也没有资格；必须由另外经过验证的证据机制提供，不能从行情行反推。
    history = deepcopy(history)
    history["2024-01-08"]["available_at"] = "2024-01-09T05:00:00+08:00"
    history["2024-01-08"]["verified"] = True
    row = audit.align_dates([("2024-01-08", "2024-01-09")], history, audit.sessions())[0]
    assert row["input_available_at"] is None
    assert row["status"] == "DIAGNOSTIC_ONLY_PROVIDER_TIMING_UNVERIFIED"


@pytest.fixture
def audit_files(tmp_path, history):
    """用完整但合成的四年响应构建回执；离线验证整体覆盖通过仍不能放行训练。"""
    base, data, output = (tmp_path / name for name in ("base", "data", "audit"))
    base.mkdir()
    data.mkdir()
    for name in ("universe", "exam"):
        audit.write_new(base / f"{name}.json", [{"fund_code": "TEST", "t": "2024-01-08", "u": "2024-01-09"}])
    audit.write_new(data / "probe-result.json", {"status": "PERMISSION_AVAILABLE_NOT_USAGE_OR_TIMING_PROOF"})
    audit.write_new(
        data / "history-reserved.json",
        {"api_name": "index_global", "index": "SPX", "years": list(audit.YEARS), "max_requests": 4},
    )
    audit.write_new(data / "source-audit.json", {"fixture": True})
    audit.write_new(data / "probe-reserved.json", {"fixture": True})
    requests = []
    for year, payload in zip(audit.YEARS, payloads(history), strict=True):
        path = data / f"history-{year}.json"
        audit.write_new(path, payload)
        item = {"year": year, "http_status": 200, "data_file_sha256": audit.file_hash(path)}
        requests.append(item)
        audit.write_new(data / f"history-{year}-receipt.json", item)
        audit.write_new(data / f"history-{year}-reserved.json", {"year": year})
    audit.write_new(data / "history-receipt.json", {"status": "RECEIVED", "api_calls": 4, "requests": requests})
    return base, data, output


def test_complete_audit_still_blocks_training_and_replays(audit_files):
    base, data, output = audit_files
    report = audit.save_audit(base, data, output)
    assert report["coverage"]["status"] == "COMPLETE"
    assert report["status"] == "BLOCKED_HISTORICAL_TIMING"
    assert report["training_ready"] is False
    assert audit.verify(output)["verification"] == "PASS"
    with pytest.raises(FileExistsError):
        audit.save_audit(base, data, output)


def test_audit_forged_ready_flag_rejected_even_with_updated_file_hash(audit_files):
    base, data, output = audit_files
    audit.save_audit(base, data, output)
    path = output / "readiness.json"
    report = audit.read(path)
    report.update(training_ready=True, status="READY")
    path.write_text(json.dumps(report), encoding="utf-8")
    receipt_path = output / "audit-receipt.json"
    receipt = audit.read(receipt_path)
    receipt["files"]["readiness.json"] = audit.file_hash(path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="AUDIT_REPLAY_MISMATCH"):
        audit.verify(output)


def test_source_tampering_rejected(audit_files):
    base, data, output = audit_files
    audit.save_audit(base, data, output)
    path = data / "history-2024.json"
    source = audit.read(path)
    source["items"][0][2] += 100
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="RECEIPT_OR_HASH_INVALID"):
        audit.verify(output)


def test_same_dates_but_different_exam_fund_rejected(audit_files):
    base, data, _ = audit_files
    path = base / "exam.json"
    source = audit.read(path)
    source[0]["fund_code"] = "OUTSIDE_UNIVERSE"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="EXAM_NOT_IN_UNIVERSE"):
        audit.inspect(base, data)
