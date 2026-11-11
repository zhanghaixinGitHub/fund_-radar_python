"""仅用人工测试年历，验证跨年能力；绝不将样例登记到生产日历。"""

import json
from datetime import date, datetime

import pytest
from app.services import trading_calendar as calendars
from app.services.direction_1d_protocol import window
from app.services.prediction_contract import target_dates
from app.services.prediction_features import public_calendar


def extension(tmp_path, monkeypatch):
    definition = calendars.load_current_calendar().definition.model_dump(mode="json")
    definition.update(
        version="CN_A_SHARE_2027_V1", coverage_start="2027-01-01", coverage_end="2027-12-31", reviewed_on="2026-11-11"
    )
    notice = definition["years"][0]
    notice.update(year=2027, announced_on="2026-11-01", closed_ranges=[["2027-01-01", "2027-01-03"]])
    # URL只用来验证来源域；本文件的日期是人工夹具，并非真实2027官方安排。
    notice["sources"] = ["https://www.sse.com.cn/test-fixture", "https://www.szse.cn/test-fixture"]
    built = calendars.build_calendar(calendars.CalendarDefinition.model_validate(definition))
    (tmp_path / "test-year.json").write_text(json.dumps(definition), encoding="utf-8")
    manifest = tmp_path / "verified_extensions.json"
    manifest.write_text(
        json.dumps({"extensions": [{"definition_file": "test-year.json", "definition_hash": built.content_hash}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(calendars, "EXTENSION_MANIFEST", manifest)
    return built, manifest


def test_registered_2027_supports_year_boundary_and_half_year(tmp_path, monkeypatch):
    built, _ = extension(tmp_path, monkeypatch)
    assert calendars.load_prediction_calendar(date(2027, 1, 4)).content_hash == built.content_hash
    w = window(datetime.fromisoformat("2027-01-04T00:30:00+08:00"))
    assert w["base_nav_date"] == "2026-12-31" and w["target_nav_date"] == "2027-01-04"
    calendar = public_calendar({"fund_code": "123456", "fund_name": "测试混合", "fund_type": "MIXED"})
    dates = target_dates(calendar, datetime.fromisoformat("2026-11-11T00:30:00+08:00"), "M6_V1")
    assert dates["endDateStatus"] == "RESOLVED" and dates["endDate"].startswith("2027-05")
    assert calendars.load_current_calendar().content_hash == calendars.EXPECTED_CURRENT_CALENDAR_HASH


def test_tampered_extension_and_missing_year_refuse_fallback(tmp_path, monkeypatch):
    _, manifest = extension(tmp_path, monkeypatch)
    raw = json.loads(manifest.read_text())
    raw["extensions"][0]["definition_hash"] = "0" * 64
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        calendars.load_live_calendars()
    manifest.write_text('{"extensions": []}', encoding="utf-8")
    with pytest.raises(calendars.CalendarCoverageError):
        calendars.load_prediction_calendar(date(2027, 1, 4))


def test_missing_originals_cannot_register_real_year(tmp_path):
    from app.commands.fund_calendar_register import register

    definition = tmp_path / "definition.json"
    definition.write_text(calendars.load_current_calendar().definition.model_dump_json(), encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"sources": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        register(definition, evidence, tmp_path / "verified_extensions.json")
    assert not (tmp_path / "verified_extensions.json").exists()


def test_registration_exports_matching_java_version_without_overwriting_old_calendar(tmp_path, monkeypatch):
    import hashlib

    from app.commands.fund_calendar_register import register

    built, _ = extension(tmp_path, monkeypatch)
    definition = built.definition.model_dump(mode="json")
    originals = []
    for index, url in enumerate(definition["years"][0]["sources"]):
        path = tmp_path / f"synthetic-original-{index}.txt"
        path.write_text("人工测试原件，不是官方年历。", encoding="utf-8")
        originals.append(
            {
                "url": url,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "review_note": "仅验证隔离登记链路的人工原件。",
            }
        )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps({"sources": originals, "reviewed_closed_ranges": definition["years"][0]["closed_ranges"]}),
        encoding="utf-8",
    )
    manifest = tmp_path / "isolated-publication" / "verified_extensions.json"
    receipt = register(tmp_path / "test-year.json", evidence, manifest)
    entries = json.loads(manifest.read_text())["extensions"]
    exported = json.loads((manifest.parent / entries[0]["java_file"]).read_text())
    assert receipt["status"] == "REGISTERED" and exported["version"] == receipt["calendar_version"]
    assert exported["sessions"][-1].startswith("2027-")
    assert calendars.load_calendar().content_hash == calendars.EXPECTED_CALENDAR_HASH
    with pytest.raises(ValueError, match="CONTIGUOUS"):
        register(tmp_path / "test-year.json", evidence, manifest)
