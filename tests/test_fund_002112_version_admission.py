"""准入边界测试使用合成来源，不加载真实模型、净值数据库或封存答案。"""

from copy import deepcopy
from datetime import date, timedelta

import pytest
from app.services.fund_002112_version_admission import load_evidence, nav_value, row_at, version_at
from app.services.fund_002112_zero_fit_review import digest, file_hash


def evidence(tmp_path, *, value="1.0000", version="2023-03-02", kind="PROVIDER_VERSION"):
    """模拟已经完成原文审阅的条目；这不是可登记到真实协议的第三方证据。"""
    source = tmp_path / (digest([value, version, kind]) + ".txt")
    source.write_text(f"SYNTHETIC ONLY 160323 2023-03-01 {value} version {version}", encoding="utf-8")
    facts = {
        "key": "NAV:160323:2023-03-01",
        "type": "NAV",
        "fund_code": "160323",
        "business_date": "2023-03-01",
        "publication_date": "2023-03-02",
        "version_publication_date": version,
        "value": value,
    }
    entry = {
        "kind": kind,
        "source": {"path": str(source), "sha256": file_hash(source)},
        "source_locator": "synthetic line 1",
        "version_semantics": "test fixture only",
        "facts": facts,
    }
    if kind == "INDEPENDENT_ARCHIVE":
        entry.update(witness_date="2023-03-03", received_at="2026-09-28T12:00:00+08:00")
    return entry


def ready(entries):
    return load_evidence(entries, {digest(e) for e in entries})


def requirement(entry):
    return {k: v for k, v in entry["facts"].items() if k != "version_publication_date"}


def test_complete_evidence_really_passes(tmp_path):
    entry = evidence(tmp_path)
    assert version_at(requirement(entry), "2023-03-03", ready([entry]))["eligible"]


def test_current_download_of_independent_old_copy_is_allowed(tmp_path):
    entry = evidence(tmp_path, kind="INDEPENDENT_ARCHIVE")
    assert version_at(requirement(entry), "2023-03-04", ready([entry]))["eligible"]
    assert not version_at(requirement(entry), "2023-03-03", ready([entry]))["eligible"]


def test_self_declared_pass_does_not_replace_source_review(tmp_path):
    entry = evidence(tmp_path)
    entry.update(eligible=True, historical_version_verified=True)
    with pytest.raises(ValueError, match="NOT_IN_REVIEWED"):
        load_evidence([entry], set())


def test_edited_date_invalidates_review_digest(tmp_path):
    entry = evidence(tmp_path)
    reviewed = {digest(entry)}
    entry["facts"]["version_publication_date"] = "2023-03-01"
    with pytest.raises(ValueError, match="NOT_IN_REVIEWED"):
        load_evidence([entry], reviewed)


def test_changed_source_bytes_block(tmp_path):
    entry = evidence(tmp_path)
    from pathlib import Path

    Path(entry["source"]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        ready([entry])


@pytest.mark.parametrize("kind", ["CURRENT_DATABASE", "DATED_CATALOG", "PDF_METADATA", "URL_DATE"])
def test_weaker_evidence_cannot_be_promoted(tmp_path, kind):
    with pytest.raises(ValueError, match="NOT_VERSION_RECORD"):
        ready([evidence(tmp_path, kind=kind)])


@pytest.mark.parametrize("value", [None, "", "2023-13-01"])
def test_unknown_or_invalid_publication_never_backdated(tmp_path, value):
    entry = evidence(tmp_path)
    entry["facts"]["version_publication_date"] = value
    with pytest.raises((ValueError, TypeError)):
        ready([entry])


def test_same_day_publication_is_not_prior_day(tmp_path):
    entry = evidence(tmp_path)
    assert not version_at(requirement(entry), "2023-03-02", ready([entry]))["eligible"]


def test_later_correction_cannot_be_backfilled_or_bypassed(tmp_path):
    old, revised = evidence(tmp_path), evidence(tmp_path, value="1.0001", version="2023-03-10")
    index = ready([old, revised])
    assert version_at(requirement(old), "2023-03-09", index)["eligible"]
    assert not version_at(requirement(revised), "2023-03-09", index)["eligible"]
    assert not version_at(requirement(old), "2023-03-11", index)["eligible"]
    assert version_at(requirement(revised), "2023-03-11", index)["eligible"]


def test_conflicting_same_version_refuses_arbitrary_choice(tmp_path):
    first, other = evidence(tmp_path), evidence(tmp_path, value="1.0001")
    assert (
        version_at(requirement(first), "2023-03-04", ready([first, other]))["reason"]
        == "CONFLICTING_HISTORICAL_VERSIONS"
    )


@pytest.mark.parametrize(
    "field,value", [("fund_code", "017493"), ("business_date", "2023-02-28"), ("publication_date", "2023-03-01")]
)
def test_identity_and_original_date_cannot_be_switched(tmp_path, field, value):
    entry = evidence(tmp_path)
    req = requirement(entry)
    req[field] = value
    assert not version_at(req, "2023-03-04", ready([entry]))["eligible"]


def test_exact_decimal_not_rounded_percent(tmp_path):
    entry = evidence(tmp_path)
    req = requirement(entry)
    req["value"] = "1.000000000000001"
    assert not version_at(req, "2023-03-04", ready([entry]))["eligible"]
    req["value"] = "1.0000000000"
    assert version_at(req, "2023-03-04", ready([entry]))["eligible"]


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "Infinity"])
def test_invalid_nav_rejected(value):
    with pytest.raises(ValueError):
        nav_value(value)


def synthetic_complete_row(tmp_path):
    """合成完整依赖：61 个输入净值、一份报告、两个标签端点。"""
    prototype = evidence(tmp_path)
    entries, requirements, keys = [], {}, []
    for i in range(63):
        entry = deepcopy(prototype)
        day = (date(2023, 1, 1) + timedelta(days=i)).isoformat()
        f = entry["facts"]
        f.update(key="NAV:160323:" + day, business_date=day, publication_date=day, version_publication_date=day)
        entries.append(entry)
        requirements[f["key"]] = requirement(entry)
        keys.append(f["key"])
    report = deepcopy(prototype)
    report["facts"].update(
        key="REPORT:160323:2022-12-31:QUARTER",
        type="REPORT",
        business_date="2022-12-31",
        publication_date="2023-01-20",
        version_publication_date="2023-01-20",
        value="a" * 64,
    )
    entries.append(report)
    requirements[report["facts"]["key"]] = requirement(report)
    row = {
        "fund_code": "160323",
        "target": "2023-03-04",
        "mature_at": "2023-03-06T08:00:00+08:00",
        "input_keys": keys[:61] + [report["facts"]["key"]],
        "label_keys": keys[-2:],
    }
    return row, requirements, ready(entries)


def test_complete_row_passes_then_missing_latest_report_stops(tmp_path):
    row, req, index = synthetic_complete_row(tmp_path)
    assert row_at(row, "2023-04-01", req, index)["material_eligible"]
    del index[row["input_keys"][-1]]
    result = row_at(row, "2023-04-01", req, index)
    assert not result["material_eligible"] and len(result["failed_dependencies"]) == 1


def test_label_known_by_training_does_not_allow_late_input(tmp_path):
    row, req, index = synthetic_complete_row(tmp_path)
    key = row["input_keys"][0]
    index[key][0]["known_on"] = "2023-03-20"
    result = row_at(row, "2023-04-01", req, index)
    assert result["time_eligible"] and not result["material_eligible"]
    assert result["failed_dependencies"][0]["before"] == row["target"]


def test_maturity_at_cutoff_blocks_even_with_all_versions(tmp_path):
    row, req, index = synthetic_complete_row(tmp_path)
    row["mature_at"] = "2023-04-01T08:00:00+08:00"
    result = row_at(row, "2023-04-01", req, index)
    assert result["version_eligible"] and not result["material_eligible"]


def test_empty_or_incomplete_dependencies_do_not_pass(tmp_path):
    row, req, index = synthetic_complete_row(tmp_path)
    row["input_keys"] = []
    with pytest.raises(ValueError, match="EXPECTED_61_NAV"):
        row_at(row, "2023-04-01", req, index)


def test_sealed_nav_year_is_outside_source_scope(tmp_path):
    entry = evidence(tmp_path)
    entry["facts"]["business_date"] = "2025-01-02"
    with pytest.raises(ValueError, match="OUTSIDE_ALLOWED_SCOPE"):
        ready([entry])
