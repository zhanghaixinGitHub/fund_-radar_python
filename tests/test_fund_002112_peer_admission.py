"""验证新准入边界；全部合成/本地文件检查，无真实模型拟合。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from app.services import fund_002112_peer_admission as admission
from app.services.fund_002112_peer_admission import (
    check_current_nav,
    checked,
    ensure_zero_budget,
    exact_label,
    recipe_from_source,
)
from app.services.fund_002112_peer_fold_impact import FOLDS
from app.services.fund_002112_round3_data import COHORT
from app.services.fund_002112_zero_fit_review import file_hash

DAYS = ["2023-06-28", "2023-06-29", "2023-06-30"]


def nav(day, value, announced):
    return {"date": day, "nav": value, "ann_date": announced, "source_hash": "a" * 64}


@pytest.mark.parametrize("value,label", [("1.00000001", "UP"), ("1.00000000", "FLAT"), ("0.99999999", "DOWN")])
def test_exact_three_classes_without_tolerance(value, label):
    result = exact_label(nav(DAYS[0], "1", DAYS[1]), nav(DAYS[1], value, DAYS[2]), DAYS)
    assert result["actual_direction"] == label
    assert result["mature_at"] == "2023-07-01T08:00:00+08:00"


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "Infinity"])
def test_invalid_nav_rejected(value):
    with pytest.raises(ValueError, match="NON_POSITIVE"):
        exact_label(nav(DAYS[0], value, DAYS[1]), nav(DAYS[1], "1", DAYS[2]), DAYS)


@pytest.mark.parametrize("day", ["2025-01-02", "2026-01-05", "2020-12-31"])
def test_forbidden_new_label_dates(day):
    with pytest.raises(ValueError, match="OUTSIDE_SCOPE"):
        exact_label(nav("2020-12-30", "1", "2020-12-31"), nav(day, "2", day), DAYS)


def test_non_adjacent_label_rejected():
    with pytest.raises(ValueError, match="NOT_ADJACENT"):
        exact_label(nav(DAYS[0], "1", DAYS[1]), nav(DAYS[2], "2", "2023-07-01"), DAYS)


@pytest.mark.parametrize("announced", [None, "2025-01-01", "2023-06-27"])
def test_missing_or_impossible_publication(announced):
    with pytest.raises(ValueError, match="PUBLICATION"):
        exact_label(nav(DAYS[0], "1", DAYS[1]), nav(DAYS[1], "2", announced), DAYS)


def test_late_quarter_end_not_backdated():
    result = exact_label(nav(DAYS[1], "1", DAYS[2]), nav(DAYS[2], "2", "2023-07-20"), DAYS)
    assert result["mature_at"] == "2023-07-21T08:00:00+08:00"


def test_base_published_later_is_not_silently_admitted():
    result = exact_label(nav(DAYS[0], "1", "2023-07-20"), nav(DAYS[1], "2", DAYS[2]), DAYS)
    assert result["both_navs_public_by_label_date"] is False


def test_source_hash_required():
    row = nav(DAYS[1], "2", DAYS[2])
    row["source_hash"] = ""
    with pytest.raises(ValueError, match="SOURCE_HASH"):
        exact_label(nav(DAYS[0], "1", DAYS[1]), row, DAYS)


@pytest.mark.parametrize("field,value", [("unit_nav", "9"), ("ann_date", DAYS[2]), ("content_hash", "b" * 64)])
def test_database_difference_not_hidden(field, value):
    frozen = nav(DAYS[0], "1", DAYS[1])
    current = {"nav_date": DAYS[0], "unit_nav": "1.00000000", "ann_date": DAYS[1], "content_hash": "a" * 64}
    assert check_current_nav(frozen, current)
    current[field] = value
    assert not check_current_nav(frozen, current)


def test_database_unavailable_is_not_equal():
    assert not check_current_nav(nav(DAYS[0], "1", DAYS[1]), None)


@pytest.mark.parametrize("budget", [1, 6, 12, -1])
def test_zero_budget_cannot_be_opened(budget):
    protocol = {"current_fit_budget": budget, "proposed_fit_cap": 12, "cohort": list(COHORT), "folds": FOLDS}
    with pytest.raises(ValueError, match="BUDGET"):
        ensure_zero_budget(protocol)


@pytest.mark.parametrize("field", ["cohort", "folds"])
def test_scope_cannot_be_changed(field):
    protocol = {"current_fit_budget": 0, "proposed_fit_cap": 12, "cohort": list(COHORT), "folds": deepcopy(FOLDS)}
    ensure_zero_budget(protocol)
    protocol[field] = []
    with pytest.raises(ValueError, match="SCOPE"):
        ensure_zero_budget(protocol)


def test_source_tamper_stops_before_read(tmp_path):
    path = tmp_path / "source.json"
    path.write_text('{"x": 1}', encoding="utf-8")
    sha = file_hash(path)
    assert checked(path, sha) == {"x": 1}
    path.write_text('{"training_eligible": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="FROZEN_SOURCE_CHANGED"):
        checked(path, sha)


def test_static_recipe_does_not_execute_training(tmp_path):
    path = tmp_path / "recipe.py"
    path.write_text("raise AssertionError('must never execute')\nLOGISTIC = {'C': 1.0}\n", encoding="utf-8")
    assert recipe_from_source(path) == {"C": 1.0}


@pytest.mark.parametrize("created", ["D:20230330120000", "D:20260927120000"])
def test_pdf_metadata_and_dated_url_never_auto_admit(tmp_path, monkeypatch, created):
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"synthetic test only")
    report = {k: [] for k in admission.BUSINESS_FIELDS}
    report.update(
        fund_code="160323",
        fund_master_code="160323",
        report_end="2022-12-31",
        report_type="ANNUAL",
        title="华夏磐泰混合2022年年度报告",
        published_date="2023-03-30",
        source_publication_date="2023-03-30",
        holding_count=10,
        raw={"path": str(pdf), "url": "https://static.cninfo.com.cn/finalpage/2023-03-30/123.PDF"},
        source={"published_date": "2023-03-30"},
    )
    item = {
        "fund_code": "160323",
        "report_end": "2022-12-31",
        "report_type": "ANNUAL",
        "raw_sha256": "a" * 64,
        "parsed_file": "synthetic",
        "additional_dates_vs_original_per_fold": {"2023Q3": 15},
    }
    reader = SimpleNamespace(
        is_encrypted=False,
        pages=[SimpleNamespace(extract_text=lambda: "华夏磐泰 基金主代码 160323")],
        metadata={"/CreationDate": created, "/ModDate": created},
    )
    monkeypatch.setattr(admission, "PdfReader", lambda _: reader)
    monkeypatch.setattr(admission, "parse_text", lambda *a, **kw: report)
    result = admission.report_review(item, report, {})
    assert result["dated_primary_pdf_url"]
    assert result["training_eligible"] is False
    assert result["historical_version_verified"] is False
    assert result["system_download_in_history_required"] is False
    if created.startswith("D:2026"):
        assert "PDF_TIMESTAMP_MISSING_OR_AFTER_DECLARED_PUBLICATION" in result["gaps"]
