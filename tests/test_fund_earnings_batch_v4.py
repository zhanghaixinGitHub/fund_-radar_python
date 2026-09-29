"""覆盖下载前用途分类、真实更正分类和版本标题匹配，保持日期及主体保护。"""

import pytest
from app.services.fund_earnings_batch_v4 import EarningsBatch, catalog_admissions, issuer_identity
from app.services.fund_earnings_titles_v3 import classify_catalog_row
from app.services.fund_information_history_v1 import save


@pytest.mark.parametrize(
    "stock,title,reason",
    [
        ("002594", "关于举行2021年年度报告网上说明会的通知", "NON_EARNINGS_MEETING_OR_CORPORATE_RETURN"),
        ("600036", "[H股公告]招商银行股份有限公司2021年度企业年度报告书", "NON_EARNINGS_MEETING_OR_CORPORATE_RETURN"),
        (
            "601318",
            "中国平安关于披露平安银行2021年第三季度报告的公告",
            "DISCLOSURE_FORWARDING_NOTICE_NOT_FINANCIAL_REPORT",
        ),
        ("601318", "中国平安：平安银行股份有限公司2021年度业绩快报", "VERIFIED_DIFFERENT_FINANCIAL_REPORT_SUBJECT"),
    ],
)
def test_verified_non_company_financial_documents_excluded(stock, title, reason):
    result = classify_catalog_row({"secCode": stock, "title_plain": title})
    assert result["kind"] is None and result["reason"] == reason


@pytest.mark.parametrize(
    "title",
    [
        "关于2021年半年度报告更正的公告",
        "2021年第三季度报告更正公告",
        "2021年度业绩预告修正公告",
    ],
)
def test_correction_is_separate_from_reported_or_forecast_value(title):
    assert classify_catalog_row({"secCode": "002460", "title_plain": title})["kind"] == "CORRECTION_NOTICE"


@pytest.mark.parametrize(
    "stock,title,kind",
    [
        ("000001", "平安银行股份有限公司2021年度业绩快报", "PRELIMINARY_RESULT"),
        ("002594", "关于披露2021年度业绩快报的公告", "PRELIMINARY_RESULT"),
        ("601318", "中国平安2021年第三季度报告", "REPORTED_RESULT"),
        ("002812", "2021年第三季度报告（更新后）", "REPORTED_RESULT"),
        ("600426", "2022年一季度业绩预增公告", "FORECAST"),
    ],
)
def test_actual_disclosures_are_kept(stock, title, kind):
    assert classify_catalog_row({"secCode": stock, "title_plain": title})["kind"] == kind


@pytest.mark.parametrize(
    "stock,title,body",
    [
        ("002812", "2021年第三季度报告（更新后）", "证券代码：002812 2021年第三季度报告"),
        ("603501", "2021年前三季度业绩预增公告", "证券代码：603501 2021年前三季度业绩预增的公告"),
        ("601636", "601636_2021年_年度报告", "证券代码：601636 2021年年度报告"),
        ("603599", "2022年一季度报告.docx", "证券代码：603599 2022年第一季度报告"),
    ],
)
def test_format_aliases_keep_original_title_and_public_date(stock, title, body):
    row = {"secCode": stock, "title_plain": title, "published_date": "2021-12-31"}
    result = issuer_identity(row, [body])
    assert result["passed"] and not result["semantic_verified"]
    assert result["original_catalog_title"] == title and row["published_date"] == "2021-12-31"


def test_title_aliases_do_not_override_explicit_different_issuer():
    row = {"secCode": "601318", "title_plain": "2021年第三季度报告（更新后）"}
    result = issuer_identity(row, ["股票代码：000001 2021年第三季度报告 母公司601318"])
    assert not result["passed"]


def test_new_driver_resume_preserves_existing_frozen_predecessor(tmp_path, monkeypatch):
    batch = EarningsBatch("20260929-earnings-v5")
    batch.out = tmp_path
    save(tmp_path / "plan.json", {})
    monkeypatch.setattr(batch, "check_plan", lambda: {"previous_run": "C:/data/20260928-earnings-v4", "windows": [1]})
    assert batch.prepare("20260928-earnings-v4")["reused_frozen_plan"]
    with pytest.raises(ValueError, match="PREDECESSOR_CHANGED"):
        batch.prepare("20260928-earnings-v3")


def test_catalog_metadata_conflict_cannot_be_hidden_by_exclusion():
    row = {
        "announcementId": "123",
        "secCode": "002594",
        "adjunctUrl": "a.pdf",
        "published_date": "2021-12-31",
        "title_plain": "2021年度报告",
    }
    changed = {**row, "title_plain": "2021年度报告说明会"}
    with pytest.raises(ValueError, match="METADATA_CONFLICT"):
        catalog_admissions([{"catalog_complete": True, "rows": [row, changed]}])


def test_issuer_code_before_numeric_date_is_not_merged():
    row = {"secCode": "601318", "title_plain": "2021年年度报告"}
    proof = issuer_identity(row, ["股票代码：000001\n2021年年度报告 母公司601318"])
    assert proof["explicit_body_codes"] == ["000001"] and not proof["passed"]
