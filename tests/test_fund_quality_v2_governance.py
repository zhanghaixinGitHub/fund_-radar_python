"""治理原文边界反例；以下测试文本不进入校准或独立验证产品集。"""

from datetime import date

import pytest
from app.services.fund_quality_v2_governance import (
    declaration_coverage,
    report_governance_sections,
    report_manager_name,
    safe_decision,
)


@pytest.mark.parametrize("counter", ["第 66 页 共 71 页", "66"])
def test_adverse_report_statement_across_pages_is_not_clean_governance_or_regulator_acceptance(counter):
    pages = [
        "基金管理人：测试基金管理有限公司\n11.6.1管理人受调查或处罚等情况\n"
        "受到调查或处罚等措施的主体 管理人\n受到调查或处罚等措施的时间 2025 年 10 月 27 日\n"
        "采取调查或处罚等措施的机构 测试监管机关\n受到调查或处罚等措施类型 行政监管措施\n"
        "受到的具体措施类型 责令改正\n受到调查或处罚等措施的原因 合规内控、其他问题\n"
        "受到处罚的依据 测试规定\n",
        "测试混合基金2025年年度报告\n" + counter + "\n"
        "管理人采取整改措施的情况 管理人声明已整改并通过验收。\n其他 -\n"
        "11.6.2 管理人相关从业人员受调查或处罚等情况\n本期无相关情形。\n"
        "11.6.3 托管人受调查或处罚等情况\n托管人内容不属于管理人章节。",
    ]
    items = report_governance_sections(
        pages, coverage=(date(2025, 1, 1), date(2025, 12, 31)), source_hash="test-only", source_url="TEST_ONLY"
    )
    assert len(items) == 2 and items[0]["pages"] == [1, 2] and items[1]["pages"] == [2]
    event = items[0]["reported_events"][0]
    assert event["event_dates"] == ["2025-10-27"]
    assert event["reported_reasons"] == "合规内控、其他问题"
    assert event["reported_legal_basis"] == "测试规定"
    assert event["reported_remediation"] == "管理人声明已整改并通过验收。"
    assert counter in items[0]["excerpt"]
    assert event["remediation_evidence_kind"] == "ISSUER_REPORT_STATEMENT_NOT_REGULATOR_ACCEPTANCE_ORIGINAL"
    assert "托管人内容" not in items[1]["excerpt"]
    assert report_manager_name(pages) == "测试基金管理有限公司"
    assert all("governance_state" not in item for item in items)


def test_missing_report_section_is_not_a_no_event_statement():
    assert (
        report_governance_sections(
            ["本报告期未发现其他重大事项。"],
            coverage=(date(2025, 1, 1), date(2025, 12, 31)),
            source_hash="test",
            source_url="TEST_ONLY",
        )
        == []
    )


def test_verified_bare_page_number_is_removed_from_field_but_raw_quote_kept():
    pages = [
        "1\n测试基金2025年年度报告\n基金管理人：测试基金管理有限公司",
        "2\n11.6.1 管理人受调查或处罚等情况\n受到调查或处罚等措施的主体 管理人\n"
        "受到调查或处罚等措施的时间 2025年11月11日\n受到的具体措施类型 责令改正",
        "3\n受到调查或处罚等措施的原因 合规内控\n受到处罚的依据 《测试规定》\n"
        "管理人采取整改措施的情况 公司自述完成整改。\n11.6.2 管理人相关从业人员受调查或处罚等情况\n"
        "本期无。\n11.6.3 托管人受调查或处罚等情况\n本期无。",
    ]
    values = report_governance_sections(pages, coverage=(date(2025, 1, 1), date(2025, 12, 31)),
                                         source_hash="TEST_ONLY", source_url="TEST_ONLY")
    event = values[0]['reported_events'][0]
    assert event['reported_measures'] == '责令改正'
    assert '\n3\n' in values[0]['excerpt']
    assert any(row['page'] == 3 and row['removed_header'] == '3' for row in values[0]['parser_header_edits'])


def test_interleaved_reprint_reasons_and_split_remediation_label_are_not_clean_fields():
    text = (
        "11.6.1 管理人受调查或处罚等情况\n受到调查或处罚等措施的主体 管理人\n"
        "受到调查或处罚等措施的时间 2025年11月11日\n受到调查或处罚等措施的原因 合规内控《测试规定前半\n"
        "受到处罚的依据 后半条款》\n管理人采取整改措施的情况（如提出整改意公司自述见）完成。\n"
        "11.6.2 管理人相关从业人员受调查或处罚等情况\n本期无。\n"
        "11.6.3 托管人受调查或处罚等情况\n本期无。"
    )
    values = report_governance_sections([text], coverage=(date(2025, 1, 1), date(2025, 12, 31)),
                                         source_hash="TEST_ONLY", source_url="TEST_ONLY")
    event = values[0]['reported_events'][0]
    assert event['field_state'] == 'LAYOUT_AMBIGUOUS_RAW_TEXT_ONLY'
    assert len(event['layout_warnings']) == 2
    assert event['reported_remediation'] == '（如提出整改意公司自述见）完成。'


def test_declaration_coverage_unions_overlaps_and_keeps_missing_quarter():
    item = {"scope": "MANAGER", "coverage": ["2026-01-01", "2026-06-30"], "source_hashes": ["test-only"]}
    result = declaration_coverage([item, item], start=date(2026, 1, 1), end=date(2026, 9, 30))
    assert result["by_report_scope"]["MANAGER"]["covered_days"] == 181
    assert result["by_report_scope"]["MANAGER"]["missing_intervals"] == [["2026-07-01", "2026-09-30"]]
    assert result["scope"] == "REPORT_DECLARATION_COVERAGE_NOT_GOVERNANCE_STATE"
    item["coverage"][1] = "2026-12-31"
    with pytest.raises(ValueError):
        declaration_coverage([item], start=date(2026, 1, 1), end=date(2026, 9, 30))


def test_old_combined_title_covers_company_and_senior_staff_but_not_all_staff():
    records = [
        {'scope': 'COMBINED_MANAGER_CUSTODIAN_OR_STAFF', 'coverage': ['2024-01-01', '2024-12-31'],
         'source_hashes': ['TEST_2024']},
        *[{'scope': scope, 'coverage': ['2025-01-01', '2026-06-30'], 'source_hashes': ['TEST_LATER']}
          for scope in ['MANAGER', 'MANAGER_RELATED_STAFF']],
    ]
    result = declaration_coverage(records+records, start=date(2023, 9, 30), end=date(2026, 9, 30))
    assert result['by_report_scope']['MANAGER']['covered_days'] == 546
    subjects = result['by_subject_scope']
    assert subjects['MANAGER_COMPANY']['covered_days'] == 912
    assert subjects['MANAGER_SENIOR_MANAGEMENT']['covered_days'] == 912
    assert subjects['MANAGER_ALL_RELATED_STAFF']['covered_days'] == 546
    assert subjects['MANAGER_COMPANY']['missing_intervals'] == [
        ['2023-09-30', '2023-12-31'], ['2026-07-01', '2026-09-30']]


def test_report_outside_window_adds_neither_coverage_nor_source():
    outside = {'scope': 'MANAGER', 'coverage': ['2022-01-01', '2022-12-31'], 'source_hashes': ['TEST_OLD']}
    current = {'scope': 'MANAGER', 'coverage': ['2026-01-01', '2026-06-30'], 'source_hashes': ['TEST_NEW']}
    result = declaration_coverage([outside, current], start=date(2023, 9, 30), end=date(2026, 9, 30))
    assert result['by_subject_scope']['MANAGER_COMPANY']['source_hashes'] == ['TEST_NEW']


def safe_html(entity="测试基金管理有限公司"):
    fields = {
        "行政处罚决定书文号": "测试〔2025〕29号",
        "违规主体名称": entity,
        "作出处罚决定的行政机关名称": "测试外汇监管机关",
        "违法事实": "违反外汇登记管理规定",
        "处罚类别": "警告、罚款",
        "罚没款金额（万元）": "22.50",
        "处罚决定日期": "2025-11-18",
    }
    return (
        "<p>更新时间2020-01-15</p><table>"
        + "".join("<tr><th>" + key + "</th><td>" + value + "</td></tr>" for key, value in fields.items())
        + "</table>"
    ).encode()


def test_regulatory_date_and_money_come_from_named_fields_not_page_update_or_success_flag():
    result = safe_decision(safe_html())
    assert result["event_date"] == "2025-11-18"
    assert result["fine_amount_cny"] == "225000.00"
    assert result["legal_entity"] == "测试基金管理有限公司"
    assert result["scope"] == "REGULATOR_DECISION_NOT_REMEDIATION_OR_SEVERITY_REVIEW"
    with pytest.raises(ValueError):
        safe_decision(b"<html>success:true</html>")


@pytest.mark.parametrize("entity", ["测试基金管理有限公司", "另一家基金管理有限公司"])
def test_regulatory_subject_binds_to_manager_not_held_security_issuer(tmp_path, monkeypatch, entity):
    from app.services import fund_quality_v2_disclosures as module

    body = (
        "测试混合基金2026年中期报告 基金主代码123456 基金管理人：测试基金管理有限公司\n"
        "本报告期自2026年1月1日至2026年6月30日\n10.6.1 管理人受调查或处罚等情况\n本期无。\n"
        "10.6.2 管理人相关从业人员受调查或处罚等情况\n本期无。\n10.6.3 托管人受调查或处罚等情况\n无。"
    )
    doc = {"pages": [body], "text": body, "ref": {"sha256": "test-report", "url": "TEST_ONLY"}}
    monkeypatch.setattr(module, "source_documents", lambda *a: {"half": doc})
    (tmp_path / "decision.html").write_bytes(safe_html(entity))
    refs = [
        {
            "label": "governance-regulator",
            "source_kind": "SAFE_PUBLIC_DECISION",
            "url": "https://www.safe.gov.cn/www/illegal/listQuery?saffno=test-only",
            "file": "decision.html",
            "sha256": "test-regulator",
            "acquired_at": "2026-10-17T00:00:00+08:00",
        }
    ]
    result = module.extract_disclosures("123456", refs, tmp_path, date(2026, 9, 30))
    assert "governance_state" not in result["metrics"]
    assert len(result["facts"]["official_governance_decisions"]) == (1 if entity == "测试基金管理有限公司" else 0)
    if entity != "测试基金管理有限公司":
        assert (
            result["facts"]["governance_source_scope_gaps"][0]["reason"]
            == "REGULATOR_SUBJECT_NOT_BOUND_TO_CURRENT_REPORT_MANAGER"
        )
    refs[0]["url"] = "https://example.com/www/illegal/listQuery"
    with pytest.raises(ValueError, match="UNTRUSTED_REGULATOR"):
        module.extract_disclosures("123456", refs, tmp_path, date(2026, 9, 30))


def test_future_report_does_not_leave_a_future_governance_coverage_after_holder_failure(tmp_path, monkeypatch):
    from app.services import fund_quality_v2_disclosures as module

    body = (
        "测试股票基金2027年中期报告 基金主代码123456 本报告期自2027年1月1日至2027年6月30日\n"
        "10.6.1管理人受调查或处罚等情况\n未来声明\n10.6.2管理人相关从业人员受调查或处罚等情况\n"
        "未来声明\n10.6.3托管人受调查或处罚等情况"
    )
    monkeypatch.setattr(
        module,
        "source_documents",
        lambda *a: {"half": {"pages": [body], "text": body, "ref": {"sha256": "test-report", "url": "TEST_ONLY"}}},
    )
    result = module.extract_disclosures("123456", [], tmp_path, date(2026, 9, 30))
    assert result["facts"]["governance_report_declarations"] == []
    assert result["findings"]["holder_pressure_band"]["reason"] == "REPORT_PERIOD_FUTURE_OR_EXPIRED"
