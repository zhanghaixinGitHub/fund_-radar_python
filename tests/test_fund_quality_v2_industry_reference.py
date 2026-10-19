"""行业表解析和缺失边界；临时测试文本不计入真实校准或来源证明。"""

import pytest
from app.services.fund_quality_v2_industry_reference import equity_industry_diagnostic, parse_capco_security_pages


def sample():
    return [
        "2026年上半年上市公司行业分类结果\n上市公司代码 门类代码\n"
        "000001 示例银行 J 金融业 66 货币金融服务\n"
        "600001示例制造 C 制造业 CG 专用设备 35 专用设备制造业\n"
        "600002示例电力 D\n电力、热力、燃气及水生产和供应业\n44 电力生产和供应业"
    ]


def test_industry_uses_own_primary_column_and_preserves_wrapped_row_page():
    result = parse_capco_security_pages(sample(), period_end="2026-06-30")
    assert result["security_count"] == 3
    assert result["securities"]["600001"]["industry_code"] == "C"
    assert result["securities"]["600002"]["industry_code"] == "D"
    assert result["securities"]["600002"]["page"] == 1
    assert "\n44" in result["securities"]["600002"]["excerpt"]


@pytest.mark.parametrize("problem", ["period", "duplicate", "missing", "empty_page"])
def test_industry_table_rejects_wrong_period_duplicate_or_incomplete_rows(problem):
    pages = sample()
    if problem == "period":
        pages[0] = pages[0].replace("2026年", "2025年")
    elif problem == "duplicate":
        pages[0] += "\n000001另一个名字 C 制造业 35 专用设备制造业"
    elif problem == "missing":
        pages[0] += "\n600003缺行业代码及大类"
    else:
        pages.append("无证券行")
    with pytest.raises(ValueError):
        parse_capco_security_pages(pages, period_end="2026-06-30")


def test_industry_slice_requires_every_security_and_same_observation_date():
    classification = parse_capco_security_pages(sample(), period_end="2026-06-30")
    classification.update(published_on="2026-09-30", source_hash="test-only")
    reference = {"observed_on": "2026-06-30", "weights": {"000001.SZ": "1"}}
    positions = [{"stock_code": "000001.SZ", "market_value_cny": "1000"}]
    result = equity_industry_diagnostic(positions, reference, classification, expected_stock_amount="1000")
    assert result["half_l1_distance"] == "0"
    assert result["classification_published_on"] == "2026-09-30"
    assert result["status"] == "EQUITY_INDUSTRY_SLICE_DIAGNOSTIC_NOT_V2_METRIC"
    positions[0]["stock_code"] = "999999.SH"
    with pytest.raises(ValueError, match="UNMAPPED"):
        equity_industry_diagnostic(positions, reference, classification, expected_stock_amount="1000")
    # 未上市持仓无法查证券分类时，只有原件完整行业表可提供整体金额；仍不补该证券映射。
    industry_rows = [{"industry_code": "C", "value_cny": "600"}, {"industry_code": "C", "value_cny": "400"}]
    result = equity_industry_diagnostic(
        positions, reference, classification, expected_stock_amount="1000", disclosed_industries=industry_rows
    )
    assert result["actual_industries"] == {"C": "1"}
    assert result["unmapped_actual_securities"] == ["999999.SH"]
    assert result["actual_industry_basis"] == "ORIGINAL_REPORT_COMPLETE_INDUSTRY_AMOUNTS"
    industry_rows.pop()
    with pytest.raises(ValueError, match="AMOUNTS_INCOMPLETE"):
        equity_industry_diagnostic(
            positions, reference, classification, expected_stock_amount="1000", disclosed_industries=industry_rows
        )
    positions[0]["stock_code"] = "000001.SZ"
    reference["observed_on"] = "2026-05-31"
    with pytest.raises(ValueError, match="DATE_MISMATCH"):
        equity_industry_diagnostic(positions, reference, classification, expected_stock_amount="1000")
