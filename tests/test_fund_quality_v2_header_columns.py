"""合成日期版式只用于拒收边界，不充当实际基金或校准样本。"""

from datetime import date

import pytest
from app.services.fund_quality_v2_header_columns import date_column_evidence


def frame(insert_stub=False):
    return "\n".join(["单位：人民币元", "资产 附注号".ljust(30) + "本期末".ljust(26) + "上年度末",
                      *( ["资 产 附注号"] if insert_stub else [] ),
                      "".ljust(30) + "2026年06月30".ljust(26) + "2025年12月",
                      "".ljust(30) + "日".ljust(26) + "31日"])


@pytest.mark.parametrize("insert_stub", [False, True])
def test_dates_are_recovered_from_independent_columns_not_line_concatenation(insert_stub):
    result = date_column_evidence(frame(insert_stub), date(2026, 6, 30))
    assert result["current_date"] == "2026-06-30"
    assert result["comparison_date"] == "2025-12-31"
    assert result["current_left_column"] == 30 and result["comparison_left_column"] == 56
    assert "2026年06月30" in result["original_header"] and "31日" in result["original_header"]


@pytest.mark.parametrize("before,after", [
    ("人民币元", "人民币万元"), ("2026年06月30", "2025年06月30"),
    ("2025年12月", "2027年12月"), ("31日", "31"), ("本期末", "比较期"),
    ("2025年12月", "2025年13月"),
])
def test_unverified_currency_dates_and_labels_have_no_fallback(before, after):
    assert date_column_evidence(frame().replace(before, after), date(2026, 6, 30)) is None


def test_missing_current_day_cannot_be_borrowed_from_comparison():
    text = frame().replace("日".ljust(26) + "31日", "".ljust(26) + "31日")
    assert date_column_evidence(text, date(2026, 6, 30)) is None


def test_repeated_asset_and_liability_headers_must_all_agree():
    result = date_column_evidence(frame(True) + "\n" + frame(True), date(2026, 6, 30))
    assert result["concordant_header_count"] == 2 and len(result["repeated_headers"]) == 1
    wrong_current = frame(True) + "\n" + frame(True).replace("2026年06月30", "2024年06月30")
    wrong_prior = frame(True) + "\n" + frame(True).replace("2025年12月", "2024年12月")
    assert date_column_evidence(wrong_current, date(2026, 6, 30)) is None
    assert date_column_evidence(wrong_prior, date(2026, 6, 30)) is None


def test_valid_header_cannot_hide_broken_later_header_or_conflicting_units():
    broken = frame() + "\n" + frame().replace("31日", "31")
    unit = frame() + "\n" + frame().replace("人民币元", "美元")
    assert date_column_evidence(broken, date(2026, 6, 30)) is None
    assert date_column_evidence(unit, date(2026, 6, 30)) is None


def test_chinese_stub_width_is_not_treated_as_the_same_number_of_ascii_columns():
    text = "\n".join(["单位：人民币元",
                      "".ljust(38) + "本期末" + " " * 8 + "上年度末",
                      "负债和净资产".ljust(30) + "2024 年 12 月  " + "2023 年 12 月 31",
                      "".ljust(40) + "31 日" + " " * 12 + "日"])
    result = date_column_evidence(text, date(2024, 12, 31))
    assert result["column_measure"] == "EAST_ASIAN_DISPLAY_WIDTH"
    assert result["current_date"] == "2024-12-31" and result["comparison_date"] == "2023-12-31"
    for fragment in result["fragments"]:
        line = fragment["original_line"]
        a, b = fragment["current_char_span"]
        c, d = fragment["comparison_char_span"]
        assert line[a:b] == fragment["current_fragment"] and line[c:d] == fragment["comparison_fragment"]
