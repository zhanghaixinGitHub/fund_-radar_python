"""报告章节与覆盖统计检查，不训练模型，不按预测表现给报告分类。"""

import pytest
from app.services.fund_002112_style_coverage import describe_rows, strategy_section


@pytest.mark.parametrize(
    "heading",
    [
        "4.4报告期内基金的投资策略和运作分析",
        "4.4.1报告期内基金投资策略和运作分析",
        "4．4报告期内基金的投资策略及运作分析",
    ],
)
def test_exact_strategy_heading(heading):
    paragraph = "本基金集中配置医药行业，并持续研究组合公司的基本面变化。" * 5
    result = strategy_section([{"page": 6, "text": heading + paragraph + "4.5报告期内基金的业绩表现"}])
    assert result["status"] == "EXTRACTED" and result["text"] == paragraph and result["pages"] == [6]


def test_preserves_page_span_and_excludes_results():
    body = "本基金配置较均衡，在不同行业中寻找投资机会。" * 5
    result = strategy_section(
        [
            {"page": 10, "text": "4.4.1报告期内基金投资策略和运作分析" + body[:50]},
            {"page": 11, "text": body[50:] + "4.4.2报告期内基金的业绩表现收益率20%"},
        ]
    )
    assert result["text"] == body and result["pages"] == [10, 11] and "20%" not in result["text"]


@pytest.mark.parametrize(
    "text", ["只有市场评论医药上涨", "4.4报告期内基金的投资策略和运作分析很短4.5报告期内基金的业绩表现"]
)
def test_missing_or_toc_is_not_evidence(text):
    assert strategy_section([{"page": 1, "text": text}])["status"] == "SECTION_NOT_UNAMBIGUOUS"


def test_duplicate_body_is_not_silently_selected():
    text = "4.4报告期内基金的投资策略和运作分析" + "本基金配置较均衡。" * 12 + "4.5报告期内基金的业绩表现"
    assert strategy_section([{"page": 1, "text": text + text}])["status"] == "SECTION_NOT_UNAMBIGUOUS"


def test_long_ascii_toc_does_not_hide_real_body():
    heading = "4.4报告期内基金的投资策略和运作分析"
    ending = "4.5报告期内基金的业绩表现"
    body = "本基金均衡配置各类行业，持续跟踪公司基本面变化。" * 5
    result = strategy_section([{"page": None, "text": heading + "." * 90 + "11" + ending + heading + body + ending}])
    assert result["status"] == "EXTRACTED" and result["text"] == body


def test_counts_unknown_and_separates_own_from_peers():
    rows = [
        {"report_sha256": "a", "fund_code": "002112", "target": "2023-06-01", "actual_direction": "UP"},
        {"report_sha256": "b", "fund_code": "004237", "target": "2023-06-01", "actual_direction": "DOWN"},
        {"report_sha256": "c", "fund_code": "006038", "target": "2023-06-02", "actual_direction": "FLAT"},
    ]
    tags = {"a": {"tag": "MEDICINE_FOCUS"}, "b": {"tag": "MEDICINE_FOCUS"}, "c": {"tag": "UNSPECIFIED"}}
    result = describe_rows(rows, tags)
    assert result["MEDICINE_FOCUS"]["rows"] == 2
    assert result["MEDICINE_FOCUS"]["unique_dates"] == 1
    assert result["MEDICINE_FOCUS"]["target_own_rows"] == result["MEDICINE_FOCUS"]["other_fund_rows"] == 1
    assert result["UNSPECIFIED"]["classes"] == {"FLAT": 1}
    assert result["AI_COMPUTE_FOCUS"]["first_target"] is None


def test_missing_classification_fails_instead_of_filling():
    with pytest.raises(KeyError):
        describe_rows([{"report_sha256": "missing"}], {})
