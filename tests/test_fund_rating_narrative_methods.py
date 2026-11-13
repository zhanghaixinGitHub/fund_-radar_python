"""说明必须对应原有类型和计分方法，不改分数，不把缺项写成正常。"""

import json
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest
from app.services.fund_rating_narrative import render
from app.services.fund_rating_narrative_methods import _legacy_components, _standard, _weighted, attach, tidy


def test_money_liquidity_uses_its_four_parts_instead_of_a_fixed_75():
    bundle = {"family": "MONEY", "reviewed_document": {"dimensions": {"liquidity": {"subcomponents": {
        "maturity_structure": {"candidate": "60"}, "security_transfer_restriction": {"candidate": "80"},
        "cash_liability_match": {"candidate": "20"}, "holder_concentration": {"candidate": "40"},
    }}}}}
    dimensions = {"liquidity": "51", "holdings": None}
    before = deepcopy((bundle, dimensions))
    facts = attach(bundle, dimensions, {}, root=Path("."))
    method = facts["score_methods"]["liquidity"]
    assert "60.00×40%" in method and "80.00×20%" in method
    assert "20.00×25%" in method and "40.00×15%" in method
    assert "得51.00分" in method and "固定给75分" not in method
    assert "不把未知部分补成0分" in facts["score_methods"]["holdings"]
    assert (bundle, dimensions) == before


def test_weighted_explanation_never_backsolves_a_missing_or_mismatched_part():
    terms = [("experience", 40), ("continuity", 20), ("governance", 40)]
    assert _weighted({"experience": "50", "continuity": "75"}, terms, Decimal(55)) is None
    assert _weighted({"experience": "50", "continuity": "75", "governance": "75"}, terms, Decimal(55)) is None


def test_legacy_reviewed_liquidity_is_not_assumed_to_be_fixed_75():
    facts = attach({"family": "ACTIVE_EQUITY"}, {"liquidity": "88.80"}, {}, root=Path("."))
    text = facts["score_methods"]["liquidity"]
    assert "40%、20%、25%、15%" in text and "固定给75分" not in text
    assert "不从总分倒推" in text


def test_index_scoring_anchor_is_distinguished_from_the_contract_target():
    methods = _standard(
        {"family": "INDEX", "evidence_v1_calculation": {"inputs": {}}},
        {"execution": {"tracking_error_limit": ".04"}, "performance": {"drawdown": ".2"}},
        {"efficiency": "75", "risk": "80"},
    )
    assert "固定的1%换分基准" in methods["efficiency"]
    assert "合同目标另行核对" in methods["efficiency"]
    assert "100－20.00＝80.00分" in methods["risk"]


def test_gold_matching_is_not_described_as_stock_concentration():
    methods = _standard(
        {"family": "INDEX", "typed_disclosure_calculation": {"profile": "GOLD"}}, {}, {"holdings": "90"}
    )
    assert "两层" in methods["holdings"] and "绝对偏差" in methods["holdings"]
    assert "100减去前十大股票" not in methods["holdings"]


def test_deduplication_preserves_sentences_with_different_local_referents():
    assessment = {
        "conclusion": "本项核对合同要求。",
        "paragraphs": ["本项核对合同要求。股票范围已核。", "现金下限5%。本期满足这一要求。",
                       "总资产上限140%。本期满足这一要求。", "总资产上限140%。本期满足这一要求。"],
        "limitations": [],
    }
    actual = tidy(assessment)
    assert actual["paragraphs"][0] == "股票范围已核。"
    assert len(actual["paragraphs"]) == 3
    assert all("本期满足这一要求" in p for p in actual["paragraphs"][1:])


def test_existing_fixed_score_explanation_is_not_appended_twice():
    actual = tidy({"conclusion": "核对申赎记录。", "paragraphs": ["资料核对通过时固定给75分。"], "limitations": []},
                  "本项目在符合条件时固定给75分。")
    assert len(actual["paragraphs"]) == 1


def test_bound_legacy_parts_need_the_same_product_and_score(monkeypatch):
    from app.services import material_content

    document = {"code": "000042", "dimension_candidate": "65", "experience": {"point": 75},
                "continuity": {"point": 75}, "governance": {"point": 50}}
    monkeypatch.setattr(material_content, "source_bytes", lambda _: json.dumps(document).encode())
    monkeypatch.setattr(material_content, "source_size", lambda _: len(json.dumps(document).encode()))
    bundle = {"fund_code": "000042", "reviewed_document": {"sources": {"management": "C:/old/000042-review.json"}}}
    assert "得65.00分" in _legacy_components(bundle, {"management": "65"}, {})["management"]
    assert _legacy_components(bundle, {"management": "55"}, {}) == {}
    document["code"] = "999999"
    with pytest.raises(ValueError, match="IDENTITY"):
        _legacy_components(bundle, {"management": "65"}, {})


def test_missing_score_keeps_its_range_and_does_not_gain_a_computed_point():
    item = {"status": "PARTIAL", "dimensions": [{"key": "holdings", "score": None,
            "scoreRange": {"lower": "20", "upper": "90"}, "explanation": "尚缺完整持仓。"}]}
    before = deepcopy(item)
    result = render(item, {"score_methods": {"holdings": "资料不足，不能计算确定分。"}}, "FOF")
    assert result["dimensions"][0]["score"] is None
    assert result["dimensions"][0]["scoreRange"] == before["dimensions"][0]["scoreRange"]
    assert "尚缺完整持仓" in result["dimensions"][0]["explanation"]
    assert item == before


def test_plain_language_keeps_dates_numbers_and_specific_missing_information():
    actual = tidy({
        "conclusion": "仍有资料需要核对。",
        "paragraphs": ["截至2026-06-30，CB/EB仍有1124万元未核，GICS分类未闭合。"],
        "limitations": ["尚缺全仓资料，继续补件。"],
    })
    assert actual["paragraphs"] == ["截至2026-06-30，可转债及可交换债仍有1124万元未核，行业分类尚未核对清楚。"]
    assert actual["limitations"] == ["尚缺全部持仓资料，继续补充核实。"]
