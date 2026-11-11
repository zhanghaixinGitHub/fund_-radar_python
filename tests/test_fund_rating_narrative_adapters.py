"""全基金适配不得混淆类型、币种参照、窗口或把缺项补为完整评分。"""

from copy import deepcopy

import pytest
from app.services.fund_rating_narrative import render
from app.services.fund_rating_narrative_adapters import base_facts, business_text, money_facts, r5_facts


def bundle(family="REIT"):
    return {
        "fund_code": "000099",
        "family": family,
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "share_class": "A",
        "public_explanations": {"liquidity": "封闭式基金没有日常申赎，退出依赖二级市场交易。"},
        "reviewed_document": {
            "dimensions": {
                "liquidity": {"candidate": None, "reason": "项目现金有融资限制，不能按开放基金现金推断可付款金额。"},
                "management": "55",
            }
        },
    }


def detail():
    return {
        "status": "PARTIAL",
        "score": None,
        "grade": None,
        "summary": "部分评分",
        "dimensions": [
            {
                "key": "liquidity",
                "score": None,
                "scoreRange": {"lower": "0", "upper": "100"},
                "explanation": "旧说明",
                "metrics": [],
                "sources": [],
            },
            {"key": "management", "score": "55", "explanation": "已核实际职责。", "metrics": [], "sources": []},
        ],
    }


def test_reit_partial_preserves_actual_exit_restrictions_and_never_gets_open_fund_assurance():
    raw = bundle()
    before = deepcopy(raw)
    result = render(detail(), base_facts(raw), "REIT")
    text = result["dimensions"][0]["explanation"]
    assert "二级市场交易" in text and "融资限制" in text and "尚无确定分" in text
    assert result["score"] is None and result["status"] == "PARTIAL"
    assert result["dimensions"][0]["scoreRange"] == {"lower": "0", "upper": "100"}
    assert raw == before


def test_business_text_keeps_qualification_but_does_not_expose_old_paths_or_internal_identifiers():
    text = "合格现金下限5%仍无法核准，不能判定违规。来源C:\\old\\facts.json。固定权重不重分配。"
    actual = business_text(text)
    assert "下限5%" in actual and "不能判定违规" in actual
    assert "C:" not in actual and "固定权重" not in actual


def test_foreign_reference_is_not_renamed_contract_benchmark():
    raw = detail()
    raw["status"] = "SCORED"
    raw["dimensions"] = [{"key": "return", "score": "88", "explanation": "", "metrics": [], "sources": []}]
    evidence = {
        "performance": {
            "fund_return": "-.10",
            "reference_return": "-.16",
            "reference_label": "同地域人民币参照",
            "period": "同一年",
        }
    }
    result = render(raw, evidence, "QDII_EQUITY")
    text = result["dimensions"][0]["explanation"]
    assert "同地域人民币参照" in text and "比参照少跌" in text and "合同基准" not in text


@pytest.mark.parametrize(
    ("unit", "target", "expected"),
    [("PERCENT_RANKS_RATIO_WINDOWS", "12", "0.12"), ("RATIO_RANKS_RATIO_WINDOWS", ".12", ".12")],
)
def test_old_rank_units_are_explicit_and_extra_drawdown_is_not_absolute_drawdown(unit, target, expected):
    raw = bundle("ACTIVE_EQUITY")
    raw["r5_calculation"] = {
        "packet": {
            "unit_contract": unit,
            "quality_metrics": {
                "observations": {
                    "return_1y": {"kind": "rank", "target": target, "peers": []},
                    "extra_drawdown": {"kind": "rank", "target": target, "peers": []},
                }
            },
        }
    }
    result = r5_facts(raw, facts=base_facts(raw))
    assert result["performance"]["fund_return"] == expected
    assert "drawdown" not in result["performance"]
    assert result["extra_metrics"]["risk"][0]["name"] == "相对基准额外回撤观测"


def test_money_uses_disclosed_maturity_and_cash_facts_without_nav_return_invention():
    raw = bundle("MONEY")
    raw["money_inputs"] = {"liquidity": {"assets_30day_ratio": ".23", "ordinary_cash_principal": "120000"}}
    result = money_facts(raw, facts=base_facts(raw))
    assert "performance" not in result
    assert [m["value"] for m in result["extra_metrics"]["liquidity"]] == ["23.00", "12.00"]


def test_gold_economic_exposure_and_residual_budget_are_not_stock_matching_or_free_cash(monkeypatch):
    from app.services import fund_rating_typed_disclosure as typed
    from app.services.fund_rating_narrative_adapters import typed_facts

    raw = bundle("INDEX")
    raw["typed_disclosure_calculation"] = {"profile": "GOLD"}
    monkeypatch.setattr(
        typed,
        "calculate",
        lambda *a, **kw: {
            "dimensions": {"holdings": "90"},
            "blocked": False,
            "facts": {
                "holdings": {
                    "economic_gold_exposure_fraction": ".9983",
                    "net_non_gold_residual_budget_fraction": ".0017",
                    "target_holding_nav_fraction": ".9285",
                }
            },
        },
    )
    facts = typed_facts(raw, {"holdings": "90"}, root=None, facts=base_facts(raw))
    metrics = facts["extra_metrics"]["holdings"]
    assert [m["value"] for m in metrics] == ["92.85", "99.83", "0.17"]
    assert "非黄金残余预算" == metrics[2]["name"] and "现金" not in metrics[2]["name"]


def test_special_type_replay_cannot_supply_metrics_from_mismatched_score(monkeypatch):
    from app.services import fund_rating_typed_disclosure as typed
    from app.services.fund_rating_narrative_adapters import typed_facts

    raw = bundle("CONVERTIBLE")
    raw["typed_disclosure_calculation"] = {"profile": "CONVERTIBLE"}
    monkeypatch.setattr(typed, "calculate", lambda *a, **kw: {"dimensions": {"risk": "88"}, "blocked": False})
    with pytest.raises(ValueError, match="SCORE_MISMATCH"):
        typed_facts(raw, {"risk": "77"}, root=None, facts=base_facts(raw))


def test_new_uncompiled_formal_fund_uses_published_notes_without_compiling(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from app.services import fund_rating_narrative_facts as compiler
    from app.services.fund_rating_narrative_store import enrich

    raw = bundle()
    snapshot = {"members": [{"fund_code": "000099", "bundle": raw}]}
    result = SimpleNamespace(fund_code="000099", batch_id="b" * 64)
    monkeypatch.setattr(compiler, "compile_facts", lambda *a, **kw: pytest.fail("GET compiled"))
    actual = enrich(detail(), result, snapshot, root=tmp_path)
    assert all(d["assessment"] for d in actual["dimensions"])
    assert "二级市场" in actual["dimensions"][0]["explanation"]
