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
    dimension = result["dimensions"][0]
    text = " ".join([dimension["explanation"], *(metric["name"] for metric in dimension["metrics"])])
    assert "同地域人民币参照" in text and "比参照少跌" in text and "合同基准" not in text
    assert dimension["metrics"][2]["name"] == "相对参照收益差"


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


@pytest.mark.parametrize(
    ("cash_state", "upper", "score", "expected"),
    [
        (False, ".0305", "45", "即使按最宽松"),
        (None, ".06", None, "还不能确认"),
        (True, ".06", "75", "本期满足这一要求"),
    ],
)
def test_mixed_execution_explains_actual_requirements_without_turning_weights_into_limits(
    monkeypatch, cash_state, upper, score, expected
):
    from app.services import fund_rating_typed_disclosure as typed
    from app.services.fund_rating_narrative_adapters import typed_facts

    raw = bundle("MIXED")
    raw["typed_disclosure_calculation"] = {
        "profile": "MIXED",
        "inputs": {"allocation_contract": {
            "stock_min": "0", "stock_max": ".30", "cash_min": ".05", "leverage_max": "1.4",
            "claim": {"quote": "开放期内和封闭期内采用不同要求。"},
        }},
    }
    execution = {
        "core_constraints": {"stock": True, "cash": cash_state, "leverage": True},
        "qualified_cash_upper_nav_ratio": upper,
    }
    monkeypatch.setattr(typed, "calculate", lambda *a, **kw: {
        "dimensions": {"efficiency": score}, "blocked": False, "facts": {"execution": execution},
    })
    facts = typed_facts(raw, {"efficiency": score}, root=None, facts=base_facts(raw))
    original = {"dimensions": [{"key": "efficiency", "score": score, "metrics": [], "explanation": "旧说明"}]}
    actual = render(original, facts, "MIXED")["dimensions"][0]
    assessment = actual["assessment"]
    text = actual["explanation"]
    assert actual["score"] == score
    assert "0.00%至30.00%" in text and "5.00%" in text and "140.00%" in text
    assert expected in assessment["paragraphs"][1]
    assert "开放期" in text and "封闭期" in text and "不能直接据此认定违法违规" in text
    assert "40%" not in text and "满足项75" not in text and "执行45分" not in text
    if cash_state is False:
        assert "现金比例未达到" in assessment["conclusion"]
        assert actual["metrics"][0]["name"] == "合同认可现金占净资产（最高估计）"
        assert actual["metrics"][0]["value"] == "3.05"
    if cash_state is None:
        assert "现金比例仍缺少确认依据" in assessment["conclusion"]


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


def test_remediation_management_separates_company_disclosure_from_regulator_confirmation():
    from app.services.fund_rating_narrative_adapters import _remediation_management_assessment

    inputs = {
        "holding_date": "2026-06-30",
        "manager_name": "示例基金管理有限公司",
        "manager_review": {
            "governance_state": "REMEDIATION_DISCLOSED",
            "current_names": ["张某", "李某"],
            "claims": [
                {"quote": "本基金经理张某、李某。"},
                {"quote": "受到调查或处罚等措施的时间2025年11月4日"
                 "采取调查或处罚等措施的机构某监管局受到调查或处罚等措施类型行政监管措施"
                 "受到的具体措施类型出具警示函受到调查或处罚等措施的原因人员管理、内部控制问题"
                 "受到处罚的依据相关规定。现已完成全部整改工作，并向监管机构报送整改报告。"},
            ],
        },
    }
    actual = _remediation_management_assessment(inputs)
    assert actual["conclusion"] not in actual["paragraphs"]
    assert "2026-06-30" in actual["paragraphs"][0] and "张某、李某" in actual["paragraphs"][0]
    assert "2025-11-04，某监管局向示例基金管理有限公司出具警示函" in actual["paragraphs"][1]
    assert "人员管理、内部控制问题" in actual["paragraphs"][1]
    assert actual["paragraphs"][2] == "管理公司在报告中表示已完成整改，并已向监管机构报送整改报告。"
    assert "还不能确认监管机构是否认可" in actual["conclusion"]
    assert "不能直接理解为本基金经理个人受到处罚" in actual["limitations"][0]
    assert "整改观察档" not in str(actual) and "55" not in str(actual)
    inputs["manager_review"]["governance_state"] = "NORMAL_REVIEWED"
    assert _remediation_management_assessment(inputs) is None
    inputs["manager_review"]["governance_state"] = "REMEDIATION_DISCLOSED"
    inputs["manager_review"]["claims"][1]["quote"] = "尚未核清日期、措施和原因。"
    assert _remediation_management_assessment(inputs) is None


def test_major_holdings_explains_sample_counts_and_all_share_denominator():
    from app.services.fund_rating_narrative_adapters import _major_holdings_assessment

    holding = {
        "stocks": [{"name": "甲股票"}, {"name": "乙股票"}],
        "bonds": [{"name": name} for name in ("甲国债", "农发债", "银行永续债甲", "银行永续债乙", "乙国债")],
        "major_risk_concentration_cny": "11.52", "net_assets": "100",
    }
    actual = _major_holdings_assessment(holding, "C", "88.48")
    paragraphs = " ".join(actual["paragraphs"])
    assert "前2只股票" in paragraphs and "前5只债券中的3只非国债" in paragraphs
    assert "各类份额合计净资产" in paragraphs and "分母不是仅C类" in paragraphs
    assert "本项目" in paragraphs and "100－11.52＝88.48分" in paragraphs
    assert "国债和其他未纳入的持仓" in actual["limitations"][0]
    assert "不代表整只基金风险低" in actual["limitations"][0]
    assert len(actual["paragraphs"]) == 2 and len(actual["limitations"]) == 1
    assert actual["conclusion"] not in actual["paragraphs"]
    assert _major_holdings_assessment({}, "C", None) is None
    assert _major_holdings_assessment(holding, "C", None) is None
    holding["major_risk_concentration_cny"] = "120"
    capped = _major_holdings_assessment(holding, "C", "0")
    assert "低于0，因此按0.00分计" in capped["paragraphs"][1]


def test_holder_liquidity_keeps_actual_flow_distinct_from_future_payment_promises():
    from app.services.fund_rating_narrative_adapters import _holder_liquidity_assessment

    inputs = {"share_class": "C", "holder_review": {
        "execution_profile": "TRUSTEE_REDEMPTION_PRICING_AND_REPORT_REVIEW",
        "actual_effect_review_complete": True, "confirmed_pressure_effects": [],
        "mechanism": {"quote": "巨额赎回时，支付投资人的赎回申请有困难，可部分延期赎回。"
                      "选择延期赎回，无优先权，按下一开放日的基金份额净值处理。"
                      "连续2个开放日以上发生巨额赎回，如基金管理人认为有必要可暂停，付款不得超过20个工作日。"
                      "场内巨额赎回另行处理。"},
    }}
    raw = {"reported_redemption_shares": "123456"}
    actual = _holder_liquidity_assessment(inputs, raw)
    paragraphs = " ".join(actual["paragraphs"])
    assert "C类" in paragraphs and "不是单日赎回量" in paragraphs
    assert "单位是“份”，不是现金金额" in paragraphs
    assert "合同赎回安排" in paragraphs and "实际申赎数据和托管核查资料" in paragraphs
    assert "固定给75分" in paragraphs and "没有按逐笔到账时间计算" in paragraphs
    assert "不表示本期已经发生延期" in paragraphs and "申请渠道" in paragraphs
    assert "20个工作日" not in paragraphs and "无优先权" not in paragraphs
    assert len(actual["paragraphs"]) == 3 and len(actual["limitations"]) == 1
    assert actual["conclusion"] not in actual["paragraphs"]
    inputs["holder_review"]["mechanism"]["quote"] = "赎回条款尚缺细节。"
    limited = _holder_liquidity_assessment(inputs, raw)
    assert len(limited["paragraphs"]) == 2 and "巨额赎回延期条款" not in str(limited)
    inputs["holder_review"]["confirmed_pressure_effects"] = ["已确认压力影响"]
    assert _holder_liquidity_assessment(inputs, raw) is None
