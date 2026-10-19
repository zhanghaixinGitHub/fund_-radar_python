"""配置投影的数学反例及来源拒收；工程假样本不属于真实校准/验证。"""

import hashlib
import random
from datetime import date
from fractions import Fraction

import numpy as np
import pytest
from app.services.fund_quality_v2_allocation import (
    allocation_history,
    contract_allocation_clauses,
    project_half_l1,
    report_allocation_point,
    report_product_net_assets,
)
from scipy.optimize import linprog


def limits(lower="0.8", upper="0.95", hk="0.5"):
    q = Fraction(hk)
    return [{"coefficients": [1, 1, 0], "upper": upper},
            {"coefficients": [-1, -1, 0], "upper": -Fraction(lower)},
            {"coefficients": [-q, 1-q, 0], "upper": 0}]


def exact(result):
    d = result["distance"]
    return Fraction(int(d["numerator"]), int(d["denominator"]))


@pytest.mark.parametrize("point,expected", [
    (["0.6", "0.1", "0.3"], Fraction("0.1")),
    (["0.1", "0.7", "0.2"], Fraction("0.3")),
    (["0.7", "0.29", "0.01"], Fraction("0.04")),
    (["0.5", "0.35", "0.15"], Fraction(0)),
])
def test_exact_projection_handles_overlapping_equity_and_hk_constraints(point, expected):
    assert exact(project_half_l1(point, limits())) == expected


def test_no_tolerance_can_hide_a_small_real_breach():
    epsilon = Fraction(1, 10**30)
    assert exact(project_half_l1([".5", ".45", ".05"], limits(Fraction(".95") + epsilon, "1"))) == epsilon


@pytest.mark.parametrize("bad", [True, 0.2, "NaN", "Infinity", "abc", "1E100000"])
def test_rejects_non_exact_or_unbounded_numbers(bad):
    with pytest.raises(ValueError):
        project_half_l1([bad, "0.5"], [])


def test_empty_allowed_set_and_incomplete_weights_are_not_zero():
    with pytest.raises(ValueError, match="EMPTY"):
        project_half_l1([".5", ".4", ".1"], limits(".96", ".95"))
    with pytest.raises(ValueError, match="COMPLETE_POINT"):
        project_half_l1([".5", ".4", ".2"], [])


def test_independent_epigraph_linear_program_checks_random_exact_projections():
    """另一种算法及变量表示核算；有理数结果不依赖浮点比较来接纳。"""
    rng = random.Random(20261017)
    for size in (2, 3, 4):
        for _ in range(10):
            integer = [rng.randint(1, 100) for _ in range(size)]
            point = [Fraction(i, sum(integer)) for i in integer]
            rows = [{"coefficients": [int(i == j) for j in range(size)], "upper": ".8"}
                    for i in range(size)]
            result = project_half_l1(point, rows)
            # t_i >= |x_i-w_i|，使用HiGHS，独立于精确顶点枚举及折点划分。
            matrix = [list(map(float, row["coefficients"])) + [0.0]*size for row in rows]
            rhs = [float(row["upper"]) for row in rows]
            for i, weight in enumerate(point):
                for sign in (1, -1):
                    matrix.append([sign*float(i == j) for j in range(size)] +
                                  [-float(i == j) for j in range(size)])
                    rhs.append(sign*float(weight))
            solution = linprog([0]*size + [.5]*size, A_ub=np.array(matrix), b_ub=rhs,
                               A_eq=[[1]*size + [0]*size], b_eq=[1], method="highs")
            assert solution.success
            assert float(exact(result)) == pytest.approx(solution.fun, abs=1e-10)


def contract():
    clause = ("本基金投资组合中股票、存托凭证的投资比例为基金资产的80%-95%，"
              "其中港股通标的股票的投资比例不超过本基金股票资产的50%")
    return ["测试股票型证券投资基金基金合同\n二、投资范围" + clause +
            "。本基金现金或到期日在一年以内的政府债券不低于净资产5%。三、投资策略"
            "四、投资限制1、组合限制基金的投资组合应遵循以下限制：（1）" + clause +
            "；（2）单一发行人限制；除上述（2）情形外须调整。2、禁止行为"]


def test_contract_parser_preserves_unresolved_clauses_and_adjustment_cross_references():
    result = contract_allocation_clauses(contract())
    assert result["stock_lower"]["value"] == "0.8"
    assert len(result["combination_clauses"]) == 2
    assert result["combination_clauses"][1]["clause_id"] == 2
    assert "除上述(2)" in result["adjustment_and_transition_excerpt"]
    assert len(result["full_allowed_set_remaining"]) == 2
    assert result["ratio_denominator_evidence"]["state"] == "CONTRACT_RATIO_DENOMINATOR_UNRESOLVED"
    explicit = contract_allocation_clauses([contract()[0].replace("基金资产的80", "基金资产净值的80")])
    assert explicit["ratio_denominator_evidence"]["economic_basis"] == "NET_ASSETS"
    with pytest.raises(ValueError, match="CONFLICTING"):
        contract_allocation_clauses([contract()[0].replace("80%-95%", "70%-95%", 1)])


def asset_point(hk="20.00"):
    return report_allocation_point(
        "权益投资100.0083.33其中：股票100.0083.33银行存款--合计120.00100.00"
        "注：上表权益投资中通过港股通交易机制投资的港股金额人民币" + hk + "元，占净值20%。",
        {"合计": "120.00", "权益投资": "100.00", "银行存款": None},
    )


def test_original_money_is_not_reconstructed_from_rounded_nav_percent_or_unknown_rows():
    point = asset_point()
    assert point["amounts_cny"]["hk_stock"] == "20.00"
    assert point["denominator"] == "TOTAL_ASSETS_NOT_NAV"
    assert point["unobserved_asset_rows_imputed"] == []
    assert point["weights"][1]["denominator"] == "6"
    with pytest.raises(ValueError, match="UNPARSED"):
        report_allocation_point("本基金可以不投资港股。", {"合计": "120", "权益投资": "100"})


def test_original_hk_table_and_explicit_absence_fill_parser_gap_without_imputation():
    block = "其中：股票100.0083.33"
    domestic = "5.2.1报告期末按行业分类的境内股票投资组合\n合计 80.00 80.00\n"
    foreign = "5.2.2报告期末按行业分类的港股通投资股票投资组合\n合计 20.00 20.00\n"
    end = "5.3报告期末按公允价值排序股票"
    amounts = {"合计": "120", "权益投资": "100"}
    result = report_allocation_point(block, amounts, industry_text=domestic+foreign+end)
    assert result["amounts_cny"]["hk_stock"] == "20.00"
    absence = "5.2.2报告期末按行业分类的港股通投资股票投资组合\n本基金本报告期末未持有港股通投资股票。\n"
    result = report_allocation_point(block, amounts, industry_text=domestic.replace("80.00", "100.00")+absence+end)
    assert result["amounts_cny"]["hk_stock"] == "0.00"
    assert result["industry_reconciliation"]["basis"] == "EXPLICIT_PERIOD_END_HK_ABSENCE"
    with pytest.raises(ValueError, match="NOT_RECONCILED"):
        report_allocation_point(block, amounts, industry_text=domestic+absence+end)
    with pytest.raises(ValueError, match="MISSING"):
        report_allocation_point(block, amounts, industry_text=domestic+absence.replace("未持有", "可以不投资")+end)


def quarter_financial():
    return ["测试股票型证券投资基金2026年第1季度报告\n"
            "下属分级基金的基金简称 测试股票A 测试股票C\n下属分级基金的交易代码 123456 123457\n"
            "3.1主要财务指标\n单位：人民币元\n报告期（2026年1月1日－2026年3月31日）\n"
            "测试股票A 测试股票C\n1.本期已实现收益 10.00 5.00\n"
            "4.期末基金资产净值 60.00 40.00\n5.期末基金份额净值 1.20 1.10\n3.2基金净值表现"]


def test_current_quarter_product_nav_requires_actual_period_currency_and_all_share_columns():
    result = report_product_net_assets(quarter_financial(), observed_on="2026-03-31", total_assets="120")
    assert result["net_assets_cny"] == "100"
    assert result["proof"]["share_codes"] == ["123456", "123457"]
    assert result["proof"]["share_net_assets_cny"] == ["60.00", "40.00"]
    for original, changed in [("人民币元", "人民币万元"), ("2026年3月31日", "2025年3月31日"),
                              ("测试股票A 测试股票C\n1.", "测试股票C 测试股票A\n1."),
                              ("60.00 40.00", "60.00"), ("123456 123457", "123456 123457 123458")]:
        with pytest.raises(ValueError):
            report_product_net_assets([quarter_financial()[0].replace(original, changed)],
                                      observed_on="2026-03-31", total_assets="120")
    with pytest.raises(ValueError, match="EXCEEDS"):
        report_product_net_assets(quarter_financial(), observed_on="2026-03-31", total_assets="99")


def test_first_year_one_column_balance_never_uses_a_missing_comparison_as_zero():
    source = ["测试证券投资基金2023年年度报告\n基金合同生效日2023年3月21日\n"
              "§7 年度财务报表\n7.1 资产负债表\n会计主体：测试证券投资基金\n"
              "报告截止日：2023年12月31日\n单位：人民币元\n资产 附注号 本期末2023年12月31日\n资产：\n"
              "资产总计 120.00\n负债合计 20.00\n净资产合计 100.00\n负债和净资产总计 120.00\n7.2 利润表"]
    result = report_product_net_assets(source, observed_on="2023-12-31", total_assets="120")
    assert result["net_assets_cny"] == "100"
    assert result["proof"]["basis"] == "EXPLICIT_FIRST_YEAR_SINGLE_CURRENT_CNY_COLUMN"
    for old, new in [("基金合同生效日2023", "基金合同生效日2022"), ("人民币元", "人民币万元"),
                     ("负债合计 20.00", "负债合计 19.99"), ("净资产合计 100.00", "净资产合计 -"),
                     ("资产 附注号 本期末2023年12月31日", "资产 附注号 本期末2023年12月31日 上年度末2022年12月31日")]:
        with pytest.raises(ValueError):
            report_product_net_assets([source[0].replace(old, new)], observed_on="2023-12-31", total_assets="120")


def test_real_hash_and_all_counterparts_are_required_and_subset_zero_never_scores(tmp_path, monkeypatch):
    from app.services import fund_rating_public_market

    source = tmp_path / "sources" / "contract.pdf"
    source.parent.mkdir()
    source.write_bytes(b"test-only-original-reader-substitute")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    ref = {"label": "historical-contract", "title": "测试股票型证券投资基金基金合同（2023年7月10日修订）",
           "file": "sources/contract.pdf", "published_on": "2023-07-10", "sha256": sha,
           "url": "https://static.cmfchina.com/fundarticle/20230710/123456.pdf"}
    monkeypatch.setattr(fund_rating_public_market, "public_document", lambda *args: (contract(), None))
    rows = [{"observed_on": "2024-12-31", "source_hashes": ["a"*64], "allocation_asset_point": asset_point()},
            {"observed_on": "2024-12-31", "source_hashes": ["b"*64], "allocation_asset_point": asset_point()}]
    kwargs = dict(refs=[ref], root=tmp_path, reports=rows, required_dates=["2024-12-31"],
                  legal_name="测试股票型证券投资基金", as_of=date(2025, 5, 1), code="123456")
    result = allocation_history(**kwargs)
    assert result["observation_count"] == 1 and result["scoring_metric"] is None
    assert exact(result["observations"][0]["conditional_subset_projections"][0]) == 0
    assert result["observations"][0]["full_metric_value"] is None
    assert result["contract_candidates"][0]["version_binding_gap"]
    first = result["observations"][0]["conditional_subset_projections"][0]
    assert first["denominator_assumption"] == "HYPOTHETICAL_FUND_ASSETS_EQUALS_TOTAL_ASSETS"
    assert "CONTRACT_RATIO_DENOMINATOR_UNRESOLVED" in first["conditions"]
    for row in rows:
        row["allocation_asset_point"]["net_asset_evidence"] = {"net_assets_cny": "100.00"}
    scenarios = allocation_history(**kwargs)["observations"][0]["conditional_subset_projections"]
    assert [p["contract_denominator_scenario"] for p in scenarios] == ["TOTAL_ASSETS", "NET_ASSETS"]
    assert scenarios[1]["source_threshold_rescaling"]["numerator"] == "5"
    assert scenarios[1]["source_threshold_rescaling"]["denominator"] == "6"
    rows[1]["allocation_asset_point"]["net_asset_evidence"] = {"net_assets_cny": "101.00"}
    conflict = allocation_history(**kwargs)["observations"][0]
    assert "CONFLICT" in conflict["reason"] and len(conflict["net_asset_counterpart_values"]) == 2
    rows[1]["allocation_asset_point"]["net_asset_evidence"] = {"net_assets_cny": "100.00"}
    rows[1]["allocation_asset_point"] = asset_point("21.00")
    assert "CONFLICT" in allocation_history(**kwargs)["observations"][0]["reason"]
    source.write_bytes(source.read_bytes() + b"changed")
    assert allocation_history(**kwargs)["contract_parse_gaps"][0]["reason"].endswith("HASH_CHANGED")
