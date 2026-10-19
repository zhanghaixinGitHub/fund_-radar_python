"""质押受限资产的合计、单位及缺失边界；样例不是任何真实校准产品。"""

import pytest
from app.services.fund_quality_v2_pledged_bonds import PledgedBondIncomplete, pledged_bond_table
from app.services.fund_quality_v2_restrictions import (
    restricted_numeric_view,
    restricted_securities,
    restriction_page_header_view,
)


def section():
    return """6.4.12.3 期末债券正回购交易中作为抵押的债券
6.4.12.3.1 银行间市场债券正回购
本金余额900.00元，以如下债券作为质押：
金额单位：人民币元
债券代码 债券名称 回购到期日 期末估值单价 数量（张） 期末估值总额
019999 测试国债 2026-07-02 100.00 5.00 500.00
123456789 测试中票 2026-07-03 100.00 6.00 600.00
合计 11.00 1,100.00
6.4.12.3.2 交易所市场债券正回购
无。
"""


def test_actual_collateral_values_are_not_financing_balance_and_9_digit_bonds_are_preserved():
    result = pledged_bond_table(section(), bond_amount="2000")
    assert result["amount_cny"] == "1100.00"
    assert [row["code"] for row in result["rows"]] == ["019999", "123456789"]
    assert result["rows"][0]["quantity_unit"] == "bond"
    assert result["markets"][1]["basis"] == "EXPLICIT_NO_PLEDGED_BONDS"
    assert "cost" not in result["rows"][0]  # 没有披露取得成本时不填0。


def test_wrapped_bond_name_is_kept_and_unresolved_second_market_preserves_only_partial_facts():
    text = section().replace("测试中票", "测试\n中票")
    assert pledged_bond_table(text, bond_amount="2000")["rows"][1]["name"] == "测试中票"
    text = text.replace("\n无。", "\n融资余额500.00元，按比例折算标准券。")
    with pytest.raises(PledgedBondIncomplete, match="NOT_DISCLOSED:EXCHANGE") as error:
        pledged_bond_table(text, bond_amount="2000")
    assert len(error.value.evidence["known_rows"]) == 2
    assert error.value.evidence["unresolved_market"] == "EXCHANGE"
    assert "amount_cny" not in error.value.evidence


def test_report_explicit_zero_balance_and_no_collateral_is_not_a_missing_statement():
    body = """6.4.12.3.1 银行间市场债券正回购
截至期末卖出回购证券款余额为0，无抵押债券。
6.4.12.3.2 交易所市场债券正回购
截至期末卖出回购证券款余额为0.00元，无抵押债券。"""
    assert pledged_bond_table(body, bond_amount="0")["amount_cny"] == "0"


def test_unnumbered_market_titles_require_actual_parent_section_and_both_market_statements():
    body = (
        "6.4.12.3 期末债券正回购交易中作为抵押的债券\n"
        "银行间市场债券正回购\n本基金无因从事银行间市场债券正回购交易形成的卖出回购证券款余额。\n"
        "交易所市场债券正回购\n本基金无因从事交易所市场债券正回购交易形成的卖出回购证券款余额。"
    )
    result = pledged_bond_table(body, bond_amount="100")
    assert result["amount_cny"] == "0" and len(result["markets"]) == 2
    assert all(row["heading_kind"] == "STANDALONE_TITLE_WITHIN_PARENT_PLEDGE_SECTION" for row in result["markets"])
    for invalid in [body.split("\n", 1)[1], body[:body.index("交易所市场债券正回购")]]:
        with pytest.raises(ValueError, match="PLEDGED_BONDS"):
            pledged_bond_table(invalid, bond_amount="100")


def test_market_number_cannot_override_title_and_negative_does_not_hide_actual_positive_integer_balance():
    body = section().replace("6.4.12.3.1 银行间", "6.4.12.3.2 银行间")
    with pytest.raises(ValueError, match="NUMBER_AND_MARKET_CONFLICT"):
        pledged_bond_table(body, bond_amount="2000")
    body = section().replace("\n无。", "\n本基金无因从事交易所市场债券正回购交易形成的卖出回购证券款余额。"
                                      "融资余额500元。")
    with pytest.raises(ValueError, match="NEGATIVE_AND_POSITIVE_CONFLICT"):
        pledged_bond_table(body, bond_amount="2000")


def test_bare_page_number_requires_first_line_and_consecutive_actual_page_ordinals():
    pages = [
        "测试基金2025年年度报告\n正文",
        "2\n本基金无因从事交易形成",
        "3\n的卖出回购证券款余额。\n数量\n51",
        "4\n普通正文\n4",
    ]
    original = list(pages)
    view, evidence = restriction_page_header_view(pages)
    assert "形成的卖出回购证券款余额" in "".join(view.split())
    assert evidence["sequential_bare_page_numbers"] == [2, 3, 4]
    assert "数量\n51" in view and "普通正文\n4" in view and pages == original
    chinese_year = [line.replace("2025年", "二0二五年") for line in pages]
    assert restriction_page_header_view(chinese_year)[1]["sequential_bare_page_numbers"] == [2, 3, 4]
    for unverified in [["1\n数量", "2\n数量", "3\n数量"],
                       ["测试基金2025年年度报告", "52\n数量", "53\n数量", "54\n数量"]]:
        value, proof = restriction_page_header_view(unverified)
        assert value == "\n".join(unverified) and not proof["sequential_bare_page_numbers"]


@pytest.mark.parametrize("problem", ["financing_only", "unit", "quantity", "amount", "balance", "duplicate"])
def test_incomplete_or_conflicting_collateral_does_not_produce_restricted_amount(problem):
    text, balance = section(), "2000"
    if problem == "financing_only":
        text = text[: text.index("金额单位")] + "应按比例折算标准券。\n6.4.12.3.2 交易所市场债券正回购\n无。"
    elif problem == "unit":
        text = text.replace("人民币元", "人民币万元")
    elif problem == "quantity":
        text = text.replace("合计 11.00", "合计 12.00")
    elif problem == "amount":
        text = text.replace("1,100.00", "1,200.00")
    elif problem == "balance":
        balance = "1099.99"
    else:
        text = text.replace("123456789 测试中票 2026-07-03", "019999 测试国债 2026-07-02")
    with pytest.raises(ValueError, match="PLEDGED_BONDS"):
        pledged_bond_table(text, bond_amount=balance)


def test_only_repeated_report_headers_and_explicit_page_numbers_can_join_a_broken_negative_sentence():
    pages = [
        "测试混合 2026 年中期报告\n6.4.12.3.1 银行间市场债券正回购\n本基金无因从事交易形",
        "测试混合 2026 年中期报告\n第42页 共70页\n成的卖出回购证券款余额。\n非页眉说明要保留",
    ]
    view, evidence = restriction_page_header_view(pages)
    assert "交易形成的卖出回购证券款余额" in "".join(view.split())
    assert "非页眉说明要保留" in view and len(evidence["removed_headers"]) == 3
    assert "测试混合 2026 年中期报告" in pages[0]  # 原页列表不改写。
    unique = ["仅出现一次的测试报告 2026 年中期报告\n正文"]
    assert restriction_page_header_view(unique)[0] == unique[0]


def test_restricted_bond_wrap_is_restored_from_visible_digits_and_added_to_pledged_values():
    text = (
        """6.4.12.1 因认购新发流通受限证券
本基金未持有认购新发流通受限股票。
6.4.12.1.2 受限证券类别：债券
118071 测试转债 2026年6月23日 1个月内 新债未上市 100.00 100.00 770 76,999.8
3
77,001.1
8 -
6.4.12.2 期末持有的暂时停牌等流通受限股票
本基金未持有暂时停牌等流通受限的股票。
"""
        + section()
        + "6.4.12.4 期末参与转融通\n本基金无参与转融通证券出借业务的证券。\n6.4.13 金融工具风险"
    )
    result = restricted_securities([text], [], "100000")
    assert result["restricted_amount"] == "78101.18"
    assert result["rows"][0]["cost"] == "76999.83"
    assert result["rows"][0]["amount"] == "77001.18"
    with pytest.raises(ValueError, match="ALL_BONDS_EXCEED"):
        restricted_securities([text], [], "78000")
    with pytest.raises(ValueError, match="PLEDGE_OVERLAP"):
        restricted_securities([text.replace("118071 测试转债", "019999 测试转债")], [], "100000")


def test_empty_stock_remark_is_allowed_only_when_visible_amount_tuple_ends_the_table_row():
    original = """6.4.12.1 因认购新发流通受限证券
600001 测试股票 2026年6月23日 6个月 10.00 12.00 10 100.00 120.00
6.4.12.2 期末持有的暂时停牌等流通受限股票
本基金未持有暂时停牌等流通受限股票。
6.4.12.3 期末债券正回购
6.4.12.3.1 银行间市场债券正回购
无。
6.4.12.3.2 交易所市场债券正回购
无。
6.4.12.4 期末参与转融通
本基金无参与转融通证券出借业务的证券。
6.4.13 金融工具风险"""
    positions = [
        {"stock_code": "600001.SH", "stock_name": "测试股票", "quantity_shares": "50", "market_value_cny": "600"}
    ]
    assert restricted_securities([original], positions, "0")["restricted_amount"] == "120.00"
    with pytest.raises(ValueError, match="RESTRICTED_VALUE"):
        restricted_securities([original.replace("100.00 120.00", "100.00 未披露")], positions, "0")


def test_only_incomplete_thousands_group_is_joined_and_complete_quantity_is_not_glued_to_cost():
    assert restricted_numeric_view("6,488 65,65\n8.56\n179,9\n12.24") == "6,488 65,658.56\n179,912.24"
    assert restricted_numeric_view("552\n8.93\n1,234.56") == "552\n8.93\n1,234.56"
    assert restricted_numeric_view("350,0\n00\n29,78\n8,500.\n00") == "350,000\n29,788,500.00"
    assert restricted_numeric_view("125,0\n40,00\n0.00") == "125,040,000.00"
    with pytest.raises(ValueError, match="MONEY_WRAP_AMBIGUOUS"):
        restricted_numeric_view("179,9\n123.45")


def test_complete_empty_bond_table_requires_every_column_and_all_five_explicit_numeric_dashes():
    header = (
        "证券代码 证券名称 成功认购日 受限期 流通受限类型 认购价格 期末估值单价 "
        "数量（单位：张） 期末成本总额 期末估值总额 备注"
    )
    body = (
        """6.4.12.1 因认购新发证券
本基金未持有因认购新发流通受限股票。
6.4.12.1.2 受限证券类别：债券
"""
        + header
        + "\n- - - - -\n6.4.12.2 期末持有的暂时停牌等流通受限股票\n本基金未持有暂时停牌等流通受限股票。\n"
        + section().replace(section()[section().index("本金余额") : section().index("6.4.12.3.2")], "无。\n")
        + "6.4.12.4 期末参与转融通\n本基金无参与转融通证券出借业务的证券。\n6.4.13 金融工具风险"
    )
    result = restricted_securities([body], [], "2000")
    assert result["restricted_amount"] == "0" and "empty_bond_table_evidence" in result
    for invalid in [
        body.replace("期末成本总额", ""),
        body.replace("- - - - -", "- - - -"),
        body.replace("- - - - -", "未知债券 - - - - -"),
    ]:
        with pytest.raises(ValueError, match="BOND_TABLE_UNVERIFIED"):
            restricted_securities([invalid], [], "2000")
