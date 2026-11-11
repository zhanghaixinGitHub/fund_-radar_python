"""A/C/D原表适配须保留同名列序和各份额守恒。"""

import pytest
from app.services.fund_rating_direct_index import _flows

TABLE = """开放式基金份额变动
单位：份
项目 大成匠心卓越三年持有混合A 大成匠心卓越三年持有混合C 大成匠心卓越三年持有混合D
基金合同生效日 404,075,986.62 29,248,160.26 -
本报告期期初基金份额总额 1,851,159,559.98 162,187,184.83 132,919,414.64
本报告期基金总申购份额 256,901,388.44 20,181,835.78 35,803,456.18
减:本报告期基金总赎回份额 68,200,116.98 9,926,131.97 318,405.30
本报告期基金拆分变动份额 - - -
本报告期期末基金份额总额 2,039,860,831.44 172,442,888.64 168,404,465.52
重大事件揭示
"""


def test_three_actual_share_columns_reconcile_and_remain_distinct():
    rows, proof = _flows([TABLE], additional_shares=("D",))
    assert [row["share"] for row in rows] == ["A", "C", "D"]
    assert [row["redeemed"] for row in rows] == ["68200116.98", "9926131.97", "318405.30"]
    assert all(row["split"] == "0.00" for row in proof)


@pytest.mark.parametrize("suffix", ["E", "Y", "另一个基金D"])
def test_wrong_class_or_product_does_not_match_same_three_column_layout(suffix):
    table = TABLE.replace("大成匠心卓越三年持有混合D", "大成匠心卓越三年持有混合" + suffix)
    with pytest.raises(ValueError, match="SHARE_FLOW_COLUMNS"):
        _flows([table], additional_shares=("D",))


def test_swapped_c_and_d_redemptions_are_rejected_by_share_conservation():
    table = TABLE.replace("9,926,131.97 318,405.30", "318,405.30 9,926,131.97")
    with pytest.raises(ValueError, match="NOT_RECONCILED"):
        _flows([table], additional_shares=("D",))
