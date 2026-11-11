import pytest
from app.services.fund_rating_direct_index import _flows


def test_wide_chinese_header_and_detached_a_keep_two_real_share_columns():
    table = """9 开放式基金份额变动
                                                                              单位：份
            项目              国泰国证有色金属行业指数  国泰国证有色金属行业指数 C
                                            A
 基金合同生效日（2021 年 1 月 1            2,620,186,171.84                            -
 日）基金份额总额
 本报告期期初基金份额总额                  815,132,784.81                486,384,737.80
 本报告期基金总申购份额                    1,559,115,943.65                1,630,947,308.97
 减：本报告期基金总赎回份额                1,375,974,071.95                1,627,285,507.07
 本报告期基金拆分变动份额                              -                            -
 本报告期期末基金份额总额                  998,274,656.51                490,046,539.70
10 重大事件揭示
"""
    rows, _ = _flows([table])
    assert [r["share"] for r in rows] == ["A", "C"]
    assert rows[1]["redeemed"] == "1627285507.07"
    with pytest.raises(ValueError, match="WRAPPED_CLASS_ALIGNMENT"):
        _flows(
            [
                table.replace(
                    "                                            A",
                    "                                                                                          A",
                )
            ]
        )
