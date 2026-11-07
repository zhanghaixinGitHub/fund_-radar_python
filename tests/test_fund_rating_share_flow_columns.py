"""份额变动表的并列折行仍须唯一识别A/C，不由金额或份额代码猜列。"""

import pytest
from app.services.fund_rating_direct_index import _flows


def report(heading, closing="90.00 240.00"):
    return [f"""开放式基金份额变动
单位：份
{heading}
基金合同生效日 100.00 200.00
本报告期期初基金份额总额 100.00 200.00
本报告期基金总申购份额 10.00 50.00
减：本报告期基金总赎回份额 20.00 10.00
本报告期基金拆分变动份额 - -
本报告期期末基金份额总额 {closing}
重大事件揭示
"""]


@pytest.mark.parametrize("heading", [
    "项目 隔离指数联接A 隔离指数联接C",
    "隔离指数ETF发  隔离指数ETF发\n项目\n起联接A 起联接C",
])
def test_verified_original_header_forms_keep_actual_share_flows(heading):
    rows, proof = _flows(report(heading))
    assert [r["share"] for r in rows] == ["A", "C"]
    assert [r["redeemed"] for r in rows] == ["20.00", "10.00"]
    assert len(proof) == 2


@pytest.mark.parametrize("heading", [
    "隔离指数ETF发  另一个指数ETF发\n项目\n起联接A 起联接C",
    "隔离指数ETF发  隔离指数ETF发\n项目\n起联接C 起联接A",
    "隔离指数ETF发  隔离指数ETF发\n项目\n起联接A 另一个后缀C",
    "隔离指数ETF发  隔离指数ETF发\n项目\n起联接A 起联接C 起联接D",
])
def test_wrong_identity_reversed_columns_or_extra_share_is_rejected(heading):
    with pytest.raises(ValueError, match="SHARE_FLOW_COLUMNS"):
        _flows(report(heading))


def test_wrapped_headers_do_not_relax_share_balance_conservation():
    with pytest.raises(ValueError):
        _flows(report("隔离指数ETF发 隔离指数ETF发\n项目\n起联接A 起联接C", closing="91.00 240.00"))
