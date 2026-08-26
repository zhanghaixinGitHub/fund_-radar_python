"""一次性手工核验目录样本的离线契约测试。"""

from app.commands.bootstrap_verified_fund_catalog import VERIFIED_FUNDS
from app.schemas.fund import InternalFundSummary


def test_verified_fund_catalog_contains_the_six_user_confirmed_share_classes() -> None:
    """目录样本必须只包含六条完整可核验的截图基金份额，且代码不重复。"""
    assert [fund.fund_code for fund in VERIFIED_FUNDS] == [
        "010710",
        "160323",
        "013275",
        "007832",
        "002112",
        "005312",
    ]
    assert len({fund.fund_code for fund in VERIFIED_FUNDS}) == len(VERIFIED_FUNDS)


def test_catalog_contract_allows_a_real_fund_without_synced_nav() -> None:
    """没有合规净值时必须返回空日期，不能伪造实时数据日期。"""
    summary = InternalFundSummary(
        fund_code="010710",
        fund_name="安信医药健康主题股票C",
        fund_type="STOCK",
        status="ACTIVE",
        as_of_date=None,
    )
    assert summary.as_of_date is None
