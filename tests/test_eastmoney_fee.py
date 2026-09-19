"""天天基金 f10 费率页（jjfl）解析单测：申购费率取第一档优惠费率，赎回分档按持有自然天映射。"""

# ruff: noqa: E501 —— 下方 HTML 常量截取自真实页面，必须保持结构原样，不折行。
from decimal import Decimal

import pytest
from app.services.eastmoney_fee import FeeBand, FeeProfile, parse_jjfl_html, parse_redeem_term

# 片段截取自真实页面 fundf10.eastmoney.com/jjfl_001021.html，保持结构原样。
PAGE_001021 = """<!DOCTYPE html><html><head>
<title>华夏亚债中国指数A(001021)基金费率 _ 基金档案 _ 天天基金网</title>
</head><body>
<div class="box"><div class="boxitem w790"><h4 class="t"><label class="left">申购费率</label><label class="right"></label></h4><div class="space0"></div><table class="w650 comm jjfl"><thead><tr><th class="first je">适用金额</th><th class="last fl speciacol w230" style='width:230px;'><span class="sgfv">原费率</span><span class="sgline">|</span><div class="sgyh">天天基金优惠费率</div></th></tr></thead><tbody><tr><td class="">小于100万元</td><td><strike class='gray'>0.80%</strike>&nbsp;&nbsp;|&nbsp;&nbsp;0.08%</td></tr><tr><td class="">大于等于100万元，小于500万元</td><td><strike class='gray'>0.60%</strike>&nbsp;&nbsp;|&nbsp;&nbsp;0.06%</td></tr></tbody></table></div></div>
<div class="box"><div class="boxitem w790"><h4 class="t"><label class="left">赎回费率<a name="shfl"></a></label><label class="right"></label></h4><div class="space0"></div><table class="w650 comm jjfl"><thead><tr><th class="first je">适用期限</th><th class="last fl">赎回费率</th></tr></thead><tbody><tr><td>小于7天</td><td>1.50%</td></tr><tr><td>大于等于7天</td><td>0.00%</td></tr></tbody></table></div></div>
</body></html>"""

# C 类基金申购费为 0，页面无折扣列。
PAGE_C_CLASS = """<!DOCTYPE html><html><head>
<title>华夏国证半导体芯片ETF联接C(008888)基金费率 _ 基金档案 _ 天天基金网</title>
</head><body>
<div class="box"><div class="boxitem w790"><h4 class="t"><label class="left">申购费率</label><label class="right"></label></h4><div class="space0"></div><table class="w650 comm jjfl"><thead><tr><th class="first je">适用金额</th><th class="last fl">费率</th></tr></thead><tbody><tr><td class="">小于100万元</td><td>0.00%</td></tr></tbody></table></div></div>
<div class="box"><div class="boxitem w790"><h4 class="t"><label class="left">赎回费率<a name="shfl"></a></label><label class="right"></label></h4><div class="space0"></div><table class="w650 comm jjfl"><thead><tr><th class="first je">适用期限</th><th class="last fl">赎回费率</th></tr></thead><tbody><tr><td>小于7天</td><td>1.50%</td></tr><tr><td>大于等于7天，小于30天</td><td>0.50%</td></tr><tr><td>大于等于30天</td><td>0.00%</td></tr></tbody></table></div></div>
</body></html>"""

# 无折扣渠道的基金：只有原费率一列。
PAGE_NO_DISCOUNT = """<!DOCTYPE html><html><head>
<title>某债券基金(110027)基金费率 _ 基金档案 _ 天天基金网</title>
</head><body>
<div class="box"><div class="boxitem w790"><h4 class="t"><label class="left">申购费率</label><label class="right"></label></h4><div class="space0"></div><table class="w650 comm jjfl"><thead><tr><th class="first je">适用金额</th><th class="last fl">费率</th></tr></thead><tbody><tr><td class="">小于100万元</td><td>0.60%</td></tr></tbody></table></div></div>
<div class="box"><div class="boxitem w790"><h4 class="t"><label class="left">赎回费率<a name="shfl"></a></label><label class="right"></label></h4><div class="space0"></div><table class="w650 comm jjfl"><thead><tr><th class="first je">适用期限</th><th class="last fl">赎回费率</th></tr></thead><tbody><tr><td>小于7天</td><td>1.50%</td></tr><tr><td>大于等于7天</td><td>0.00%</td></tr></tbody></table></div></div>
</body></html>"""


def test_redeem_term_descriptions_map_to_natural_day_ranges():
    assert parse_redeem_term("小于7天") == (0, 6)
    assert parse_redeem_term("大于等于7天，小于30天") == (7, 29)
    assert parse_redeem_term("大于等于30天") == (30, None)


def test_redeem_term_rejects_non_day_units_for_manual_review():
    with pytest.raises(ValueError, match="MANUAL_REQUIRED"):
        parse_redeem_term("小于1年")


def test_jjfl_page_parses_purchase_discount_and_redeem_bands():
    profile = parse_jjfl_html(PAGE_001021, "001021")
    assert profile == FeeProfile(
        fund_code="001021",
        fund_name="华夏亚债中国指数A",
        purchase_rate=Decimal("0.0008"),
        purchase_original_rate=Decimal("0.0080"),
        discount_info="天天基金优惠费率",
        redeem_bands=(FeeBand(0, 6, Decimal("0.0150")), FeeBand(7, None, Decimal("0.0000"))),
    )


def test_c_class_fund_has_zero_purchase_fee_and_three_redeem_bands():
    profile = parse_jjfl_html(PAGE_C_CLASS, "008888")
    assert profile.purchase_rate == Decimal("0")
    assert profile.purchase_original_rate == Decimal("0")
    assert profile.discount_info is None
    assert profile.redeem_bands == (
        FeeBand(0, 6, Decimal("0.0150")),
        FeeBand(7, 29, Decimal("0.0050")),
        FeeBand(30, None, Decimal("0.0000")),
    )


def test_page_without_discount_uses_original_rate():
    profile = parse_jjfl_html(PAGE_NO_DISCOUNT, "110027")
    assert profile.purchase_rate == Decimal("0.0060")
    assert profile.purchase_original_rate == Decimal("0.0060")
    assert profile.discount_info is None


def test_fixed_amount_first_tier_requires_manual_review():
    page = PAGE_001021.replace("0.08%", "每笔1000元")
    with pytest.raises(ValueError, match="MANUAL_REQUIRED"):
        parse_jjfl_html(page, "001021")


def test_fund_code_mismatch_in_title_is_rejected():
    with pytest.raises(ValueError, match="FUND_CODE_MISMATCH"):
        parse_jjfl_html(PAGE_001021, "999999")
