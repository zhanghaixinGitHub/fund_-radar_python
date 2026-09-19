"""模拟账本费率契约：天天基金 f10 解析结果，供 Java 核心服务落库 sim_fee_rule。"""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class FeeModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class FeeBandModel(FeeModel):
    """单档赎回费率区间；max_days 为 None 表示最高档无上限。"""

    min_days: int
    max_days: int | None
    rate: Decimal


class FundFeeProfile(FeeModel):
    fund_code: str
    fund_name: str
    purchase_rate: Decimal
    purchase_original_rate: Decimal
    discount_info: str | None
    redeem_bands: tuple[FeeBandModel, ...]
    data_source: str = "EASTMONEY_F10"
