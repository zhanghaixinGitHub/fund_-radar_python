"""基金份额关联的公共读模型；不向浏览器暴露采集地址或本地原件路径。"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class RelatedShare(BaseModel):
    fundCode: str = Field(pattern=r"^[0-9]{6}$")
    fundName: str
    shareClass: str | None = None
    currency: str | None = None


class FundShareClasses(BaseModel):
    fundCode: str = Field(pattern=r"^[0-9]{6}$")
    status: Literal["VERIFIED", "STALE", "UNVERIFIED"]
    asOfDate: date | None = None
    checkedAt: datetime | None = None
    shares: list[RelatedShare] = Field(default_factory=list, max_length=30)
