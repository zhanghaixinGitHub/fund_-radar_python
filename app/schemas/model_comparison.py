"""离线比较的输入契约；不依赖 Chronos，不接受答案或任意附加特征。"""

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Fund = Literal["001632", "006730", "008888"]


class ComparisonInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    fund_code: Fund
    cutoff_date: date = Field(ge=date(2022, 1, 1), le=date(2024, 12, 31))
    batch_id: UUID
    anchor_nav_date: date
    anchor_lag_sessions: Literal[0, 1]
    label_base_date: date  # 日历元数据，不是起点的未知净值。
    label_end_date: date
    history_dates: tuple[date, ...] = Field(min_length=61, max_length=61)
    history_values: tuple[Decimal, ...] = Field(min_length=61, max_length=61)
    history_available_at: tuple[date, ...] = Field(min_length=61, max_length=61)
    x: tuple[Decimal, ...] = Field(min_length=7, max_length=7)
    source_feature_hash: Hash
    input_hash: Hash

    @property
    def key(self) -> str:
        return f"{self.fund_code}:{self.cutoff_date}:{self.label_end_date}"

    @model_validator(mode="after")
    def check_values(self) -> Self:
        if any(not n.is_finite() or n <= 0 for n in self.history_values):
            raise ValueError("nonpositive/nonfinite history")
        if any(not n.is_finite() for n in self.x):
            raise ValueError("nonfinite features")
        if any(d > self.cutoff_date for d in self.history_available_at):
            raise ValueError("history not yet available")
        if not self.anchor_nav_date <= self.label_base_date <= self.cutoff_date < self.label_end_date:
            raise ValueError("invalid input dates")
        if self.label_end_date > date(2024, 12, 31):
            raise ValueError("TEST_PERIOD_PROTECTED")
        if (
            tuple(sorted(set(self.history_dates))) != self.history_dates
            or self.history_dates[-1] != self.anchor_nav_date
        ):
            raise ValueError("invalid history dates")
        return self


class ComparisonAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    key: str
    input_hash: Hash
    label_hash: Hash
    label_available_at: date = Field(le=date(2024, 12, 31))
    y: Literal[0, 1]
    future_return_20d: Decimal

    @model_validator(mode="after")
    def check_direction(self) -> Self:
        if not self.future_return_20d.is_finite() or self.y != int(self.future_return_20d > 0):
            raise ValueError("label direction mismatch")
        return self
