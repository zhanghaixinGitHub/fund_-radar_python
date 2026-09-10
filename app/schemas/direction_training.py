"""自训练专项的输入与答案分离契约；禁止保留测试数值。"""

from datetime import date
from decimal import Decimal
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

Fund = Literal["001632", "006730", "008888"]


class DirectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    fund: Fund
    cutoff: date = Field(ge=date(2021, 1, 1), le=date(2024, 12, 31))
    anchor: date
    available_at: date
    x: tuple[FiniteFloat, ...] = Field(min_length=7, max_length=7)
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def chronology(self) -> Self:
        if not date(2021, 1, 1) <= self.anchor <= self.available_at <= self.cutoff:
            raise ValueError("INPUT_TIME_BOUNDARY")
        return self

    @property
    def key(self) -> str:
        return f"{self.fund}:{self.cutoff}"


class DirectionAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    fund: Fund
    cutoff: date = Field(ge=date(2021, 1, 1), le=date(2024, 12, 31))
    end: date
    available_at: date
    y: Literal[0, 1]
    future_return: str

    @model_validator(mode="after")
    def chronology(self) -> Self:
        if not self.cutoff < self.end <= self.available_at <= date(2024, 12, 31):
            raise ValueError("ANSWER_TIME_BOUNDARY")
        value = Decimal(self.future_return)
        if not value.is_finite() or self.y != int(value > 0):
            raise ValueError("ANSWER_DIRECTION_MISMATCH")
        return self

    @property
    def key(self) -> str:
        return f"{self.fund}:{self.cutoff}"
