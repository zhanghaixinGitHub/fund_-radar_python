"""T05—T09的显式研究队列，旧三基金API契约保持不变。"""

from datetime import date
from decimal import Decimal
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class StudyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    fund: str = Field(pattern=r"^\d{6}$")
    cutoff: date = Field(ge=date(2021, 1, 1), le=date(2024, 12, 31))
    anchor: date
    available_at: date
    anchor_lag_sessions: int = Field(ge=0, le=1)
    x: tuple[FiniteFloat, ...] = Field(min_length=7, max_length=10)
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def chronology(self) -> Self:
        if not date(2021, 1, 1) <= self.anchor <= self.available_at <= self.cutoff or len(self.x) not in (7, 10):
            raise ValueError("INPUT_BOUNDARY")
        return self

    @property
    def key(self):
        return f"{self.fund}:{self.cutoff}"


class StudyAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    fund: str = Field(pattern=r"^\d{6}$")
    cutoff: date = Field(ge=date(2021, 1, 1), le=date(2024, 12, 31))
    base: date
    end: date
    available_at: date
    y: Literal[0, 1]
    future_return: str

    @model_validator(mode="after")
    def chronology(self) -> Self:
        from app.services.trading_calendar import load_calendar

        calendar = load_calendar()
        if not self.base <= self.cutoff < self.end <= self.available_at <= date(2024, 12, 31):
            raise ValueError("ANSWER_BOUNDARY")
        if (
            self.base != calendar.sessions[calendar.at_or_before_index(self.cutoff)]
            or self.end != calendar.future_sessions(self.cutoff)[-1]
        ):
            raise ValueError("ANSWER_NOT_TWENTY_SESSIONS")
        value = Decimal(self.future_return)
        if not value.is_finite() or self.y != int(value > 0):
            raise ValueError("ANSWER_DIRECTION")
        return self


class StudyCohort(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["DIRECTION_COHORT_V1"] = "DIRECTION_COHORT_V1"
    original: tuple[Literal["001632", "006730", "008888"], ...] = ("001632", "006730", "008888")
    funds: tuple[str, ...] = Field(min_length=3, max_length=10)

    @model_validator(mode="after")
    def identities(self) -> Self:
        import re

        if self.original != ("001632", "006730", "008888") or not set(self.original).issubset(self.funds):
            raise ValueError("ORIGINAL_COHORT_CHANGED")
        if len(set(self.funds)) != len(self.funds) or any(not re.fullmatch(r"\d{6}", f) for f in self.funds):
            raise ValueError("COHORT_CODES")
        return self
