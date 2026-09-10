"""实验推理独立契约；分数永不投影到正式上涨概率字段。"""

from datetime import date, datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExperimentModelScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    branch: Literal["DROP_60D_GROUP_L2", "REFERENCE"]
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    direction: Literal["UP", "NON_UP"]
    model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_end: date

    @model_validator(mode="after")
    def consistent_direction(self) -> Self:
        if self.direction != ("UP" if self.score > 0.5 else "NON_UP"):
            raise ValueError("EXPERIMENT_DIRECTION_MISMATCH")
        return self


class DirectionExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fund_code: str = Field(pattern=r"^\d{6}$")
    version: Literal["DIRECTION_PAGE_TRIAL_V1"] = "DIRECTION_PAGE_TRIAL_V1"
    status: Literal["EXPERIMENTAL", "DATA_INSUFFICIENT", "NOT_APPLICABLE", "UNAVAILABLE"]
    model_released: Literal[False] = False
    horizon_trading_days: Literal[20] = 20
    research_run_id: UUID = UUID("0c0e06a9-725e-4b68-b813-de6ff5124b29")
    cutoff_date: date | None = None
    latest_nav_date: date | None = None
    target_base_date: date | None = None
    target_end_date: date | None = None
    read_at: datetime
    input_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_revision_id: UUID | None = None
    models: tuple[ExperimentModelScore, ...] = Field(default=(), max_length=2)
    reason_codes: tuple[str, ...] = Field(default=(), max_length=12)
    message: str = Field(max_length=300)

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        if self.status != "EXPERIMENTAL":
            if self.models or not self.reason_codes:
                raise ValueError("UNAVAILABLE_EXPERIMENT_HAS_SCORES")
            return self
        if (
            tuple(m.branch for m in self.models) != ("DROP_60D_GROUP_L2", "REFERENCE")
            or self.input_hash is None
            or self.source_revision_id is None
            or self.cutoff_date is None
            or self.latest_nav_date is None
            or self.target_end_date is None
            or not self.latest_nav_date < self.cutoff_date == self.target_base_date < self.target_end_date
            or any(m.fit_end >= self.latest_nav_date for m in self.models)
            or self.reason_codes
        ):
            raise ValueError("EXPERIMENT_IDENTITY_OR_DATES")
        return self
