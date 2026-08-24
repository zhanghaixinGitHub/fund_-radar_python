"""Pydantic contracts for M0 mock fund read models."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class InternalFundSummary(BaseModel):
    """Minimal public-safe fund data passed to the Java core service."""

    model_config = ConfigDict(frozen=True)

    fund_code: str
    fund_name: str
    fund_type: str
    status: str
    as_of_date: date


class InternalFundDetail(InternalFundSummary):
    """Detailed M0 mock fund read model with data freshness metadata."""

    nav_status: str
    data_source: str


class InternalFundPage(BaseModel):
    """Cursor-compatible page returned to the Java core service."""

    model_config = ConfigDict(frozen=True)

    items: tuple[InternalFundSummary, ...]
    next_cursor: str | None = Field(default=None, serialization_alias="next_cursor")
