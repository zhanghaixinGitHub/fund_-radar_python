"""System response schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class InternalHealthResponse(BaseModel):
    """Authenticated internal service health response."""

    model_config = ConfigDict(frozen=True)

    service: str
    status: str
    time: datetime
