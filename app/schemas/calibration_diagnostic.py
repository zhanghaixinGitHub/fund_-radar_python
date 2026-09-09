"""校准方向的解释性信息；它不是发布许可或数值有效性检查的替代。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, FiniteFloat


class CalibrationDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    slope: FiniteFloat
    status: Literal["FORWARD_MAPPING", "REVERSED_PENDING_VALIDATION", "CONSTANT_PENDING_VALIDATION"]
    message: str


class WindowCalibrationDiagnostic(CalibrationDiagnostic):
    window_id: str
