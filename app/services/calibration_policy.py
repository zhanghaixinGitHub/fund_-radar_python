"""有限参数可以计算；斜率符号只说明映射方向，不授予发布资格。"""

import math

from app.schemas.calibration_diagnostic import CalibrationDiagnostic


def calibration_diagnostic(slope: float, intercept: float) -> CalibrationDiagnostic:
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError("CALIBRATION_NONFINITE")
    if slope < 0:
        status, message = "REVERSED_PENDING_VALIDATION", "反向校准，需要独立数据验证；不表示算法失败或模型合格。"
    elif slope == 0:
        status, message = "CONSTANT_PENDING_VALIDATION", "常数校准，未利用模型分数区分涨跌；按固定概率对照验证。"
    else:
        status, message = "FORWARD_MAPPING", "正向校准，仍须通过独立验证和发布检查。"
    return CalibrationDiagnostic(slope=slope, status=status, message=message)
