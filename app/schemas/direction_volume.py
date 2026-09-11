"""量价实验输入：七项原特征、可选成交活跃度及一个事前确定的交互项。"""

from pydantic import Field, FiniteFloat

from app.schemas.direction_training import DirectionInput


class VolumeDirectionInput(DirectionInput):
    """日期和答案隔离沿用原契约，确切维数再由冻结分支核对。"""

    x: tuple[FiniteFloat, ...] = Field(min_length=7, max_length=9)
