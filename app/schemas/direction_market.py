"""匹配市场实验专用输入；旧七特征及页面输入契约保持原样。"""

from typing import Self

from pydantic import Field, FiniteFloat, model_validator

from app.schemas.direction_training import DirectionInput


class MarketDirectionInput(DirectionInput):
    """仍限制在2021—2024年，只在原七项后附加恰好三项市场特征。"""

    x: tuple[FiniteFloat, ...] = Field(min_length=7, max_length=10)

    @model_validator(mode="after")
    def dimensions(self) -> Self:
        if len(self.x) not in (7, 10):
            raise ValueError("MARKET_INPUT_DIMENSIONS")
        return self
