"""内部1日入口只接收受限公共代码/作业编号，禁止用户身份和任意模型参数。"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

FundCode = Annotated[str, Field(pattern=r"^[0-9]{6}$")]


class Scope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fund_codes: list[FundCode] = Field(min_length=1, max_length=100)


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fund_code: FundCode


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssessmentAck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
