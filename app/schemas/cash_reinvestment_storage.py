"""保存的是一份当时的试跑快照；读回不会重新查询原净值。"""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.cash_reinvestment_batch import CashBatchRequest, CashBatchResponse


class CashBatchSaveRequest(CashBatchRequest):
    request_key: UUID = Field(alias="requestKey", description="同一保存操作重试沿用；重新生成换新UUID")
    page_size: int = Field(
        default=10, alias="pageSize", ge=1, le=30, strict=True, description="内部页大小，不属于请求身份"
    )


class CashStoredBatch(BaseModel):
    mode: Literal["STORED_CASH_REINVESTMENT_BATCH"] = "STORED_CASH_REINVESTMENT_BATCH"
    batch_id: UUID = Field(description="保存后编号，GET按它原样读回")
    request_key: UUID = Field(description="这份批次的重试凭证")
    created_at: datetime = Field(description="带时区的实际保存时刻")
    database_written: Literal[True] = Field(default=True, description="这份批次已保存，不表示每次GET都写库")
    preview: CashBatchResponse = Field(
        description="当时原计算快照；内部database_written=false描述计算器行为，不否认外层已保存"
    )
