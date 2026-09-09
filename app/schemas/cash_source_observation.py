"""准入的本地留档诊断，只返回计数/时刻，不把写入日志变成首次公开证明。"""

from datetime import date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CashObservationCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    fund_code: Literal["001632", "006730", "008888"] = Field(alias="fundCode", description="固定三只现金研究试点")
    start_date: date = Field(
        alias="startDate",
        ge=date(2021, 1, 1),
        le=date(2024, 12, 31),
        description="检查源值业务日期的开始日，含输入回看所需2021年",
    )
    end_date: date = Field(
        alias="endDate",
        ge=date(2021, 1, 1),
        le=date(2024, 12, 31),
        description="检查源值业务日期的结束日，不读取2025数值",
    )
    cutoff_date: date = Field(
        alias="cutoffDate",
        ge=date(2022, 1, 1),
        le=date(2024, 12, 31),
        description="要核对的研究截点，按上海自然日结束理解，不修改真实留档时间",
    )

    @model_validator(mode="after")
    def date_order(self) -> Self:
        if not self.start_date <= self.end_date <= self.cutoff_date:
            raise ValueError("source date range must not extend past research cutoff")
        return self


class CashObservationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["READ_ONLY_LOCAL_SOURCE_OBSERVATION_CHECK"] = "READ_ONLY_LOCAL_SOURCE_OBSERVATION_CHECK"
    request: CashObservationCheckRequest = Field(description="实际核对的基金、源日期范围和研究截点")
    checked_at: datetime = Field(description="本次诊断的数据库时间")
    status: Literal["NO_LOCAL_RECORDS", "ONLY_AFTER_CUTOFF", "LOCAL_WRITE_RECORDS_ONLY"] = Field(
        description="没有留档、只有截点后的留档、或有截点前写入记录；均非正式准入"
    )
    active_record_count: int = Field(ge=0, description="范围内未过保留期的本地变更记录数，不是净值天数或完整事件数")
    recorded_before_cutoff_count: int = Field(
        ge=0, description="数据库写入时间不晚于研究截点日结束的记录数，不代表提交可见性或首次公告"
    )
    expired_record_count: int = Field(ge=0, description="范围内已过来源保留期的记录数；不作为有效证据")
    first_observed_at: datetime | None = Field(description="范围内未过期记录的最早本地写入时间；没有则null")
    historical_first_versions_verified: Literal[False] = Field(
        default=False, description="此日志本身不能证明供应商首次公开版本"
    )
    event_completeness_verified: Literal[False] = Field(
        default=False, description="零条分红日志不是没有分红/拆分的证明"
    )
    training_eligible: Literal[False] = False
    publication_allowed: Literal[False] = False
    database_written: Literal[False] = False
    source_payload_read: Literal[False] = Field(
        default=False, description="只聚合元数据，不选择前后净值、分红金额或2025答案正文"
    )

    @model_validator(mode="after")
    def truthful_status(self) -> Self:
        if self.recorded_before_cutoff_count > self.active_record_count:
            raise ValueError("before-cutoff count exceeds active observations")
        expected = (
            "NO_LOCAL_RECORDS"
            if not self.active_record_count
            else "ONLY_AFTER_CUTOFF"
            if not self.recorded_before_cutoff_count
            else "LOCAL_WRITE_RECORDS_ONLY"
        )
        if self.status != expected or (self.first_observed_at is None) != (self.active_record_count == 0):
            raise ValueError("observation summary contradicts counts")
        return self
