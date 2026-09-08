"""历史样本保存与查询契约；调用者只提供范围，不接受外部传入的特征或答案。"""

from datetime import date, datetime, timedelta, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_serializer

from app.schemas.historical_nav import HistoricalNavBatchPreviewRequest
from app.services.historical_nav_samples import HistoricalNavSample


class HistoricalNavBatchSaveRequest(HistoricalNavBatchPreviewRequest):
    """沿用单基金/最多31自然日的参数；请求凭证必填，重试不得更换。"""

    request_key: UUID = Field(alias="requestKey", description="本次保存的唯一凭证；重试沿用，明确重算换新UUID")
    page_size: int = Field(
        default=10,
        alias="pageSize",
        ge=1,
        le=30,
        strict=True,
        description="内部每页起点数，只接受整数，不影响请求身份和样本内容",
    )


class HistoricalNavStoredBatch(BaseModel):
    """完整已保存批次；新建、重试与GET读回的JSON结构和内容相同。"""

    mode: Literal["STORED"] = Field(default="STORED", description="已保存快照，不是模型预测")
    batch_id: UUID = Field(description="已保存批次编号，用于GET查询")
    request_key: UUID = Field(description="保存请求凭证，识别重复请求")
    fund_code: str = Field(description="整批所属基金代码")
    fund_type: Literal["STOCK"] = Field(description="保存时的基金品类，固定股票型")
    start_date: date = Field(description="样本起点范围首日，含当天")
    end_date: date = Field(description="样本起点范围末日，含当天")
    source_code: str = Field(description="本批数据来源编码快照")
    source_sync_run_id: UUID | None = Field(description="本批来源同步水位，不是历史首次可得证明")
    feature_version: str = Field(description="指标计算规则版本")
    sample_rule_version: str = Field(description="样本筛选规则版本")
    label_version: str = Field(description="离线答案规则版本")
    purpose: Literal["LEARNING_ONLY"] = Field(description="仅供学习，未取得模型发布资格")
    sample_count: int = Field(ge=0, le=31, description="完整、拒收、答案未齐样本的总数")
    scorable_count: int = Field(ge=0, description="特征和答案均可用的数量")
    data_insufficient_count: int = Field(ge=0, description="数据不合格的数量")
    label_not_matured_count: int = Field(ge=0, description="特征已具备但答案未齐的数量")
    unavailable_reasons: dict[str, int] = Field(description="原因代码到对应数量的汇总")
    created_at: datetime = Field(description="批次保存时间，输出上海时区，不是净值公告日")
    items: tuple[HistoricalNavSample, ...] = Field(max_length=31, description="日期递增的样本，与单日预览结构相同")

    @field_serializer("created_at", when_used="json")
    def serialize_created_at(self, value: datetime) -> str:
        """数据库存带时区时间，接口统一输出+08:00，避免本机时区影响展示。"""
        if value.tzinfo is None:
            raise ValueError("stored created_at must include timezone")
        return value.astimezone(timezone(timedelta(hours=8))).isoformat()
