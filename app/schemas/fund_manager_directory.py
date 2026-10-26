"""经理目录公共契约；统计缺项允许为空，不把资料不足转换成零。"""

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ManagerCoverage(BaseModel):
    scopeLabel: str = "已收录任职资料"
    sourceScanComplete: bool = False
    identityComplete: bool = False
    productMappingComplete: bool = False
    pendingShareCount: int = 0
    lastSuccessfulCheckAt: datetime | None = None
    notice: str | None = None


class ManagerSummary(BaseModel):
    managerId: UUID
    managerName: str
    snapshotId: UUID
    companies: list[str] = Field(default_factory=list)
    education: str | None = None
    shareCount: int | None = None
    productCount: int | None = None
    historyCount: int | None = None
    anchorFundCode: str
    anchorFundName: str
    checkedAt: datetime | None = None


class ManagerPerformance(BaseModel):
    """所有收益和回撤都为小数比例；前端仅格式化一次。"""

    status: str = "UNAVAILABLE"
    returnRate: Decimal | None = None
    maxDrawdownRate: Decimal | None = None
    annualizedRate: Decimal | None = None
    startDate: date | None = None
    endDate: date | None = None
    includesCoManagement: bool = False
    calculationBasis: str = "CASH_REINVESTED_NAV_V1"
    reason: str | None = "任期表现资料暂不完整"


class ManagerFund(BaseModel):
    relatedFundId: UUID
    tenureVersionId: UUID
    fundCode: str
    fundName: str
    fundType: str
    companyName: str | None = None
    mappedFundCode: str | None = None
    state: str
    beginDate: date | None = None
    endDate: date | None = None
    teamStatus: str = "UNKNOWN"
    performance: ManagerPerformance = Field(default_factory=ManagerPerformance)
    soloPerformance: ManagerPerformance | None = None
    sourceUrl: str | None = None


class ManagerSource(BaseModel):
    title: str
    url: str
    publishedDate: date | None = None


class ManagerDetail(ManagerSummary):
    coverage: ManagerCoverage
    professionalResume: str | None = None
    profilePublishedOn: date | None = None
    sources: list[ManagerSource] = Field(default_factory=list)
    disclosures: list[ManagerSource] = Field(default_factory=list)
    funds: list[ManagerFund] = Field(default_factory=list)
    stale: bool = False


class ManagerPage(BaseModel):
    items: list[ManagerSummary]
    page: int
    pageSize: int
    totalCount: int
    totalPages: int
    snapshotId: UUID | None = None
    notice: str = "仅展示已收录资料，不代表全市场基金经理。"
    stale: bool = False


class ManagerFundPage(BaseModel):
    items: list[ManagerFund]
    page: int
    pageSize: int
    totalCount: int
    totalPages: int
    snapshotId: UUID


class ManagerSyncRequest(BaseModel):
    """管理员只能指定已登记人员；不接受客户端文件路径或来源URL。"""

    model_config = ConfigDict(extra="forbid")
    managerIds: list[UUID] = Field(default_factory=list, max_length=20)
    requestId: UUID
