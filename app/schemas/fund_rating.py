"""对外评级白名单；展示正式分数，内部权重、方法号及研究点值不对外。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RatingSource(PublicModel):
    title: str
    url: str
    publishedOn: str | None = None


class RatingMetric(PublicModel):
    name: str
    value: str
    unit: str
    period: str


class RatingScoreRange(PublicModel):
    """当前已发布的分数范围；不是概率或未来损失区间。"""
    lower: str
    upper: str


class RatingCoverage(PublicModel):
    """原八维权重的点分、有效区间和未评分覆盖，均为百分数。"""
    pointWeight: str
    intervalWeight: str
    missingWeight: str
    pointDimensions: int
    intervalDimensions: int


class RatingDimension(PublicModel):
    key: str
    name: str
    grade: str | None = None
    gradeLabel: str | None = None
    # 同正式批次的0至100分，两位小数；暂无有效评分时为空，不补零。
    score: str | None = None
    scoreRange: RatingScoreRange | None = None
    explanation: str
    metrics: list[RatingMetric] = Field(default_factory=list)
    sources: list[RatingSource] = Field(default_factory=list)


class RatingSummary(PublicModel):
    fundCode: str
    status: Literal["RATED", "PARTIAL", "NOT_RATED", "UNSUPPORTED", "STALE", "NOT_FOUND", "WITHDRAWN"]
    grade: str | None = None
    gradeLabel: str | None = None
    asOfDate: str | None = None
    validUntil: str | None = None
    ratingRef: str | None = None
    message: str | None = None
    scoreRange: RatingScoreRange | None = None
    partialScore: str | None = None
    scoreCoverage: RatingCoverage | None = None


class RatingComparison(PublicModel):
    category: str
    currency: str | None = None
    productCount: int
    scope: str
    period: str


class RatingPoint(PublicModel):
    dimension: str
    text: str


class RatingDates(PublicModel):
    nav: str | None = None
    holdings: str | None = None
    manager: str | None = None
    fees: str | None = None


class RatingDetail(RatingSummary):
    # 仅基金详情展示正式综合分；市场列表摘要保持现有契约。
    score: str | None = None
    summary: str
    comparison: RatingComparison
    dimensions: list[RatingDimension]
    strengths: list[RatingPoint] = Field(default_factory=list)
    weaknesses: list[RatingPoint] = Field(default_factory=list)
    dataDates: RatingDates
    limitations: list[str]
    previousRating: RatingSummary | None = None


class RatingPage(PublicModel):
    items: list[RatingSummary]


class RatingSingleRequest(PublicModel):
    fundCode: str = Field(pattern=r"^[0-9]{6}$")
