"""对外评级白名单；内部原始分数、权重和方法号不属于查询契约。"""

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


class RatingDimension(PublicModel):
    key: str
    name: str
    grade: str | None = None
    gradeLabel: str | None = None
    explanation: str
    metrics: list[RatingMetric] = Field(default_factory=list)
    sources: list[RatingSource] = Field(default_factory=list)


class RatingSummary(PublicModel):
    fundCode: str
    status: Literal["RATED", "NOT_RATED", "UNSUPPORTED", "STALE", "NOT_FOUND", "WITHDRAWN"]
    grade: str | None = None
    gradeLabel: str | None = None
    asOfDate: str | None = None
    validUntil: str | None = None
    ratingRef: str | None = None
    message: str | None = None


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
