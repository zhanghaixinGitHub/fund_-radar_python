"""由 AI 服务维护的 SQLAlchemy 领域模型。"""

from app.models.analysis import (
    AnalysisExplanationSnapshot,
    AnalysisModelRelease,
    AnalysisRun,
    BacktestRun,
    FeatureSnapshot,
    ForecastResult,
)
from app.models.benchmark import BenchmarkNavDaily, BenchmarkSeries
from app.models.event import EventRelation, MarketEvent, NewsItem, NewsSourceReference
from app.models.fund import (
    FundDividend,
    FundManagerAssignment,
    FundMaster,
    FundProfile,
    FundShareClass,
    FundShareSnapshot,
    NavDaily,
    SourceRegistry,
    SourceSyncRun,
)

__all__ = [
    "BacktestRun",
    "BenchmarkNavDaily",
    "BenchmarkSeries",
    "AnalysisModelRelease",
    "AnalysisExplanationSnapshot",
    "AnalysisRun",
    "EventRelation",
    "FeatureSnapshot",
    "ForecastResult",
    "FundDividend",
    "FundManagerAssignment",
    "FundMaster",
    "FundProfile",
    "FundShareClass",
    "FundShareSnapshot",
    "MarketEvent",
    "NavDaily",
    "NewsItem",
    "NewsSourceReference",
    "SourceRegistry",
    "SourceSyncRun",
]
