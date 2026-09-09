"""由 AI 服务维护的 SQLAlchemy 领域模型。"""

from app.models.analysis import (
    AnalysisModelRelease,
    BacktestRun,
    FeatureSnapshot,
    ForecastResult,
)
from app.models.benchmark import BenchmarkNavDaily, BenchmarkSeries
from app.models.cash_forecast import CashForecastRecord
from app.models.cash_policy_freeze import CashPolicyFreezeRecord
from app.models.cash_prediction_attempt import CashPredictionAttemptRecord
from app.models.cash_reinvestment import CashResearchRun, CashSampleBatch, CashSampleLabel, CashSampleRecord
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
    SourceSyncCursor,
    SourceSyncRun,
)
from app.models.historical_nav_sample import (
    HistoricalNavSampleBatch,
    HistoricalNavSampleLabel,
    HistoricalNavSampleRecord,
)
from app.models.market_reference import (
    FundExchangeDaily,
    IndexWeightSnapshot,
    MarketIndexCatalog,
    MarketIndexClassification,
)

__all__ = [
    "CashForecastRecord",
    "CashPolicyFreezeRecord",
    "CashPredictionAttemptRecord",
    "CashResearchRun",
    "CashSampleBatch",
    "CashSampleLabel",
    "CashSampleRecord",
    "BacktestRun",
    "BenchmarkNavDaily",
    "BenchmarkSeries",
    "AnalysisModelRelease",
    "EventRelation",
    "FeatureSnapshot",
    "ForecastResult",
    "FundDividend",
    "FundExchangeDaily",
    "FundManagerAssignment",
    "FundMaster",
    "FundProfile",
    "FundShareClass",
    "FundShareSnapshot",
    "HistoricalNavSampleBatch",
    "HistoricalNavSampleLabel",
    "HistoricalNavSampleRecord",
    "MarketEvent",
    "MarketIndexCatalog",
    "MarketIndexClassification",
    "NavDaily",
    "NewsItem",
    "NewsSourceReference",
    "SourceRegistry",
    "SourceSyncCursor",
    "SourceSyncRun",
    "IndexWeightSnapshot",
]
