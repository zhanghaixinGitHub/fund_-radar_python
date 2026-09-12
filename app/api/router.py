"""汇总仅供 Java 核心服务访问的内部路由。"""

from fastapi import APIRouter

from app.api.routes.analysis import router as analysis_router
from app.api.routes.cash_reinvestment_batch import router as cash_reinvestment_batch_router
from app.api.routes.cash_reinvestment_samples import router as cash_reinvestment_samples_router
from app.api.routes.cash_reinvestment_storage import router as cash_reinvestment_storage_router
from app.api.routes.direction_1d import router as direction_1d_router
from app.api.routes.events import router as events_router
from app.api.routes.features import router as features_router
from app.api.routes.funds import router as funds_router
from app.api.routes.health import router as health_router
from app.api.routes.historical_nav_calibration import router as historical_nav_calibration_router
from app.api.routes.historical_nav_evaluation import router as historical_nav_evaluation_router
from app.api.routes.historical_nav_storage import router as historical_nav_storage_router
from app.api.routes.historical_nav_training import router as historical_nav_training_router
from app.api.routes.nav_basis_audit import router as nav_basis_audit_router
from app.api.routes.portfolio_advice import router as portfolio_advice_router
from app.api.routes.signals import router as signals_router
from app.api.routes.simulation_market import router as simulation_market_router
from app.api.routes.sources import router as sources_router
from app.api.routes.trading_nav_window import router as trading_nav_window_router
from app.api.routes.watchlist_prediction import router as watchlist_prediction_router

"""内部 API 根路由；由应用入口统一加上 `/internal/v1` 前缀。"""
api_router = APIRouter()
api_router.include_router(direction_1d_router, prefix="/direction-1d", tags=["direction-1d"])
api_router.include_router(portfolio_advice_router, prefix="/portfolio-advice", tags=["portfolio-advice"])
api_router.include_router(simulation_market_router, prefix="/simulation", tags=["simulation-market"])

# 服务健康检查：供 Java 核心服务确认 Python 服务可访问，并关联请求追踪标识。
api_router.include_router(health_router, tags=["system"])

# 基金资料：查询基金列表、详情、净值和份额历史、同类比较，并提供数据同步任务及进度查询。
api_router.include_router(funds_router, prefix="/funds", tags=["funds"])

# 基金相关事件：查询与指定基金相关、已审核且未过期的事件摘要。
api_router.include_router(events_router, prefix="/events", tags=["events"])

# 特征与样本预览：读取最新特征状态，单条或批量试算历史净值样本，不保存试算结果。
api_router.include_router(features_router, prefix="/features", tags=["features"])

# 历史样本存储：将历史净值样本按批次保存到数据库，并按批次编号读取已存结果。
api_router.include_router(historical_nav_storage_router, prefix="/features", tags=["features"])

# 历史样本评估：检查已保存样本是否足够用于研究，并评估基线方法，作为后续模型的比较标准。
api_router.include_router(historical_nav_evaluation_router, prefix="/features", tags=["features"])

# 候选模型训练：使用合格的历史样本执行固定训练实验，返回研究结果，不发布模型。
api_router.include_router(historical_nav_training_router, prefix="/features", tags=["features"])

# 模型校准研究：按时间隔离训练、校准和评估数据，检查概率校准与滚动评估效果，不激活模型。
api_router.include_router(historical_nav_calibration_router, prefix="/features", tags=["features"])

# 交易日窗口核验：预览样本所需的历史和未来交易日，检查对应净值是否齐全，不生成预测。
api_router.include_router(trading_nav_window_router, prefix="/features", tags=["features"])

# 净值口径核验：对照单位净值、来源复权净值及现金分红计算的收益，定位计算口径差异。
api_router.include_router(nav_basis_audit_router, prefix="/features", tags=["features"])

# 现金分红再投资样本：预览单只基金在指定日期的特征和未来实际收益标签，仅供研究，不落库。
api_router.include_router(cash_reinvestment_samples_router, prefix="/features", tags=["features"])

# 现金分红再投资批量预览：按日期范围试算研究样本，汇总可用样本和缺失原因，不落库。
api_router.include_router(cash_reinvestment_batch_router, prefix="/features", tags=["features"])

# 现金分红再投资研究管理：保存和读取样本批次、研究报告，并提供数据留档核验及研究准备检查。
api_router.include_router(cash_reinvestment_storage_router, prefix="/features", tags=["features"])

# 评分信号：读取已保存的基金评分及已发布模型的评分增量，不在查询时执行模型。
api_router.include_router(signals_router, prefix="/signals", tags=["signals"])

# 基金分析摘要：读取已保存的分析结论，供页面展示，不在查询时重新分析或评分。
api_router.include_router(analysis_router, prefix="/analysis", tags=["analysis"])

# 数据源诊断：查看已配置数据源的安全状态，不返回凭据、原始内容，也不触发外部采集。
api_router.include_router(sources_router, prefix="/sources", tags=["sources"])

# 关注基金预测：读取预测视图，管理发布规则冻结和资格检查，并在资格通过后生成、保存预测。
api_router.include_router(watchlist_prediction_router, prefix="/predictions", tags=["predictions"])
