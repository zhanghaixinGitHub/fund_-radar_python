"""严格交易日窗口预览契约；只展示日期与完整性，不生成净值收益、特征或模型标签。"""

from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TradingNavWindowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    fund_code: Literal["001632", "006730", "008888"] = Field(alias="fundCode", description="暂限已核验的三只股票型试点")
    cutoff_date: date = Field(
        alias="cutoffDate", description="站在该自然日结束时看已公告信息，不是旧asOfDate净值业务日"
    )


class CalendarProvenance(BaseModel):
    version: str = Field(description="独立日历版本，不是旧样本或模型版本")
    content_hash: str = Field(description="规范化日历定义SHA256")
    market: str = Field(description="沪深A股市场日历，不是申赎日历或境外市场日历")
    coverage_start: date = Field(description="已核验覆盖起日")
    coverage_end: date = Field(description="已核验覆盖止日；超出不自动延伸")
    reviewed_on: date = Field(description="本地核对官方休市事实的日期，不是历史首次可得日期")
    construction: str = Field(description="官方节假日区间加周末休市；静态研究快照，不是实时交易状态")
    source_urls: tuple[str, ...] = Field(description="本窗口涉及年份的沪深官方公告链接；运行时不联网抓取")
    future_schedule_known_at_cutoff: bool = Field(
        description="未来窗口所涉及年度安排是否都已在cutoff日公告；false表示用了事后已知的日历日期"
    )


class NavDateIssue(BaseModel):
    nav_date: date = Field(description="缺失或有问题的准确交易日，不会拿后一天代替")
    reason: Literal["MISSING_NAV", "MISSING_ANN_DATE", "ANN_BEFORE_NAV_DATE", "NOT_KNOWN_AT_CUTOFF"] = Field(
        description="只有日期质量，净值数值另待核验"
    )


class TradingNavWindowResponse(BaseModel):
    mode: Literal["READ_ONLY_TRADING_WINDOW_PREVIEW"] = Field(
        default="READ_ONLY_TRADING_WINDOW_PREVIEW", description="只读日期预览，不是样本保存或预测接口"
    )
    window_rule_version: Literal["TRADING_WINDOW_AFTER_CUTOFF_V1"] = Field(
        default="TRADING_WINDOW_AFTER_CUTOFF_V1", description="独立计数规则版本，不替换旧样本规则"
    )
    status: Literal["WINDOW_DATES_COMPLETE", "INPUT_DATES_INCOMPLETE", "FUTURE_DATES_INCOMPLETE"] = Field(
        description="日期窗口是否齐备，不表示可训练或可发布"
    )
    fund_code: str = Field(description="当前试点基金")
    cutoff_date: date = Field(description="信息可得截止日，按当天结束解释")
    source_code: str = Field(description="当前启用净值来源，非日历来源")
    source_sync_run_id: UUID = Field(description="当前成功同步水位；不证明历史首次版本")
    calendar: CalendarProvenance = Field(description="计数日历及来源证据")
    anchor_nav_date: date | None = Field(description="截止时已公告的最新交易日净值所属日；周末/节假日净值不作起点")
    anchor_ann_date: date | None = Field(description="所选起点的公告日，必须不晚于cutoff")
    anchor_lag_sessions: int | None = Field(description="起点比截至cutoff的最近交易日落后几场；最多允许1场，超出拒收")
    anchor_issue: str | None = Field(description="没有可得起点或起点过旧；不偷偷回退后宣称正常")
    history_dates: tuple[date, ...] = Field(description="截至起点必须连续对应的61个交易日；支持未来计算60区间指标")
    history_issues: tuple[NavDateIssue, ...] = Field(description="逐个要求日期的缺失/公告问题；缺一天不向前多找一天")
    future_dates: tuple[date, ...] = Field(description="严格晚于cutoff的第1至20个交易日；cutoff当天不算第1日")
    future_end_date: date = Field(description="未来第20个交易日，不根据净值缺失顺延")
    future_issues: tuple[NavDateIssue, ...] = Field(
        description="未来要求日期是否有源记录和合法公告；这是离线质量诊断，不放进模型输入"
    )
    future_navs_available_at: date | None = Field(description="未来20日公告齐全时最晚公告日；不是标签，没有计算收益")
    ignored_non_trading_nav_dates: tuple[date, ...] = Field(
        description="本读取范围的非交易日净值，保留源记录但不占窗口位置"
    )
    anchor_based_20th_trading_date: date | None = Field(
        description="仅对照：若从净值业务日而非cutoff起算，日历第20日落在哪天"
    )
    legacy_20th_nav_date_from_anchor: date | None = Field(
        description="仅对照旧记录计数，在当前读取范围中从起点后数第20条；不足则null，不额外扩查"
    )
    database_written: Literal[False] = Field(default=False, description="固定false：接口不写数据库")
    feature_generated: Literal[False] = Field(default=False, description="固定false：没有生成模型输入指标")
    label_generated: Literal[False] = Field(default=False, description="固定false：没有计算后来涨跌答案")
    nav_values_verified: Literal[False] = Field(default=False, description="固定false：没有核验净值金额与分红口径")
    training_eligible: Literal[False] = Field(default=False, description="固定false：日期齐全不等于已获训练资格")
    publication_status: Literal["MODEL_NOT_RELEASED"] = Field(default="MODEL_NOT_RELEASED", description="尚未发布模型")
    limitations: tuple[str, ...] = Field(description="数值/复权/首次可得/市场适用及非发布限制")
